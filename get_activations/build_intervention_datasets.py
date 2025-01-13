
'''
Build training adn validation datasets for the interventio module / PEFT
Build in an on-the-fly mode, each sample will be uploaded to HuggingFace after building
'''
# Pyvene method of getting activations
import os
import torch
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
import time
sys.path.append('/h/382/momoka/HKU/honest_llama')
# sys.path.append('../')

# import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_step_reps_pyvene, tokenized_math_shepherd_4_intervene, tokenized_dpo_4_intervene
from interveners import wrapper, Collector
import pyvene as pv

print(f'The current working directory: {os.getcwd()}')

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.1_8b_instruct': "meta-llama/Llama-3.1-8B-Instruct"
}

# '''限制intervention module输入的step hidden states长度为100'''
# def pad(array, fix_len=100):
#     if array.shape[0] > 100: #truncate the step length to be exa
#         array = array[:fix_len, :]
#     pad_len = fix_len - array.shape[0]
#     paddings = np.zeros((pad_len, array.shape[1]), dtype=torch.float32).to(device)
#     padded_array = np.concatenate((array, paddings), axis=0)
#     return padded_array

# def convert2JSONserializable(data_dict):
#     for key in data_dict:
#         if key != 'labels':
#             h_list = data_dict[key]
#             for h_id in range(len(h_list)):
#                 h_list[h_id] = [float(ele) for ele in h_list[h_id]]

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--n_shot', type=int, default=8, help='The number of examples in the Input prompt')
    parser.add_argument('--get_hidden_states', type=bool, default=True, help='get hidden states instead for training the probe') #added
    parser.add_argument('--split_num', type=int, default=5, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--local_save', type=bool, default=False, help='set True to save locally, otherwise upload to the HF dataset. Default False.')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    '''
    1. load and tokenize dataset
    '''
    if args.dataset_name == 'math_shepherd':
        # dataset = load_dataset("peiyi9979/Math-Shepherd")["train"]# [args.start_sample_idx:args.end_sample_idx] 太大了！
        dataset = f'./features/MATH-Shepherd_part_{args.split_num}.csv'
        formatter = tokenized_math_shepherd_4_intervene
    elif args.dataset_name == 'dpo':
        print('\nHERE!!!!\n') #################
        dataset = f'./features/241127-1.json'
        formatter = tokenized_dpo_4_intervene
    else: 
        raise ValueError("Invalid dataset name")

    print("\nTokenizing prompts...\n")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'./features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else:  #TODO:下面的variable会很大
        prefix_len, train_prompt_pairs, validate_prompt_pairs = formatter(dataset, tokenizer, args.n_shot) # for math-shepherd, prompt = I+O (problem + steps so-far), already tokenized; 2D arrays: samples[sample_steps[{till_curr, one_look_ahead}]]
        print(f'\nprefix_len: {prefix_len}\n')
        # assert (len(train_prompt_pairs) + args.n_shot) == 4*len(validate_prompt_pairs) #train:validate = 4:1############################
        print(f'train set size: {len(train_prompt_pairs)}, validate set size: {len(validate_prompt_pairs)}')
        train_steps_count = 0
        validate_steps_count = 0
        for sample in train_prompt_pairs:
            train_steps_count += len(sample)
        for sample in validate_prompt_pairs:
            validate_steps_count += len(sample)
        print(f' - Training set: total samples: {len(train_prompt_pairs)}, total steps: {train_steps_count}\n - Validation set: total samples: {len(validate_prompt_pairs)}, total steps: {validate_steps_count}') #34467 (5k samples, ~7 steps per sample)


    '''
    2. build pv model
    '''
    collectors = []
    pv_config = []
    for layer_id, layer in enumerate(range(model.config.num_hidden_layers)): 
        if args.get_hidden_states: #只收集指定层的mlp output(会包含prompt吗？是不是没有head概念？)
            if layer == args.layer:
                print(f'\n^^^\nThe layer we look into: {layer_id}\n^^^\n')
                collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
                collectors.append(collector) #每层一个collector收集所有heads的states和actions (activations包含这俩)
                pv_config.append({
                    "component": f"model.layers[{layer}].post_attention_layernorm.output", #.mlp.output 现在收集的是transformer layer的output, 对应representations
                    "intervention": wrapper(collector), #“收集”作为intervention function
                })
            else: 
                continue
        else:
            collector = Collector(multiplier=0, head=-1)
            collectors.append(collector) 
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input", # hidden states? torch.Size([1, 156, 4096])
                "intervention": wrapper(collector), #让collector可以接受*args, **kwargs? 伪intervention??
            })
    collected_model = pv.IntervenableModel(pv_config, model) #subclass of nn.Module

    '''
    3. Get sample hidden states and upload
    '''
    if args.get_hidden_states:
        print("\nGetting hidden states\n...")
        start_t = time.time()
        for is_validate, all_sample_pairs in enumerate([train_prompt_pairs, validate_prompt_pairs]): #train / valiate set
            all_samples = []
            if not is_validate:
                save_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_train_set.jsonl'
            else:
                save_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_validation_set.jsonl'
            if not is_validate: #train set
                print('Now build training set...')
                split='train'
                dir_name = "train/"
            else:
                print('Now build validation set...')
                split='validation'
                dir_name = "validate/"
            error_count = 0
            for i, sample  in enumerate(tqdm(all_sample_pairs)): #遍历sample
                # Upload the training and validation sets in a sample-by-sample base!
                labels = []
                h_priors = []
                h_posteriors = []
                for step in sample:
                    if args.dataset_name == 'math_shepherd':
                        label = step['label']
                        prompt_curr = step['till_curr_step']
                        prompt_lookahead = step['lookahead_step']
                        curr_step_len = step['curr_step_len']
                    elif args.dataset_name == 'dpo':
                        label = None
                        prompt_curr = step['chosen_prompt_curr']
                        post_step_len = step['chosen_step_len']
                        prompt_lookahead = step['rejected_prompt_curr']
                        pre_step_len = step['rejected_step_len']


                    gen_hs_1, original_hs, _, input_len = get_llama_step_reps_pyvene(tokenizer, collected_model, collectors, prefix_len, prompt_curr, device)
                    gen_hs_2, intervened_hs, curr_step_range, input_len = get_llama_step_reps_pyvene(tokenizer, collected_model, collectors, prefix_len, prompt_lookahead, device)
                    
                    if args.dataset_name == 'math_shepherd':
                        hs = gen_hs_1[-curr_step_len:].tolist() #ground truth hidden states
                        hs_prime = gen_hs_2[curr_step_range[0]:curr_step_range[1]].tolist()
                    elif args.dataset_name == 'dpo':
                        hs = gen_hs_1[input_len-post_step_len:input_len]
                        hs_prime = gen_hs_2[input_len-pre_step_len:input_len]
                    # padded_hs = pad(hs)
                    # padded_hs_prime = pad(hs_prime)
                        if i == 0:    
                            print(f'gen_hs_1 shape: {gen_hs_1.shape}, gen_hs_2 shape: {gen_hs_2.shape}') # 
                            print(f'hs shape: {hs.shape}, hs_prime shape: {hs_prime.shape}') 
                        hs = hs.tolist()
                        hs_prime = hs_prime.tolist()
                        # print(f'after padding, hs shape: {padded_hs.shape}, hs_prime shape: {padded_hs_prime.shape}')
                        # print(f'collector_hs shape: {collector_hs.shape}') # (65, 4096)
                    h_priors.append(hs_prime)
                    h_posteriors.append(hs)
                    labels.append(label)
                    # if i == 4:
                    #     break#################
                if i == 0:
                    print(f'\n\nlabels: {labels}\n\n')
                sample_dict = {
                    'h_prior': h_priors,
                    'h_posterior': h_posteriors,
                    'labels': labels
                }
                if h_priors == [] or h_posteriors == []:
                    print('\n!!!\nError: h_priors or h_posteriors is empty!! Skip.\n!!!\n')
                    error_count += 1
                    continue
                sample_data = Dataset.from_dict(sample_dict)
                if not args.local_save:
                    if args.dataset_name == 'math_shepherd':
                        sample_data.push_to_hub(f'Lo-Fi-gahara/intervene_{args.split_num}k', split=split, max_shard_size="1GB", data_dir=f'{dir_name}sample{i}') #upload a single sample to HF
                    elif args.dataset_name == 'dpo':
                        print(f'pushing sample {i} to the HF Hub...')
                        sample_data.push_to_pub(f'Lo-Fi-gahara/intervene_dpo')
                else: ###################
                    # convert2JSONserializable(sample_dict)
                    all_samples.append(sample_dict)

                if i == 10:
                    break #####################



            if args.local_save:###################
                with open(save_path, 'w') as f:
                    for sample in all_samples:
                        json.dump(sample, f)
                        f.write('\n')
                print(f'\nDataset saved to {save_path}\n')
            else: #########################
                print(f'Dataset uploaded to HF.\n')
            print(f'Total error encountered: {error_count}')

        spent = time.time() - start_t
        hrs = int(spent // 3600)
        spent %= 3600
        mins = int(spent // 60)
        secs = int(spent % 60)
        print(f'\n***\nTotal time spent on converting 1k samples: {hrs}:{mins}:{secs}\n***\n')

        
    else:
        all_layer_wise_activations = []
        all_head_wise_activations = []

        print("Getting activations")
        for prompt in tqdm(prompts):
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
            all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
            all_head_wise_activations.append(head_wise_activations.copy())

        print("Saving labels")
        np.save(f'./features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

        print("Saving layer wise activations")
        np.save(f'./features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
        
        print("Saving head wise activations")
        np.save(f'./features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
