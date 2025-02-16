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
sys.path.append('/home/yichuan/HKU/honest_llama')
# sys.path.append('../')

# import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, get_llama_hidden_states, tokenized_math_shepherd, convert_time, upload_batch_to_HF
from interveners import wrapper, Collector, ITI_Intervener
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
    parser.add_argument('--n_shot', type=int, default=8, help='number of examples given in the Input. Represeantions are ignored during classifying and intervention')
    parser.add_argument('--get_hidden_states', type=bool, default=True, help='get hidden states instead for training the probe') #added
    parser.add_argument('--split_num', type=int, default=1, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--local_save', type=bool, default=False, help='whether to save locally as a jsonl file or upload the dataset to the HF hub')
    parser.add_argument('--hf_batch_size', type=int, default=100, help='how many samples in one push to the HF, default to be 100.')
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'math_shepherd':
        # dataset = load_dataset("peiyi9979/Math-Shepherd")["train"]# [args.start_sample_idx:args.end_sample_idx] 太大了！
        dataset = f'./features/MATH-Shepherd_part_{args.split_num}.csv'
        formatter = tokenized_math_shepherd
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'./features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else:  #TODO:下面的variable会很大
        #prefix_len是interpret representations时需要忽视的Input部分（sys_prmompt和n-shots部分) 的token长度
        prefix_len, train_prompts, train_labels, validate_prompts, validate_labels = formatter(dataset, tokenizer,args.n_shot) # for math-shepherd, prompt = I+O (problem + steps so-far), already tokenized; label is binary
        assert (len(train_prompts) == len(train_labels)) and (len(validate_prompts) == len(validate_labels))
        print(f'The number of total steps in the training set: {len(train_prompts)}\nThe number of total steps in the validation set: {len(validate_prompts)}') #34467 (5k samples, ~7 steps per sample)



    
    collectors = []
    pv_config = []
    for layer_id, layer in enumerate(range(model.config.num_hidden_layers)): 
        if args.get_hidden_states: #只收集指定层的mlp output(会包含prompt吗？是不是没有head概念？)
            if layer == args.layer:
                print(f'\n^^^\nThe layer we look into: {layer_id}\n^^^\n')
                collector = Collector(multiplier=0, head=-1, prefix_len=prefix_len) #head=-1 to collect all head activations, multiplier doens't matter
                collectors.append(collector) #每层一个collector收集所有heads的states和actions (activations包含这俩)
                pv_config.append({
                    "component": f"model.layers[{layer}].post_attention_layernorm.output", #.mlp.output??
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

    if args.get_hidden_states:
        print("Getting hidden states...")
        all_collector_hs = []
        all_prompts = [train_prompts, validate_prompts]
        all_labels = [train_labels, validate_labels]
        hf_batch = []
        hf_path = f'Lo-Fi-gahara/classify_layer{args.layer}_split{args.split_num}'
        start_t = time.time()
        for is_validate, (prompt_set, label_set) in enumerate(zip(all_prompts, all_labels)):
            if not is_validate:
                save_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_classifier_train_set.jsonl'
                split = 'train'
            else:
                save_path = f'./features/{args.model_name}_{args.layer}_{args.dataset_name}_{args.split_num}k_classifier_validate_set.jsonl'
                split = 'validation'
            with open(save_path, "w") as f:

                for i  in tqdm(range(len(prompt_set))): #train_set / validate_set 这里每个sample是一个step
                    prompt = prompt_set[i]
                    label = label_set[i]
                    if label.strip() == '+':
                        label = 1
                    else:
                        label = 0
                    #output_hs shape: torch.Size([161, 4096])
                    gen_t = time.time()
                    output_hs, collector_hs, _ = get_llama_activations_pyvene(args.layer, tokenizer, collected_model, collectors, prompt, device) #specify the layer to reduce storage
                    gen_spent = convert_time(gen_t) #~3s per sample
                    # print(f'\n***\nTime spent for a single generation: {gen_spent[0]}:{gen_spent[1]}:{gen_spent[2]}\n***\n')
                    if i == 0:    
                        print(f'output_hs shape: {output_hs.shape}') # torch.Size(I+O_length, 4096)
                        # print(f'collector_hs shape: {collector_hs.shape}') # (65, 4096)
                    h_l = { 
                        'hidden_states': output_hs.copy().tolist(),
                        'label': label
                    }
                    # all_collector_hs.append(collector_hs.copy())
                    if args.local_save:
                        f.write(json.dumps(h_l) + "\n")
                    else:
                        hf_batch.append(h_l)
                        if (i+1) % args.hf_batch_size == 0 or i+1 == len(prompt_set): #batch is full or the last batch
                            batch_name = f'sample{i+1-len(hf_batch)}-{i}'
                            upload_batch_to_HF(hf_batch, hf_path, split=split, batch_name=batch_name)
                            hf_batch.clear()
                    # if i == 4:
                    #     break#################

                assert os.path.exists(save_path)
                print(f"Building dataset is completed. Saved to {save_path}")
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
