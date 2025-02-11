'''
Train intervention module online.
'''

import time
import json
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig
from datasets import Dataset, load_dataset

sys.path.append('/home/yichuan/HKU/honest_llama')
print(f'The current working directory: {os.getcwd()}')
import pyvene as pv
from probes import Classifier, InterventionModule
from interveners import Collector, wrapper
from utils import build_nshot_examples, build_assistant_prompt, build_user_prompt, find_subiter_range, convert_time, extract_q_s_l, extract_a
from templates import apply_template

HF_NAMES = {
    'llama3.1_8b_instruct': {'hf_path': "meta-llama/Llama-3.1-8B-Instruct", 'use_template': True},
    'mistral_7b_sft': {'hf_path': 'peiyi9979/mistral-7b-sft', 'use_template': False},
}

def pad(array, fix_len=256, dim=1): 
    if array.size(dim) > fix_len: #truncate the step length to be exa
        array = array[:fix_len, :]
    pad_len = fix_len - array.shape[0]
    try:
        paddings = torch.zeros((pad_len, array.shape[1]), dtype=torch.float32)
    except:
        print(f'\n!!!\nError: array has shape: {array.shape}, cannot be padded\n!!!\n')
        return torch.Tensor([-1]) 
    padded_array = torch.cat((array, paddings), axis=0)
    return padded_array

# find the range of the given keyward tokens in a list of tokens
def find_tokens_range(tokens_list, key_tokens):
    key_len = len(key_tokens)
    for i in range(len(tokens_list)-key_len):
        if tokens_list[i:i+key_len-2] == key_tokens[:-2]:
            return i, i+key_len
    print(f'cannnot find key tokens!')
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--probe_type', type=str, default='itv', help='choose from {cls, itv}')
    parser.add_argument('--max_samples', type=int, default=1000, help='the number of samples used in each split')
    parser.add_argument('--wrong_only', type=bool, default=False, help='whether to do intervention based on the generated wrong solutions only. Default to be false')
    args = parser.parse_args()
    #______hyper params______
    device='cuda'
    n_shot = 8
    lr = 0.001
    epochs = 5
    split_nums = [1, 2]
    max_samples = args.max_samples
    stat_prefix = f'./trained_probes/{args.dataset_name}/{args.model_name}_layer{args.layer}/{args.probe_type}'
    # prior_path = f'{stat_prefix}/intervener_test.pth'
    if args.wrong_only:
        probe_path = f'{stat_prefix}/intervener_{len(split_nums)}k_wrong_only.pth'##############
    else:
        probe_path = f'{stat_prefix}/intervener_{len(split_nums)}k_all.pth'
    # probe_path = f'{stat_prefix}/intervener_test.pth'##############
    prior_path = probe_path
    if not os.path.exists(stat_prefix):
        os.mkdir(stat_prefix)
        print(f'{stat_prefix} just created.')
    if args.wrong_only:
        training_stat_path = f'{stat_prefix}/running_losses_log_{len(split_nums)}k_wrong_only.jsonl' 
    else:
        training_stat_path = f'{stat_prefix}/running_losses_log_{len(split_nums)}k_all.jsonl' 
    # training_stat_path = f'{stat_prefix}/running_losses_log_test.jsonl' ##############
    #________________________
    start_t = time.time()

    '''
    1. Prepare n-shot prefix
    '''
    if args.dataset_name == 'math_shepherd':
        _, others = build_nshot_examples(args.dataset_name, 1, 8)  
        nshot = others['nshot']
    else:
        raise NotImplementedError
    
    
    '''
    2. Load model and probe(s)
    '''
    print('\nLoading model and probe...\n')
    model_name_or_path = HF_NAMES[args.model_name]['hf_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    probe = InterventionModule(
        'lstm',
        depth=3, ############
    )
    probe.to('cuda')
    print(f'\nInitialized probe\n')

    '''
    3. Build pv model
    '''
    print(f'\nBuilding the pyvene collection model...\n')
    use_template = HF_NAMES[args.model_name]['use_template']
    if use_template:
        msg_dict = {
            'sys': '',
            'user': nshot,
            'assistant': '',
            'remove_last_eot': True,
        }
        prefix = apply_template(args.dataset_name, msg_dict)
        # chat = [{
        #     'role': 'user',
        #     'content': nshot,
        # }]
        # prefix = tokenizer.apply_chat_template(chat, tokenize=False).rstrip('<|eot_id|>')
        prefix_len = tokenizer(prefix, return_tensors='pt').input_ids[0].size(0)
    else:
        prefix_len = tokenizer(nshot, return_tensors='pt').input_ids[0].size(0)
    # print(f'\nprefix_len: {prefix_len}\n') 
    collector = Collector(multiplier=0, head=-1, prefix_len=prefix_len)
    pv_config = pv.IntervenableConfig(
        representations=[{
            'layer': args.layer,
            "component": f"model.layers[{args.layer}].post_attention_layernorm.output", #.mlp.output??
            "low_rank_dimension": 1,
            "intervention": wrapper(collector)
        }]
    )
    collect_model = pv.IntervenableModel(pv_config, model)
    # collect_model.set_device('cuda')

    '''
    4. Train and Evaluate
    '''
    early_break = False #################
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(probe.params, lr=lr)
    with open(training_stat_path, 'w') as statfile:
        for split_num in split_nums:
            for epoch in range(epochs):
                print(f'\n===\nNow start split {split_num}, epoch {epoch}\n===\n')
                if os.path.exists(prior_path):
                    print(f'\nFound trained stat_dict from {prior_path}!\n')
                    state_dict = torch.load(prior_path)
                    probe.load_state_dict(state_dict)
                    print(f'loaded state_dict...')
                if max_samples != -1:
                    pbar = tqdm(total=max_samples, desc='sample')
                # print(f'\nmax_samples: {max_samples}\n')
                running_loss = 0.0
                if args.dataset_name == 'math_shepherd':
                    csv_path = f'./features/MATH-Shepherd_part_{split_num}.csv'
                else:
                    raise NotImplementedError
                with open(csv_path, newline='') as csvfile:
                    sample_count = 0
                    csvreader = csv.DictReader(csvfile)
                    for sample_id, row in enumerate(csvreader):
                        if max_samples != -1 and sample_id == (max_samples+n_shot):
                            break
                        # print(f'\n  - sample_id: {sample_id} / max_samples: {max_samples}\n') 
                        # line_count += 1
                        if sample_id < n_shot:
                            continue
                        if args.dataset_name == 'math_shepherd':
                            label = row['label']
                            question, steps, labels = extract_q_s_l(label)
                            answer = extract_a(label)
                            sample_label = 1 #math-shepherd only has samples with correct final answers
                        else:
                            raise NotImplementedError
                        sample_count += 1
                        pbar.update(1)
                        gen_config = GenerationConfig(
                            max_new_tokens=500,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                        )
                        msg_dict = {
                            'sys': '',
                            'user': build_user_prompt(question, nshot),
                            'remove_last_eot': True,
                        }
                        '''
                        Prepare hs (complete ground truth I+O) and h's (I + model generate)
                        '''
                        hiddens_len = 0
                        intervene_range = {}
                        hiddens = {} # prior and posterior, both of shape Size(1, seq_len, 4096)
                        # tokens = {}
                        for msg_id in range(2):
                            # print(f'\n___\nmsg_id: {msg_id}\n___\n')
                            if msg_id == 0:
                                # print(f'len(steps)-1: {len(steps)-1}')
                                msg_dict['assistant'] = build_assistant_prompt(steps, len(steps)-1) #complete steps
                            else:
                                msg_dict['assistant'] = build_assistant_prompt(steps, 0, for_inference=True) #input question only
                            prompt = apply_template(args.dataset_name, msg_dict)
                            prompt_len = tokenizer(prompt, return_tensors='pt').input_ids[0].size(0)
                            tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
                            # print(f'\nfull_prompt:\n_____\n{prompt}\n______\n')
                            _, output_dict = collect_model.generate(
                                tokenized_prompt, 
                                generation_config = gen_config,
                                unit_locations=None,      # set to None means intervention will be applied for each forward call
                                intervene_on_prompt=True, # intervention will be called for the prompt kv cache call
                                subspaces=[{"logging": False}], # other metadata
                            )
                            #prepare hiddens and tokens
                            io_hiddens = torch.cat(collector.states, dim=1) #full I+O len
                            o_hiddens = torch.cat([base[:, -1:, :] for base in collector.states], dim=1)
                            # print(f"io_hiddens.shape: {io_hiddens.shape}, collector.states[0].size(1): {collector.states[0].size(1)}")
                            if msg_id == 0: # I+O
                                decoded_io_tokens = [tokenizer.decode(token) for token in tokenized_prompt.input_ids[0][prefix_len:]]
                                ass_start, ass_end = find_tokens_range(decoded_io_tokens, ['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n', 'Answer'])
                                o_tokens = tokenized_prompt.input_ids[0][prefix_len+ass_end:]

                                curr_step_tokens = tokenizer(steps[-1], return_tensors='pt').input_ids.tolist()[0][1:] #discard the first token '<|begin_of_text|>' (128000)
                                curr_start, curr_end = find_subiter_range(o_tokens.tolist(), curr_step_tokens)
                                hiddens['posterior'] = io_hiddens[:, :curr_end, :].to(torch.float32) #ground_truth
                                tokens = o_tokens[:curr_end] #含input question
                                collector.reset()
                            else: #input only
                                decoded_gen = [tokenizer.decode(token) for token in output_dict.sequences[0].tolist()[prefix_len:]]
                                # print(f'decoded generated tokens list: {decoded_gen}')
                                ass_start, ass_end = find_tokens_range(decoded_gen, ['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n', 'Answer', ':', ' \n\n'])
                                gen_tokens = output_dict.sequences[0].tolist()[prefix_len+ass_end:]
                                # hiddens['prior'] = o_hiddens[:, :, :].to(torch.float32)
                                hiddens['prior'] = io_hiddens[:, ass_end-1:, :].to(torch.float32)
                                tokens = gen_tokens
                                collector.reset()
                                decoded_solution = tokenizer.decode(tokens)
                                pred_ans = extract_a(decoded_solution, src='generated')
                                # print(f'\n{msg_id+1}th decoded_solution: {decoded_solution}\npred_ans: {pred_ans}\n')
                            
                        #end of generation type
                        # print(f"\nh' shape: {hiddens['prior'].shape}")
                        # print(f"\nh shape: {hiddens['posterior'].shape}")
                        #本想padding的。后来发现padded的话就不是残差了
                        # if not hiddens['prior'].size(1) == hiddens['posterior'].size(1): # ends earlier than the groundtruth solution
                        #     # print(f"\nh' shape: {hiddens['prior'].shape}")
                        #     # print(f"h shape: {hiddens['posterior'].shape}\n")
                        #     if hiddens['prior'].size(1) < hiddens['posterior'].size(1):
                        #         pad_len = hiddens['posterior'].size(1) - hiddens['prior'].size(1)
                        #         paddings = torch.zeros((hiddens['prior'].size(0), pad_len, hiddens['prior'].size(2)), dtype=torch.float32).to(device)
                        #         hiddens['prior'] = torch.cat([hiddens['prior'], paddings], dim=1)

                        # print(f"\nprior len: {hiddens['prior'].size(1)}, posterior len: {hiddens['posterior'].size(1)}")
                        #if intervene on wrong solutions only, skip the correct solutions
                        # print(f'\nanswer: {answer}, pred_ans: {pred_ans}, Acc: {pred_ans.strip() == answer}')
                        if args.wrong_only and pred_ans != None and pred_ans.strip() == answer: 
                            print('correct!!')
                            continue
                        '''
                        calculate loss per token
                        '''
                        sample_loss = 0.0
                        itv_range = min(hiddens['prior'].size(1), hiddens['posterior'].size(1))
                        for itv_pos in range(itv_range):
                            optimizer.zero_grad()
                            itv_input = torch.cat([hiddens['posterior'][:, :itv_pos, :], hiddens['prior'][:, itv_pos:itv_pos+1, :]], dim=1)
                            itv_output = probe(itv_input)
                            assert itv_output.shape == itv_input.shape
                            # if itv_pos == intervene_range['start']+10:
                            #     print(f'itv_input shape: {itv_input.shape}, itv_output.shape: {itv_output.shape}') #torch.Size([1, 43, 4096])
                            label = hiddens['posterior'][-1][itv_pos] #Size(4096)
                            pred = itv_output[-1][-1] #Size(4096)
                            pred = hiddens['prior'][-1][itv_pos] + F.normalize(pred, p=2, dim=0) #L2 normalize the pred before add it back to the raw tensor
                            loss = loss_func(pred, label)
                            sample_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        #end of token iter
                        # print(f'sample_loss before: {sample_loss}, hiddens_len: {hiddens_len}')
                        sample_loss /= itv_range
                        # print(f'sample_loss after: {sample_loss}')
                        running_loss += sample_loss
                        # if True:
                        if (sample_id+1-n_shot) % 50 == 49:
                            print(f'Epoch [{epoch+1}/{epochs}], Sample [{sample_count}/{max_samples}], avg loss [{running_loss / sample_count:.5f}], last sample loss [{sample_loss:.5f}]')
                            loss_stat = {
                                'split': split_num,
                                'epoch': epoch,
                                'samples': sample_id,
                                'loss': round(running_loss / sample_count, 4), 
                            }
                            statfile.write(json.dumps(loss_stat)+'\n')
                    #end of sample iter
                #end of csv reading
                if early_break:
                    break
                '''
                5. Save the trained intervener (update the state_dict)
                '''
                state_dict = probe.state_dict()
                torch.save(state_dict, probe_path)
                print(f'trained probe saved to {probe_path}')
            #end of epoch
            if early_break:
                break
        #end of split
    #end of stat writing
    '''
    5. Save the trained intervener
    '''
    state_dict = probe.state_dict()
    torch.save(state_dict, probe_path)
    print(f'trained probe saved to {probe_path}')
                            

    print('Finished Training')
    spent = convert_time(start_t)
    print(f'Time spent on loading dataset: {spent[0]}:{spent[1]}:{spent[2]}')


if __name__ == '__main__':
    main()