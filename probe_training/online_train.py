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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--probe_type', type=str, default='itv', help='choose from {cls, itv}')
    parser.add_argument('--max_samples', type=int, default=1000, help='the number of samples used in each split')
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
    prior_path = ''
    # probe_path = f'{stat_prefix}/intervener_{len(split_nums)}k.pth'##############
    probe_path = f'{stat_prefix}/intervener_test.pth'##############
    # prior_path = probe_path
    if not os.path.exists(stat_prefix):
        os.mkdir(stat_prefix)
        print(f'{stat_prefix} just created.')
    # training_stat_path = f'{stat_prefix}/running_losses_log_{len(split_nums)}k_test.jsonl' ##############
    training_stat_path = f'{stat_prefix}/running_losses_log_test.jsonl' ##############
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
        depth=4, ############
    )
    probe.to('cuda')
    print(f'\nInitialized probe\n')

    '''
    3. Build pv model
    '''
    print(f'\nBuilding the pyvene collection model...\n')
    use_template = HF_NAMES[args.model_name]['use_template']
    if use_template:
        prefix = [{ #just for calculating the prefix len, the chat is dummy
            'role': 'user',
            'content': nshot,
        }]
        prefix_len = tokenizer.apply_chat_template(prefix, tokenize=True, return_tensors='pt').size(-1)
    else:
        prefix_len = tokenizer(nshot, return_tensors='pt').input_ids[0].size(0)
    print(f'\nprefix_len: {prefix_len}\n') ###########
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
    early_break = False
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
                        # print(f'\n  - sample_id: {sample_id} / max_samples: {max_samples}\n') ###############
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
                        for msg_id in range(2):
                            # print(f'\n___\nmsg_id: {msg_id}\n___\n')
                            if msg_id == 0:
                                # print(f'len(steps)-1: {len(steps)-1}')
                                msg_dict['assistant'] = build_assistant_prompt(steps, len(steps)-1) #complete steps
                            else:
                                msg_dict['assistant'] = build_assistant_prompt(steps, 0, for_inference=True) #input question only
                            prompt = apply_template(args.dataset_name, msg_dict)
                            tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
                            # print(f'\nfull_prompt:\n_____\n{prompt}\n______\n')
                            _, step_output_dict = collect_model.generate(
                                tokenized_prompt, 
                                generation_config = gen_config,
                                unit_locations=None,      # set to None means intervention will be applied for each forward call
                                intervene_on_prompt=True, # intervention will be called for the prompt kv cache call
                                subspaces=[{"logging": False}], # other metadata
                            )
                            #prepare hiddens and tokens
                            io_tokens = tokenized_prompt.input_ids[0][prefix_len:] #Input part only
                            decoded_io = tokenizer.decode(io_tokens)
                            # print(f'\ndecoded_io (len={len(io_tokens)}): \n++++++++++++++++\n{decoded_io}\n++++++++++++++++\n')
                            io_hiddens = torch.cat(collector.states, dim=1) #full I+O len
                            # print(f"io_hiddens.shape: {io_hiddens.shape}, collector.states[0].size(1): {collector.states[0].size(1)}")
                            if msg_id == 0: # I+O
                                curr_step_tokens = tokenizer(steps[-1], return_tensors='pt').input_ids.tolist()[0][1:] #discard the first token '<|begin_of_text|>' (128000)
                                curr_start, curr_end = find_subiter_range(io_tokens.tolist(), curr_step_tokens)
                                hiddens['posterior'] = io_hiddens[:, :curr_end, :].to(torch.float32) #ground_truth
                                collector.reset()
                                intervene_range['end'] = curr_end #intervene end
                            else: #input only
                                intervene_range['start'] = collector.states[0].size(1) #intervene start
                                # print(f'intervene_range: {intervene_range}')
                                hiddens_len = intervene_range['end'] - intervene_range['start']
                                hiddens['prior'] = io_hiddens[:, :intervene_range['end'], :].to(torch.float32)
                                collector.reset()
                        #end of generation type
                        # print(f"\nh' shape: {hiddens['prior'].shape}")
                        # print(f"\nh shape: {hiddens['posterior'].shape}")
                        if not hiddens['prior'].size(1) == hiddens['posterior'].size(1): # ends earlier than the groundtruth solution
                            # print(f"\nh' shape: {hiddens['prior'].shape}")
                            # print(f"h shape: {hiddens['posterior'].shape}\n")
                            if hiddens['prior'].size(1) < hiddens['posterior'].size(1):
                                pad_len = hiddens['posterior'].size(1) - hiddens['prior'].size(1)
                                paddings = torch.zeros((hiddens['prior'].size(0), pad_len, hiddens['prior'].size(2)), dtype=torch.float32).to(device)
                                hiddens['prior'] = torch.cat([hiddens['prior'], paddings], dim=1)

                        assert hiddens['prior'].size(1) == hiddens['posterior'].size(1)
                        '''
                        calculate loss per token
                        '''
                        sample_loss = 0.0
                        for itv_pos in range(intervene_range['start'], intervene_range['end']):
                            optimizer.zero_grad()
                            itv_input = torch.cat([hiddens['posterior'][:, :itv_pos, :], hiddens['prior'][:, itv_pos:itv_pos+1, :]], dim=1)
                            itv_output = probe(itv_input)
                            assert itv_output.shape == itv_input.shape
                            # if itv_pos == intervene_range['start']+10:
                            #     print(f'itv_input shape: {itv_input.shape}, itv_output.shape: {itv_output.shape}') #torch.Size([1, 43, 4096])
                            label = hiddens['posterior'][-1][itv_pos] #Size(4096)
                            pred = itv_output[-1][-1] #Size(4096)
                            loss = loss_func(pred, label)
                            sample_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        #end of token iter
                        # print(f'sample_loss before: {sample_loss}, hiddens_len: {hiddens_len}')
                        sample_loss /= hiddens_len
                        # print(f'sample_loss after: {sample_loss}')
                        running_loss += sample_loss
                        if True:
                        # if (sample_id+1-n_shot) % 50 == 49:
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
    # probe_path = f'{stat_prefix}/intervener_{len(split_nums)}.pth'
    probe_path = f'{stat_prefix}/intervener_test.pth' ##############
    state_dict = probe.state_dict()
    torch.save(state_dict, probe_path)
    print(f'trained probe saved to {probe_path}')
                            

    print('Finished Training')
    spent = convert_time(start_t)
    print(f'Time spent on loading dataset: {spent[0]}:{spent[1]}:{spent[2]}')


if __name__ == '__main__':
    main()