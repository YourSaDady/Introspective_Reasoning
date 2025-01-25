'''
on-the-fly版本
    - freezed mode: 收集hiddens用I+O训练的probe同时处理多种input hiddens
    - speified mode: 只收集指定的一种hiddens type, 先用split1 split2和5epochs训练, 再用split3进行同样hiddens type的analysis

1. featured with: 
- models: llama3.1-8b-instruct vs. mistral-7b-sft
- mode: online (probe right after the full hiddens are collected) vs. offline (hiddens are collected and uploaded to HF first)
- labeling method: build-from-scratch (non-labeled) vs. math_shepherd (middle steps are labeled)
- train set and validation set
- indexed: sample idx, step idx
- texted: input, steps, answers
...


2. format:
{
    'sample_id': int,
    'input_text': str,
    'input_hiddens': torch.Size(len, 4096),
    'output': [
        {
            'step_id': int,
            'step_text': str,
            'step_hiddens': torch.Size(len, 4096),
            'step_label': bool,
        }
    ],
    'answer_text': str,
    'answer_label': bool,
}

'''

import os
import torch
from torch.utils.data import DataLoader, Dataset
import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
import time
sys.path.append('/home/yichuan/HKU/honest_llama')
os.chdir('/home/yichuan/HKU/honest_llama')
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# Specific pyvene imports
from utils import load_math_shepherd, build_nshot_examples, probing_analysis, convert_time, online_training
import pyvene as pv
# crychic imports
from probes import Classifier
from interveners import Collector, wrapper

print(f'The current working directory: {os.getcwd()}')

HF_NAMES = {
    'llama3.1_8b_instruct': {'hf_path': "meta-llama/Llama-3.1-8B-Instruct", 'use_template': True},
    'mistral_7b_sft': {'hf_path': 'peiyi9979/mistral-7b-sft', 'use_template': False},
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--probe_path', type=str, default='./trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_-1_pos_ans.json') #./trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_I+O.json
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--split_num', type=int, default=3) #default: use the split3 for analysis, and split1 and 2 for prpbes training
    parser.add_argument('--sample_num', type=int, default=1000)
    parser.add_argument('--n_shot', type=int, default=8)
    parser.add_argument('--online', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='freezed', help='choose from {"freezed", "specified"}. "freezed" use a probe trained on I+O for all hiddens types; "specified" trains diffierent probes for different hidden types')
    parser.add_argument('--online_training', type=bool, default=False)
    # parser.add_argument('--specified_type', type=str, default='I+O')
    parser.add_argument('--build_from_scratch', type=bool, default=False)
    args = parser.parse_args()

    #___hyper params___
    sample_num = args.sample_num
    hiddens_range = [0, -1, -2, -3, 8848] #hiddens types
    trained_probes_paths = [
        './trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_I_pos_ans.json',
        './trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_-3_pos_ans.json',
        './trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_-2_pos_ans.json',
        './trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_-1_pos_ans.json',
        './trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_I+O_pos_ans.json'
    ]
    #__________________

    #0. load dataset
    print('\nLoading dataset...\n')
    if args.online: #online, build nshot first
        formatter = build_nshot_examples
    else:
        formatter = load_math_shepherd

    result, others = formatter(args.dataset_name, args.split_num, args.n_shot)  


    #1. load model and probe(s)
    print('\nLoading model and probe...\n')
    model_name_or_path = HF_NAMES[args.model_name]['hf_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # print(f"\n___{args.model_name}'s architechture___\n")
    # print(model)
    # return ######################

    probes = []
    for _ in hiddens_range: #initialize probes
        probe = Classifier(
            'lstm',
            hidden_size=4096,
            output_size=1,
        )
        probe.to('cuda')
        probes.append(probe)
    print(f'\nInitialized {len(probes)} probes\n')

    #2. build pv model
    print(f'\nBuilding the pyvene collection model...\n')
    nshot = others['nshot']
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



    #3. generation
    print('\nStarting generation...\n')
    if args.online:
        '''
        hiddens_range
            - specifies what hiddens are collected as the input to the probe.
            - a list storing the hiddens indicies in ascending order
         
        Specificially:
        0: Input
        8848: Input + Output (full)
        1: I + Step1
        2: I + Step1 + Step2
        ...
        -1: I + O \ last step
        -2: I + O \ last two steps
        ...

        Here we use the settings from FackCheckMate Table 2 by default
        '''
        stat_path = f'./probing_survey/{args.model_name}_layer{args.layer}_{args.dataset_name}_split{args.split_num}_probing_stat_{args.mode}_test.jsonl' ########
        start_t = time.time()
        probbing_config = {
            'split_num': args.split_num,
            'n_shot': args.n_shot,
            'examples': nshot,
            'prefix_len': prefix_len,
            'stat_path': stat_path,
            'hiddens_range': hiddens_range,
            'use_template': use_template,
            'max_samples': sample_num,
        }
        #1. train probes online or load trained stat_dicts
        if args.mode == 'specified' and args.online_training:
            training_config = probbing_config
            training_config['stat_prefix'] = f'./trained_probes/{args.dataset_name}/{args.model_name}_layer{args.layer}'
            os.makedirs(training_config['stat_prefix'], exist_ok=True)

            trained_probes = online_training(args.dataset_name, tokenizer, collect_model, collector, probes, training_config)

        elif args.mode == 'specified' and not args.online_training:
            for idx, probe_path in enumerate(trained_probes_paths):
                if os.path.exists(probe_path):
                    probes[idx].load_state_dict(probe_path)
                    print(f'loaded trained {idx}th probe from {probe_path}')
                else:
                    print(f'\nThe specified probe path: {probe_path} does not exist, initialized a probe.\n')
            trained_probes = probes
        else: #freezed
            probe_path = args.probe_path
            if os.path.exists(probe_path):
                for idx in range(len(probes)):
                    probes[idx].load_state_dict(probe_path)
                trained_probes = probes
                print(f'loaded trained {idx}th probe from {probe_path}')
            else:
                print(f'\nThe specified probe path: {probe_path} does not exist, initialized a probe.\n')
        #2. probbing analysis
        probing_analysis(args.dataset_name, tokenizer, collect_model, collector, trained_probes, probbing_config)

        spent = convert_time(start_t)
        print(f'\nTotal time spent: {spent[0]}:{spent[1]}:{spent[2]}')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()