'''
Evaluate with different decoding method:
    - pretrained (baseline): original step-by-step decoding
    - crychic: step-wise, classify + intervene decoding
    - intervene (peft): token-wise, intervene only
    - ... (TBD)

Metrics:
    - Total Accuracy
    - ... (TBD) 

'''
import os
import torch
import csv
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

# import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig

# Specific pyvene imports
from utils import build_nshot_examples, extract_q_s_l, extract_a, build_assistant_prompt, build_user_prompt, convert_time
import pyvene as pv

from probes import Classifier, InterventionModule
from interveners import ClassifyIntervene, PEFTIntervene
from templates import apply_template

print(f'The current working directory: {os.getcwd()}')

HF_NAMES = {
    'llama3.1_8b_instruct': {'hf_path': "meta-llama/Llama-3.1-8B-Instruct", 'use_template': True},
    'qwen2.5_math_7b': {'hf_path': 'Qwen/Qwen2.5-Math-7B', 'use_template': True},
    'deepseek_r1_distill_qwen_7b': {'hfpath': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'use_template': True},
    'llama3.1-70b': {'hfpath': 'meta-llama/Llama-3.1-70B', 'use_template': False}
}

def main(): 
    # #check whether a model has a chat_template
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
    # print(f'template: ------\n{tokenizer.chat_template}\n-----')
    # return

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd', help='to regenerate the experiment result, choose from {gsm8k, math, prm800k, processbench}')
    parser.add_argument('--n_shot', type=int, default=8, help='number of examples given in the Input. Represeantions are ignored during classifying and intervention')
    parser.add_argument('--split_num', type=int, default=1, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--mode', type=str, default='crychic', help='choose from {pretrained, crychic, intervene}')
    parser.add_argument('--classifier_path', type=str, default='./trained_probes/math_shepherd/llama3.1_8b_instruct_layer16/probe_-1_pos_ans.json', help='state_dict (json) path to the classifier')
    parser.add_argument('--intervention_module_path', type=str, default='./trained_probes/interventor_lstm_10k_classify-True.pth', help='state_dict path to the intervention module')
    parser.add_argument('--compare_baseline', type=bool, default=True, help='whether compare with baseline. default to be True')
    parser.add_argument('--max_step_iters', type=int, default=15, help='the max number of reasoning steps per sample, default to be 15')
    args = parser.parse_args()

    '''
    0. Load model, classifier and intervention module 
    '''
    path_prefix = f'./trained_probes/{args.dataset_name}/{args.model_name}_layer{args.layer}'
    cls_path = f'{path_prefix}/probe_I+O_5k.json'
    itv_path = f'{path_prefix}/itv/intervener_2k_wrong_only.pth' #intervener_2k_wrong_only.pth
    device = 'cuda'
    print(f'loading tokenizer and model...')
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]['hf_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16) #, device_map="auto"
    model.to(device)
    print(f'loading classifier and intervention module...')
    if args.mode == 'crychic':
        classifier = Classifier(
            'lstm',
            hidden_size=4096,
            output_size=1,
        )
        if os.path.exists(cls_path):
            classifier.load_state_dict(cls_path)
            print(f'loaded exisiting classifier state_dict from {cls_path}.')
        else:
            raise ValueError(f'the classifier state_dict path "{cls_path}" does not exist!')
        classifier.to('cuda')
    if args.mode != 'pretrained':
        intervener = InterventionModule(
            'lstm',
            depth=3,
        )
        if os.path.exists(itv_path):
            state_dict = torch.load(itv_path)
            intervener.load_state_dict(state_dict)
            print(f'loaded exisiting intervener state_dict from {itv_path}.')
        else:
            raise ValueError(f'the intervener state_dict path "{itv_path}" does not exist!')
        intervener.to('cuda')

    '''
    1. Load n-shot examples
    '''
    _, others = build_nshot_examples(args.dataset_name, args.split_num, args.n_shot)
    examples = others['nshot']
    use_template = HF_NAMES[args.model_name]['use_template']
    if use_template:
        msg_dict = {
            'sys': '',
            'user': examples,
            'assistant': '',
            'remove_last_eot': True,
        }
        prefix = apply_template(args.dataset_name, msg_dict)
        prefix_len = tokenizer(prefix, return_tensors='pt').input_ids[0].size(0)
    else:
        prefix_len = tokenizer(examples, return_tensors='pt').input_ids[0].size(0)

    '''
    2. Build pyvene model
    '''
    if args.mode == 'crychic': #use cls and itv
        pv_module = ClassifyIntervene(classifier, intervener, prefix_len)
    elif args.mode == 'intervene':
        pv_module = PEFTIntervene(intervener)
    if args.mode != 'pretrained':
        pv_config = pv.IntervenableConfig(
            representations=[{
                "layer": args.layer,
                "component": f"model.layers[{args.layer}].post_attention_layernorm.output", #.mlp.output??
                "low_rank_dimension": 1,
                "intervention": pv_module, # a wrapper containing the clasifier and the intervention module
            }]
        )
        pv_model = pv.IntervenableModel(pv_config, model)
        # pv_model.set_device("cuda") 

    '''
    3. Generate and Evaluate
    '''
    if args.dataset_name == 'math_shepherd':
        csv_path = f'./features/MATH-Shepherd_part_{args.split_num}.csv'
    else:
        raise NotImplementedError
    max_step_num = args.max_step_iters
    stat_path = f'./evaluate/{args.mode}_wrong_only_split{args.split_num}_result.jsonl'
    n_shot = args.n_shot
    print('Start generation and evaluation...')
    start_t = time.time()
    sample_count = 0
    org_acc_count = 0
    itv_acc_count = 0
    with open(stat_path, 'w') as statfile:
        with open(csv_path, newline='') as csvfile:
            pbar = tqdm(total=args.max_samples, desc='sample')
            csvreader = csv.DictReader(csvfile)
            for sample_id, row in enumerate(csvreader):
                if sample_count == args.max_samples:
                    break
                # print(f'\n  - sample_id: {sample_id} / max_samples: {max_samples}\n') ###############
                # line_count += 1
                if sample_id < n_shot:
                    continue
                if args.dataset_name == 'math_shepherd':
                    label = row['label']
                    question, steps, labels = extract_q_s_l(label)
                    steps = [step for step in steps if step.strip()]
                    answer = extract_a(label)
                else:
                    raise NotImplementedError

                sample_count += 1
                pbar.update(1)

                solution = build_assistant_prompt(steps, len(steps)-1).strip('\n')
                stat = {
                    'sample_id': sample_id,
                    # 'label': labels[step_id],
                    'original_pred_ans': None,
                    'org_acc': None,
                    'intervened_pred_ans': None,
                    'intervened_acc': None,
                    'ans': answer,
                    'question': question,
                    # 'original_pred_sol'
                    # 'intervened_pred_sol'
                    'sol': solution,
                }
                # assistant_content = build_assistant_prompt(steps, len(steps)-1)
                if use_template:
                    msg_dict = {
                        'sys': '',
                        'user': build_user_prompt(question, examples),
                        'assistant': '', #assistant_content,
                        'remove_last_eot': True,
                    }
                    prompt = apply_template(args.dataset_name, msg_dict)
                else:
                    prompt = f'{examples}\n\nSolve the following Question step by step.\n\nQuestion: {question}\n\n{assistant_content}'
                tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
                gen_config = GenerationConfig(
                    max_new_tokens=500,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                if args.compare_baseline or args.mode == 'pretrained':
                    org_output_dict = model.generate(
                        tokenized_prompt['input_ids'],
                        gen_config,
                        tokenizer=tokenizer,
                        return_dict_in_generate=True,
                    )
                    org_sequence = tokenizer.decode(org_output_dict.sequences[0].tolist()) #skip_special_tokens=True
                    stat['org_pred_sol'] = org_sequence.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1]
                    stat['org_pred_ans'] = extract_a(stat['org_pred_sol'], src='generated') 
                    stat['org_acc'] = (stat['org_pred_ans'].strip() == stat['ans'].strip()) if stat['org_pred_ans'] else 0
                    org_acc_count +=  stat['org_acc']
                _, pv_output_dict = pv_model.generate(
                    tokenized_prompt, 
                    generation_config = gen_config,
                    unit_locations=None,      # set to None means intervention will be applied for each forward call
                    intervene_on_prompt=True, # intervention will be called for the prompt kv cache call
                    subspaces=[{"logging": False}], # other metadata
                )
                itv_sequence = tokenizer.decode(pv_output_dict.sequences[0].tolist()) #skip_special_tokens=True
                stat['itv_pred_sol'] = itv_sequence.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1]
                stat['itv_pred_ans'] = extract_a(stat['itv_pred_sol'], src='generated') 
                stat['itv_acc'] = (stat['itv_pred_ans'].strip() == stat['ans'].strip()) if stat['itv_pred_ans'] else 0
                itv_acc_count +=  stat['itv_acc']
                # if True: ####################
                if sample_count%10 == 0: ####################
                    print(f"\nsplit:[{args.split_num}], sample: [{sample_count}/{args.max_samples}], org_acc: {org_acc_count*100/sample_count:.4f}, itv_acc: {itv_acc_count*100/sample_count:.4f}")
                    statfile.write(json.dumps(stat)+'\n')
            #end of sample iter
            if args.compare_baseline or args.mode == 'pretrained':
                org_acc = round((org_acc_count/sample_count)*100, 4)
            if args.mode != 'pretrained':
                itv_acc = round((itv_acc_count)*100, 4)
            print(f'\nTotal Acc: org {org_acc}%, itv {itv_acc}%')
            statfile.write('\n'+json.dumps({'org_acc': org_acc, 'itv_acc': itv_acc}))
        #end of csv read
    # end of stat write
    print(f'Statistics saved to {stat_path}')

    spent = convert_time(start_t)
    print(f'- Time spent on decoding: {spent[0]}:{spent[1]}:{spent[2]}, or {(time.time() - start_t):.5f} secs')


if __name__ == '__main__':
    main()
