'''
Train and Evaluate probes for different layers on-the-fly

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False) #
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False) #
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05) #
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
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

'''
online training / evaluation
args:
    - mode: {'train', 'eval'}
    - stat_path: path to save running losses or evaluation metrics
    - splits: data splits used for training
    - epochs: epochs used per split during training
    - max_samples
    - probes: a list of dict


'''
def all_layers_online(hyper_params):
    device = 'cuda'
    dataset_name = hyper_params['dataset_name']
    mode = hyper_params['mode']
    n_shot = hyper_params['n_shot']
    examples = hyper_params['examples']
    prefix_len = hyper_params['prefix_len']
    model = hyper_params['model']
    tokenizer = hyper_params['tokenizer']
    stat_path = hyper_params['stat_path']
    max_samples = hyper_params['max_samples']
    probes = hyper_params['probes']
    #_______________________________
    if mode == 'train':
        early_stop = False
        loss_func = nn.BCELoss()
        splits = hyper_params['splits']
        epochs = hyper_params['epochs']
    else:
        early_stop = False
        splits = hyper_params['splits']
        epochs = 1
    with open(stat_path, 'w') as statfile:
        for split_num in splits:
            for epoch in range(epochs):
                print(f'\n===\nNow start split {split_num}, epoch {epoch}\n===\n')
                for probe_dict in probes: #save and load state_dict for each epoch\
                    probe_dict['running_loss'] = 0.0 ###############
                    if os.path.exists(probe_dict['save_path']):
                        probe_dict['probe'].load_state_dict(probe_dict["save_path"])
                        if probe_dict['layer'] == 31:
                            print(f'Found trained state_dicts. Last stat_dict loaded from {probe_dict["save_path"]}!')
                pbar = tqdm(total=max_samples, desc='sample')

                if dataset_name == 'math_shepherd':
                    csv_path = f'./features/MATH-Shepherd_part_{split_num}.csv'
                else:
                    raise NotImplementedError
                with open(csv_path, newline='') as csvfile:
                    sample_count = 0
                    csvreader = csv.DictReader(csvfile)
                    step_count = 0
                    for sample_id, row in enumerate(csvreader):
                        if max_samples != -1 and sample_id == (max_samples+n_shot):
                            break
                        # print(f'\n  - sample_id: {sample_id} / max_samples: {max_samples}\n') ###############
                        # line_count += 1
                        if sample_id < n_shot:
                            continue
                        if dataset_name == 'math_shepherd':
                            label = row['label']
                            question, steps, labels = extract_q_s_l(label)
                            steps = [step.strip(' \n') for step in steps if step.strip(' \n')]
                            answer = extract_a(label)
                            sample_label = 1 #math-shepherd only has samples with correct final answers
                        else:
                            raise NotImplementedError
                        sample_count += 1
                        pbar.update(1)
                        gen_config = GenerationConfig(
                            max_new_tokens=10,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                        )
                        for step_id, step in enumerate(steps):
                            step_count += 1
                            assistant_content = build_assistant_prompt(steps, step_id)
                            msg_dict = {
                                'sys': '',
                                'user': build_user_prompt(question, examples),
                                'assistant': assistant_content,
                                'remove_last_eot': True,
                            }
                            prompt = apply_template(dataset_name, msg_dict)
                            tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
                            _, step_output_dict = model.generate(
                                tokenized_prompt, 
                                generation_config = gen_config,
                                unit_locations=None,      # set to None means intervention will be applied for each forward call
                                intervene_on_prompt=True, # intervention will be called for the prompt kv cache call
                                subspaces=[{"logging": False}]*len(probes), # other metadata
                            )
                            '''
                            collect hiddens and propagate for each layer
                            '''
                            io_tokens = tokenized_prompt.input_ids[0][prefix_len:] #Input part only
                            curr_step_tokens = tokenizer(step, return_tensors='pt').input_ids.tolist()[0][1:] #discard the first token '<|begin_of_text|>' (128000)
                            curr_start, curr_end = find_subiter_range(io_tokens.tolist(), curr_step_tokens)
                            step_losses = [0.0]*len(probes)###############
                            for probe_id, probe_dict in enumerate(probes):
                                # io_hiddens = torch.cat(probe_dict['collector'].states, dim=1).squeeze(0)
                                io_hiddens = probe_dict['collector'].states[0].squeeze(0)
                                # if not io_hiddens.size(0) >= curr_end:
                                #     print(f'steps: {steps}')
                                #     print(f'step_id: {step_id}')
                                #     print(f'io_hiddens.size(0): {io_hiddens.size(0)}, curr_end: {curr_end}')
                                #     print(f'current_step tokens: "{tokenizer.decode(curr_step_tokens)}"')
                                #     print(f'io_tokens: \n++++++++++++\n{tokenizer.decode(io_tokens)}\n++++++++++++\n')
                                #     raise AssertionError
                                io_hiddens = io_hiddens[:curr_end, :]
                                probe_dict['collector'].reset()
            
                                if mode == 'train':
                                    label = torch.tensor([int(labels[step_id] == '+')], dtype=torch.float32).to(device) 
                                    probe_dict['optimizer'].zero_grad()
                                    pred_logits = probe_dict['probe'](io_hiddens)
                                    loss = loss_func(pred_logits, label)
                                    step_losses[probe_id] += loss.item() #############
                                    probe_dict['running_loss'] += loss.item()
                                    # probe_dict['loss_count'] += 1
                                    loss.backward()
                                    probe_dict['optimizer'].step()
                                else:
                                    with torch.no_grad():
                                        label = labels[step_id]
                                        pred = probe_dict['probe'](io_hiddens).item()
                                        pred = '+' if pred >= 0.5 else '-' #boolean output
                                    probe_dict['acc_count'] += int(pred == label)
                            #end of probe iter
                            # print(f'\nStep{step_id}(step_count: {step_count}), acc_stat: \n{acc_stat}\n') ######################
                            # break

                            # if mode == 'train':
                            #     losses = [round((probe_dict['running_loss']/step_count), 4) for probe_dict in probes]
                            #     print(f'\nsample_id: {sample_id}, step_id: {step_id}')
                            #     print(f'step losses: {step_losses[:5]}')
                            #     print(f'running_losses: {losses[:5]}')
                            #     print(f"step_count: {step_count}, probe_dict['loss_count']: {probes[0]['loss_count']}\n")
                        #end of step iter
                        # if True: ######################
                        if (sample_count % 100 == 0): #no early break!!
                            if mode == 'train':
                                losses = [round((probe_dict['running_loss']/step_count), 4) for probe_dict in probes]
                                print(f'\nsplit:[{split_num}], epoch:[{epoch+1}/{epochs}], sample: [{sample_count}/{max_samples}]')
                                print(f'running_losses: {losses[:2]}...{losses[-2:]}\n')
                                loss_stat = {
                                    'split': split_num,
                                    'epoch': epoch,
                                    'samples': sample_count,
                                    'running_losses': losses, #[round(loss/(sample_count+1), 4) for loss in running_losses]
                                }
                                statfile.write(json.dumps(loss_stat)+'\n')
                            else:
                                acc_stat = []
                                for probe_dict in probes:
                                    stat = {
                                        'layer': probe_dict['layer'],
                                        'component': probe_dict['component'],
                                        'Acc': round((probe_dict['acc_count']/step_count)*100, 4),
                                    }
                                    acc_stat.append(stat)
                                print(f'\nsplit:[{split_num}], epoch:[{epoch+1}/{epochs}], sample: [{sample_count}/{max_samples}], steps: {step_count}')
                                print(f'Acc: {acc_stat[:2]}...{acc_stat[-2:]}\n')
                    #end of sample iter
                    if mode == 'train':
                        for probe_dict in probes:
                            probe_dict['running_loss'] /= sample_count
                            state_dict = probe_dict['probe'].state_dict()
                            with open(probe_dict['save_path'], 'w') as f:
                                json.dump(state_dict, f)
                                if probe_dict['probe_id'] == len(probes)-1:
                                    print(f"all the trained probes saved. Last probe saved to {probe_dict['save_path']}")
                    
                #end of csv read
                if early_stop:
                    break
            #end of epoch
            if early_stop:
                break
        #end of split
        if mode == 'eval':
            # print(f'\n(step_count: {step_count}), acc_stat: \n{acc_stat}\n') ######################
            acc_stat = []
            for probe_dict in probes:
                stat = {
                    'layer': probe_dict['layer'],
                    'component': probe_dict['component'],
                    'Acc': round((probe_dict['acc_count']/step_count)*100, 4),
                }
                statfile.write(json.dumps(stat)+'\n')
                acc_stat.append(stat)
            last = {'sample num': max_samples, 'step num': step_count}
            statfile.write(json.dumps(last))
            print(f'Total Acc: {acc_stat[:2]}...{acc_stat[-2:]}, evaluation stat saved to {stat_path}.')
    #end of stat write

    if mode == 'train':
        return probes
    else:
        print(f'Finished evaluation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--layer_start', type=int, default=0, help='the starting layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--layer_end', type=int, default=31, help='the ending layer of the model to access the stat vars')
    parser.add_argument('--probe_type', type=str, default='lstm')
    parser.add_argument('--max_samples', type=int, default=1000, help='the number of samples used in each split')
    args = parser.parse_args()
    #______hyper params______
    n_shot = 8
    layer_start = args.layer_start
    layer_end = args.layer_end #inclusive
    probe_type = args.probe_type
    max_samples = args.max_samples #for train & evaluate

    train_splits = [1, 2]
    eval_split = [3]
    epochs = 5
    lr = 0.001

    stat_prefix = f'./trained_probes/{args.dataset_name}/{args.model_name}_full_layers/cls'
    if not os.path.exists(stat_prefix):
        os.makedirs(stat_prefix, exist_ok=True)
        print(f'{stat_prefix} just created.')
    train_stat_path = f'{stat_prefix}/train_loss_{max_samples}.jsonl' 
    eval_stat_path = f'{stat_prefix}/eval_stat_{max_samples}.jsonl' ##############
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
    2. Load model and probes
    '''
    print('\nLoading model and probe...\n')
    model_name_or_path = HF_NAMES[args.model_name]['hf_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    probes = []
    probe_id = 0
    for layer_id in range(layer_start, layer_end+1):
        for component in ['post_attention_layernorm', 'self_attn.o_proj', 'mlp.down_proj']:
            probe = Classifier(
                probe_type,
                hidden_size=4096,
                output_size=1,
                aggregate_method='mean', #dummy
            )
            probe.to('cuda')
            probe_dict = {
                'probe_id': probe_id,
                'layer': layer_id,
                'component': component,
                'probe': probe,
                'optimizer': optim.AdamW(probe.params, lr=lr),  
                'running_loss': 0.0,
                'loss_count': 0,
                'acc_count': 0,
            }
            if component == 'post_attention_layernorm':
                probe_dict['save_path'] = f'{stat_prefix}/probe{probe_id}_layer{layer_id}_out.json'
            elif component == 'self_attn.o_proj':
                probe_dict['save_path'] = f'{stat_prefix}/probe{probe_id}_layer{layer_id}_attn.json'
            else:
                probe_dict['save_path'] = f'{stat_prefix}/probe{probe_id}_layer{layer_id}_mlp.json'
            probes.append(probe_dict)
            probe_id += 1
    print(f'\nInitialized {len(probes)} probes.\n')

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
        prefix_len = tokenizer(prefix, return_tensors='pt').input_ids[0].size(0)
    else:
        prefix_len = tokenizer(nshot, return_tensors='pt').input_ids[0].size(0)
    print(f'\nprefix_len: {prefix_len}\n')
    pv_config = []
    for probe_dict in probes: # each layer we collect three representations (layer output, attention, mlp)
        collector = Collector(multiplier=0, head=-1, prefix_len=prefix_len)
        probe_dict['collector'] = collector
        config_dict = {
            'layer': layer_id,
            "component": f"model.layers[{probe_dict['layer']}].{probe_dict['component']}.output", #.mlp.output??
            "low_rank_dimension": 1,
            "intervention": wrapper(collector),
        }
        pv_config.append(config_dict)
        

    collect_model = pv.IntervenableModel(pv_config, model)

    '''
    4. Train
    '''
    online_args = {
        'dataset_name': args.dataset_name,
        'model': collect_model,
        'tokenizer': tokenizer,
        'n_shot': n_shot,
        'examples': nshot,
        'prefix_len': prefix_len,
        'max_samples': max_samples,
    }

    print(f'\nNow start training...\n')
    train_start = time.time()
    train_args = online_args
    train_args['mode'] = 'train'
    train_args['stat_path'] = train_stat_path
    train_args['probes'] = probes
    train_args['epochs'] = epochs
    train_args['splits'] = train_splits

    trained_probes = all_layers_online(train_args)
    train_spent = convert_time(train_start)
    print(f'\nFinished training. Total time spent: {train_spent[0]}:{train_spent[1]}:{train_spent[2]}\n')
                
    '''
    5. Evaluate
    '''
    # for probe_dict in probes:
    #     state_dict_path = f"./trained_probes/math_shepherd/llama3.1_8b_instruct_full_layers/cls/probe_{probe_dict['layer']}.json"
    #     probe_dict['probe'].load_state_dict(state_dict_path)
    # trained_probes = probes

    print(f'\nNow start evaluation...\n')
    eval_start = time.time()
    eval_args = online_args
    eval_args['mode'] = 'eval'
    eval_args['stat_path'] = eval_stat_path
    eval_args['probes'] = trained_probes
    eval_args['epochs'] = 1
    eval_args['splits'] = eval_split

    all_layers_online(eval_args)
    eval_spent = convert_time(eval_start)
    print(f'\nFinished evaluation. Total time spent: {eval_spent[0]}:{eval_spent[1]}:{eval_spent[2]}\n偶妹得多')



if __name__ == '__main__':
    main()