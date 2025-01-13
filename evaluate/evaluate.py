'''
Evaluate the classifier + intervention module decoding method

Metrics:
1. Total Accuracy

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
sys.path.append('/h/382/momoka/HKU/honest_llama')
# sys.path.append('../')

# import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import eval_math_shepherd, gen_steps_ans, convert_time
import pyvene as pv

from probes import Classifier, InterventionModule
from interveners import ClassifyIntervene

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
    parser.add_argument('--dataset_name', type=str, default='math-shepherd')
    parser.add_argument('--n_shot', type=int, default=8, help='number of examples given in the Input. Represeantions are ignored during classifying and intervention')
    parser.add_argument('--split_num', type=int, default=1, help='the number of dataset splits used. a single split contains 1k samples of the original dataset')
    parser.add_argument('--layer', type=int, default=16, help='the layer of the model to access the stat vars') #llama3.1-8b-instruct has 32 transformer layers, where the middle layers are supposed to be related to reasoning
    parser.add_argument('--classifier_path', type=str, default='./trained_probes/layer3_lstm_llama3.1_8b_instruct_16_math_shepherd_2k_best84.json', help='state_dict (json) path to the classifier')
    parser.add_argument('--intervention_module_path', type=str, default='./trained_probes/interventor_lstm_10k_classify-True.pth', help='state_dict path to the intervention module')
    parser.add_argument('--compare_baseline', type=bool, default=True, help='whether compare with baseline. default to be True')
    parser.add_argument('--max_step_iters', type=int, default=30, help='the max number of reasoning steps per sample, default to be 15')
    parser.add_argument('--stat_path', type=str, default='./evaluate/classify_intervene_result.jsonl', help='the path to save the evaluation statistics')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    if args.dataset_name == 'math-shepherd':
        data_path = f'./features/MATH-Shepherd_part_{args.split_num}.csv'
        formatter = eval_math_shepherd
    else: 
        raise ValueError("Invalid dataset name")

    '''
    1. Load test set
    '''
    start_t = time.time()
    print("Loading dataset...")
    #prefix_len是interpret representations时需要忽视的Input部分（sys_prmompt + examples + instruction) 的token长度
    eval_set, prefix = formatter(data_path, tokenizer, args.n_shot) # for math-shepherd, prompt = I+O (problem + steps so-far), already tokenized; label is binary
    prefix_len = tokenizer(prefix, return_tensors='pt').input_ids.size(-1)
    prefix_seq_len = len(prefix)
    
    print(f'Eval set size: {len(eval_set)}')
    # eval_loader = DataLoader(eval_set, batch_size=4, shuffle=False) #TODO: batchalize not implemented
    spent = convert_time(start_t)
    print(f'- Time spent on loading test set: {spent[0]}:{spent[1]}:{spent[2]}, or {(time.time() - start_t):.5f} secs')

    '''
    2. Load classifier and intervention module
    '''
    start_t = time.time()
    print('Loading classifier and intervention module...')
    #___hyper params___
    classifier_path = args.classifier_path
    interventor_path = args.intervention_module_path
    #__________________
    classifier = Classifier(
        'lstm',
        hidden_size=4096,
        output_size=1,
    )
    intervener = InterventionModule(
        'lstm',
        depth=1,
    )
    try:
        classifier.load_state_dict(classifier_path)
    except:
        raise ValueError(f"The specified classifier save_path: {classifier_path} does not exist!")
    try:
        state_dict = torch.load(interventor_path)
        intervener.load_state_dict(state_dict)
    except:
        raise ValueError(f"The specified classifier save_path: {classifier_path} does not exist!")
    spent = convert_time(start_t)
    print(f'- Time spent on loading probes: {spent[0]}:{spent[1]}:{spent[2]}, or {(time.time() - start_t):.5f} secs')

    # return ################

    '''
    3. Mount the classifier and the intervener to the base model
    '''
    start_t = time.time()
    print('Building pv_model...')
    print(f'The layer we look into: {args.layer}')
    pv_config = pv.IntervenableConfig(
        representations=[{
            "layer": args.layer,
            "component": f"model.layers[{args.layer}].post_attention_layernorm.output", #.mlp.output??
            "low_rank_dimension": 1,
            "intervention": ClassifyIntervene(classifier, intervener, prefix_len) # a wrapper containing the clasifier and the intervention module
        }]
    )
    pv_model = pv.IntervenableModel(pv_config, model)
    pv_model.set_device("cuda") 
    spent = convert_time(start_t)
    print(f'- Time spent on loading pv_model: {spent[0]}:{spent[1]}:{spent[2]}, or {(time.time() - start_t):.5f} secs')
    
    # return ##################
    
    '''
    4. Generate and evaluate
    '''
    print('Starting generation...')
    #___hyper params___
    max_step_num = args.max_step_iters
    compare_baseline = args.compare_baseline # compare the original model inference with step-by-step intervention inference
    stat_path = args.stat_path
    #__________________
    start_t = time.time()
    all_ans = []
    pv_acc_count = 0
    org_acc_count = 0
    invalid_count = 0
    with open(stat_path, 'w') as file:
        for i, sample  in enumerate(tqdm(eval_set)):
            question = prefix + sample['question'] #tokenized full_prompt (except the past steps)
            answer = sample['answer'] #int
            if not answer: #answer is null, invalid sample
                invalid_count += 1
                continue
            # 1. generate solutions before and after intervention
            pv_steps, pv_ans, org_steps, org_ans = gen_steps_ans(tokenizer, pv_model, model, question, max_step_num, compare_baseline, prefix_len=len(prefix))
            if i == 0:
                print(f'\n^^^^^^\n - pv_steps: \n{pv_steps}\n - pv_ans: [{pv_ans}]\n - org_steps: \n{org_steps}\n - org_ans: [{org_ans}]\n^^^^^^\n')
            pv_acc_count += (pv_ans == answer and answer)
            org_acc_count += (org_ans == answer and answer is None)
            dict = {
                'question': question[prefix_seq_len:],
                'answer': answer,
                'pv_solution': pv_steps,
                'pv_answer': pv_ans,
                'pv_correct': (pv_ans == answer),
                'org_solution': org_steps,
                'org_ans': org_ans,
                'org_correct': (org_ans == answer),
            }
            all_ans.append(answer)
            file.write(json.dumps(dict) + '\n')
            # break ##########################test
        # end of sample iter
        result= {
            'Acc': {'original': round(org_acc_count / (len(eval_set)-invalid_count), 5), 'intervened': round(pv_acc_count / (len(eval_set)-invalid_count), 5)}
        }
        print(f'\n\n___Evaluation Result___\noriginal Acc: [{org_acc_count}/{len(eval_set)}] ({result["Acc"]["original"]*100:.3f}%)\intervened Acc: [{pv_acc_count}/{len(eval_set)}] ({result["Acc"]["intervened"]*100:.3f}%)\n_______________________\n')
        file.write('\n' + json.dumps(result) + '\n')
    # end of write
    print(f'Statistics saved to {stat_path}')
    print(f'all_ans: {all_ans}')

    spent = convert_time(start_t)
    print(f'- Time spent on decoding: {spent[0]}:{spent[1]}:{spent[2]}, or {(time.time() - start_t):.5f} secs')


if __name__ == '__main__':
    main()
