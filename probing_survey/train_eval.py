'''
on-the-fly probe training and evaluate: probe只接受一种类型的hiddens。在遍历samples时online进行train+evaluate.
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
from utils import load_math_shepherd, build_nshot_examples, probing_analysis, convert_time
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
    parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--eval_split', type=int, default=1) #default to use split 1 and 2 and 5 epochs for training 
    parser.add_argument('--n_shot', type=int, default=8)
    parser.add_argument('--build_from_scratch', type=bool, default=False)
    args = parser.parse_args()

    #0. prepare nshot
    print('\nbuilding n-shot examples from split1...\n')
    result, others = build_nshot_examples(args.dataset_name, 1, args.n_shot)  
    nshot = others['nshot']

    #1. load modela and initialize probe