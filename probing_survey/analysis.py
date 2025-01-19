'''
Survey on the reasoning behaviors of the base model through probing the hidden states.

Specificially:
    - Suppose: 
        - I: input question
        - O: partial steps before n
        - F: partial steps after n
        - R: (correctness) of the result

    P1. Causal Intervention Analysis (R always on the left; training-free?):
        - The importance of each step to the final correctness: 
            - The importance of O: Corr(I+O, R) - Corr(I, R)
            - The importance of F: Corr(I+O+F, R) - Corr(I+O, R)
        - Whether there exists pre-caching(whether O contains information for preficiton):
            - Corr(I+O+F) - Corr(I+F)
    P2. Myopic Extent Analysis (in-process correctness on the left; training-free?)
        - If pre-caching exists (proved), is predicting distant future more difficult than predicting near future?

    P3. Layer Analysis:
        - where does the future planning / final answer happen / generated?

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
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import eval_gsm8k, eval_math_shepherd, gen_steps_ans, convert_time, run_test_gen, gen_gsm8k
import pyvene as pv

from probes import Classifier, InterventionModule
from interveners import ClassifyIntervene
print(f'The current working directory: {os.getcwd()}')

HF_NAMES = {
    'llama3.1_8b_instruct': "meta-llama/Llama-3.1-8B-Instruct",
}

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='llama3.1_8b_instruct')
    # parser.add_argument('--dataset_name', type=str, default='math_shepherd')
    parser.add_argument('--hiddens_path', type=str, help='the path to the hiddens datasets')
    parser.add_argument('--corr_base', type=str, default='hiddens', help='hiddens-based or score-based. The former is free of probes training and use the hiddens directly in calculating correlation. The latter is otherwise.')
    parser.add_argument('--analysis_type', type=str, default='causal_intervention', help='choose from {causal_intervention, myopic_extent, layerwise}. Default: causal_intervention')
    parser.add_argument('--layer', default=16, type=int)

    # load hiddens datasets


if __name__ == '__main__':
    main()