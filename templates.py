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

def apply_template(dataset_name, msg_dict):
    sys_msg = msg_dict['sys']
    user_msg = msg_dict['user']
    ass_msg = msg_dict['assistant']
    if dataset_name == 'math_shepherd':  #no system, only user and assistant
        full_prompt = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_msg}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{ass_msg}\n<|eot_id|>'
    else:
        raise NotImplementedError
    if msg_dict['remove_last_eot']:
        full_prompt = full_prompt.rstrip('<|eot_id|>')

    return full_prompt
