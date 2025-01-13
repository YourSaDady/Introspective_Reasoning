'''
Load and save the data split from HuggingFace to the local jsonl file.

1. download the data split from HF to the local ./features dir
2. run this file to convert data split into jsonl format

train size: 791
validate size: 201

'''


import os
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
import sys
import json
from tqdm import tqdm
sys.path.append('/h/382/momoka/HKU/honest_llama')
# Path to the 'train' directory

def convert2JSONserializable(data_dict):
    for key in data_dict:
        h_list = data_dict[key]
        data_dict[key] = str(h_list)

layer = 16
split_num = 9
divs = {'train': 791, 'validation': 201} #size

# for split_num in range(6, 10):
# def load_HF_data_slit()
for div in divs:
    if div == 'validation': ################
        continue ################
    print(f'Now on {div}...')
    src_dir = f'./features/intervene_{split_num}k/{div}'
    save_dir = f'./features/llama3.1-8b-instruct_{layer}_math-shepherd_{split_num}k_{div}_set.jsonl'
    # Initialize empty lists to store the data
    all_data = []

    # Iterate through each subdirectory
    print("Iterate through each subdirectory")
    with open(save_dir, 'w') as f:
        for i in tqdm(range(divs[div])):
            subdir = os.path.join(src_dir, f'sample{i}')
            parquet_file = os.path.join(subdir, f'{div}-00000-of-00001.parquet')

            if os.path.exists(parquet_file):
                # Read the Parquet file
                dataset = pq.read_table(parquet_file).to_pandas()

                # Convert to Hugging Face Dataset
                data = Dataset.from_pandas(dataset)
                # print(f"h_prior: {data['h_prior']}")
                # print(f'h_posterior: {data["h_posterior"]}')
                data_dict = data.to_dict()
                # convert2JSONserializable(data_dict)

                # # Append the dataset to the list
                # all_data.append(data_dict)
                f.write(json.dumps(data_dict) + '\n')
        # break

    # # Concatenate the datasets from all subdirectories
    # print("Converting dataset into a jsonl file...")

    # # Iterate through each dataset in all_data and write to the output JSONL file
    # with open(save_dir, 'w') as f:
    #     for data_dict in all_data:
    #         f.write(json.dumps(data_dict) + '\n')
    print(f'dataset saved to {save_dir}')