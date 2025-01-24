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
sys.path.append('/home/yichuan/HKU/honest_llama')
os.chdir('/home/yichuan/HKU/honest_llama')
print(f'The current working directory: {os.getcwd()}')
# Path to the 'train' directory

'''
convert a data batch into a list of samples
'''
def debatch(dataset):
    sample_list = []
    for sample_id in range(len(dataset)):
        sample = {key: dataset[key][sample_id] for key in dataset}
        sample_list.append(sample)
    return sample_list

def convert2JSONserializable(data_dict):
    for key in data_dict:
        h_list = data_dict[key]
        data_dict[key] = str(h_list)
#_______hyper params________
layer = 16
split_num = 1
divs = {'train': 791, 'validation': 201} #size
#___________________________


# for split_num in range(6, 10):
# def load_HF_data_slit()
for div in divs:
    # if div == 'validation': ################
    #     continue ################
    print(f'Now on {div}...')
    src_dir = f'./features/classify_{split_num}k/'
    save_dir = f'./features/llama3.1-8b-instruct_{layer}_math-shepherd_{split_num}k_{div}_set.jsonl'
    # Initialize empty lists to store the data
    all_data = []
    count = 0
    # Iterate through each subdirectory
    print(f"Now converting {div}...")
    with open(save_dir, 'w') as f:
        for i in tqdm(range(0, 5400, 100)):
            subdir = os.path.join(src_dir, f'sample{i}-{i+99}')
            parquet_file_names = [f'{div}-00000-of-00001.parquet', f'{div}-00000-of-00002.parquet', f'{div}-00001-of-00002.parquet']
            for p_name in parquet_file_names:
                parquet_file = os.path.join(subdir, p_name)
                if os.path.exists(parquet_file):
                    print(f'\n file "{parquet_file}" exists!\n')
                    # Read the Parquet file
                    dataset = pq.read_table(parquet_file).to_pandas()

                    # Convert to Hugging Face Dataset
                    data = Dataset.from_pandas(dataset)
                    # print(f"h_prior: {data['h_prior']}")
                    # print(f'h_posterior: {data["h_posterior"]}')
                    data_dict = data.to_dict()
                    # convert2JSONserializable(data_dict)
                    data_list = debatch(data_dict)
                    # # Append the dataset to the list
                    # all_data.append(data_dict)
                    count += len(data_list)
                    for data in data_list:
                        f.write(json.dumps(data) + '\n')
            #end of parquet files
            break ##############test
        #end of dirs
    print(f'\nCompleted converting {count} {div} samples. Saved to {save_dir}')
        # break

    # # Concatenate the datasets from all subdirectories
    # print("Converting dataset into a jsonl file...")

    # # Iterate through each dataset in all_data and write to the output JSONL file
    # with open(save_dir, 'w') as f:
    #     for data_dict in all_data:
    #         f.write(json.dumps(data_dict) + '\n')
    # print(f'dataset saved to {save_dir}')