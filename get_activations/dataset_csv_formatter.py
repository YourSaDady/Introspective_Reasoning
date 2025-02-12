'''
Prepare all kinds of datasets into local csv data splits.
Preliminary for all kinds of training and evaluation.
'''
import sys
import os
import sys
import csv
import argparse
from datasets import load_dataset

HF_NAMES = {
    'math_shepherd': {'hf_path': 'peiyi9979/Math-Shepherd', 'splits': ['train']},
    'math': {'hf_path': 'openai/gsm8k', 'splits': ['train', 'test']},
    'gsm8k': {'hf_path': 'openai/gsm8k', 'splits': ['train', 'test']},
    'prm800k': {'hf_path': 'tasksource/PRM800K', 'splits': ['train', 'test']},
    'processbench': {'hf_path': 'Qwen/ProcessBench', 'splits': ['gsm8k', 'math', 'olympiadbench', 'omnimath']},
}

# sys.path.append('../') 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='choose from {math_shepherd, gsm8k, math, prm800k, processbench}')
    parser.add_argument('--split', type=str, default='train', help='choose from {test, train}. Depend on the dataset')
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(HF_NAMES[args.dataset_name]['hf_path'])[args.split]

    # Shuffle the dataset (you may specify a seed for reproducibility)
    shuffled_dataset = dataset.shuffle(seed=42)

    if args.dataset_name == 'math_shepherd':
        # Get the first 10,000 samples
        sampled_dataset = shuffled_dataset.select(range(10000))

        # Split the sampled dataset into 10 separate datasets
        split_size = 1000  # 10,000 samples / 10 datasets
        for i in range(10):
            # Get the slice for the current dataset
            split_data = sampled_dataset.select(range(i * split_size, (i + 1) * split_size))
            
            # Save the split dataset locally as a CSV file or any other format
            split_data.to_csv(f'./features/MATH-Shepherd_part_{i+1}.csv') #pwd is honest_llama
            print(f'Data split created to: "./features/MATH-Shepherd_part_{i+1}.csv"')
    elif args.dataset_name == 'gsm8k':
        return
    elif args.dataset_name == 'math':
        return
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()