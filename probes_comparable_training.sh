#!/bin/bash

# on-the-fly pipline:
#    1. Building training and validation datasets
#    2. Probe training & validation
#    3. Deleting datasets

# hyper params:
#    - layer
#    - sample_num
#    - probe_type

# layers = (10 12 14 15 16 18 20)
# sample_nums = (5000 10000)

: '
incremental on-the-fly training: 
    Each epoch train with 1k samples, the parameters are from the previous epoch
'

layer=16
split=3

# for split_ in $(seq 1 2); do

# echo -e "\n---------------------\nBuilding datasets from data split (2) for layer ($layer): \nNow running get_activations.py...\n---------------------\n"
# python ./get_activations/get_activations.py --model_name llama3.1_8b_instruct --layer $layer --dataset math_shepherd --split_num 3

echo -e "\n---------------------\nAll kinds of probes training & validation, on split ($split) for layer ($layer) : \nNow running probe_training.py...\n---------------------\n"
python ./probe_training/probe_training.py --model_name llama3.1_8b_instruct --layer $layer --dataset math_shepherd --split_num $split

# echo -e "\n---------------------\nProbe training complete. Now deleting the datasets...\n---------------------\n"
# rm -f ./features/llama3.1_8b_instruct_${layer}_math_shepherd_1k_train_set.jsonl
# rm -f ./features/llama3.1_8b_instruct_${layer}_math_shepherd_1k_validate_set.jsonl

# done