#!/bin/bash

# on-the-fly pipline:
#    1. Building training and validation datasets
#    2. Probe training & validation
#    3. Deleting datasets

# hyper params:
#    - layer
#    - sample_num
#    - probe_type


probe_types=("lstm" "linear_bow" "linear_mean" "nonlinear_bow" "nonlinear_mean")
# layers = (10 12 14 15 16 18 20)
# sample_nums = (5000 10000)

# Train all the probes in one run.     

for split_ in $(seq 3 4); do
    for layer_ in $(seq 16 20); do
        echo -e "\n---------------------\n1. Building datasets from data split ($split_) for layer ($layer_): \nNow running get_activations.py...\n---------------------\n"
        python ./get_activations/get_activations.py --model_name llama3.1_8b_instruct --layer $layer_ --dataset math_shepherd --split_num $split_

        # echo -e "\n---------------------\n2. Probe training & validation from data split ($split_) for layer ($layer_): \nNow running probe_training.py...\n---------------------\n"
        # python ./probe_training/probe_training.py --model_name llama3.1_8b_instruct --layer $layer_ --dataset math_shepherd --split_num $split_

        # echo "\n---------------------\n3. Probe training complete. Now deleting the datasets...\n---------------------\n"
        # rm -f ./features/llama3.1_8b_instruct_${layer_}_math_shepherd_${split_}k_train_set_0721.jsonl
        # rm -f ./features/llama3.1_8b_instruct_${layer_}_math_shepherd_${split_}k_validate_set_0721.jsonl

        break ########

    done
    break ##############
done
