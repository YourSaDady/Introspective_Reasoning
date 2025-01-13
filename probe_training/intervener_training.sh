#!/bin/bash

# Use multiple data splits to train the intervention module

# splits=(6 7 8 9 10)
layer=16

# sample_nums = (5000 10000)

# Train all the probes in one run.     

for split in $(seq 9 10); do
        echo -e "\n---------------------\nIntervener training from data split ($split): \nNow running probe_training.py...\n---------------------\n"
        python ./probe_training/probe_training.py --probe_type intervention_module --split_num $split --test True

        # echo "\n---------------------\n3. Probe training complete. Now deleting the datasets...\n---------------------\n"
        # rm -f ./features/llama3.1_8b_instruct_${layer_}_math_shepherd_${split_}k_train_set_0721.jsonl
        # rm -f ./features/llama3.1_8b_instruct_${layer_}_math_shepherd_${split_}k_validate_set_0721.jsonl
    break ##############
done
