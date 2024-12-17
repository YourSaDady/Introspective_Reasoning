#!/bin/bash


# Prerequisite: Already built the ITI environement.

# Set up the new files and folders on the ITI code base.


mkdir ./honest_llama/features
mkdir ./honest_llama/trained_probes
cp "./Introspective_Reasoning/get_activations/get_activations.py" "./honest_llama/get_activations/get_activations.py"
mv "./Introspective_Reasoning/get_activations/prepare_math_shepherd.py" "./honest_llama/get_activations/"
mkdir ./honest_llama/probe_training
mv "./Introspective_Reasoning/probe_training/probe_training.py" "./honest_llama/probe_training/"
cp "./Introspective_Reasoning/utils.py" "./honest_llama/utils.py"
mv "./Introspective_Reasoning/on-the-fly_probe_training.sh" "./honest_llama/"
mv "./Introspective_Reasoning/probes_comparable_training.sh" "./honest_llama/"
mv "./Introspective_Reasoning/trained_probes/layer3_lstm_llama3.1_8b_instruct_16_math_shepherd_2k_best84.json" "./honest_llama/trained_probes/"
