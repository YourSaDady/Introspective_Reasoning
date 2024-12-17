#!/bin/bash

:'
Prerequisite: Already built the ITI environement.

Set up the new files and folders on the ITI code base.

'
cd honest_llama
mkdir features
mkdir trained_probes
cp "Preeptive-Reasoning/get_activations/get_activations.py" "honest_llama/get_activations/get_activations.py"
mv "Preeptive-Reasoning/get_activations/prepare_math_shepherd.py" "honest_llama/get_activations/"
mkdir probe_trainng
mv "Preeptive-Reasoning/probe_trainng/probe_training.py" "honest_llama/probe_training/"
cp "Preeptive-Reasoning/utils.py" "honest_llama/utils.py"
mv "Preeptive-Reasoning/on-the-fly_probe_training.sh" "honest_llama/"
mv "Preeptive-Reasoning/probes_comparable_training.sh" "honest_llama/"
mkdir trained_probes
mv "Preemptive-Reasoning/trained_probes/layer3_lstm_llama3.1_8b_instruct_16_math_shepherd_2k_best84.json" "honest_llama/trained_probes/"