# Introspective-Reasoning
   - Use classifier + intervention module + etc. to analyze and improve model reasoning...
### 0. Set up honest_llama and introspective-reasoning
   - `git clone git@github.com:likenneth/honest_llama.git`
   - `git clone git@github.com:YourSaDady/Introspective_Reasoning.git`
   - `chmod +x ./Introspective_Reasoning/setup.sh`
   - `./Introspective_Reasoning/setup.sh`
   - `cd honest_llama`
   - `chmod +x ./on-the-fly_probe_training.sh`
   - `chmod +x ./probes_comparable_training.sh`
### 1. Set up the ITI environment
  - Follow the installation guide in https://github.com/likenneth/honest_llama
### 2. Build classifiers
  - Sample data splits from MATH-Shepherd: `python ./get_activations/prepare_math_shepherd.py`
  - Build hidden_states-label pairs for each data split: `./on-the-fly_probe_training.sh`
  - Train and evaluate classifiers: `./probes_comparable_taining.sh`
### 4. Build intervention modules
  - Build hs-hs'-label pairs (hidden states before and after intervention and the step label) for each data split: `python ./get_activations/build_intervention_datsets.py --split_num {n}`, with split index `n` ranging from 1 to 10
  - Datasets for training the intervention module / PEFT is on the HuggingFace: `Lo-Fi-gahara/intervene_{n}k`
  - Train the intervention module: `python ./probes_training/probe_training.py --split_num {n} --probe_type intervention_module`
### 5. Analysis
  - FactCheckMate-style analysis:
       - probes online training (and analysis): `python ./probing_survey/analysis.py --mode specified --online_training True`
       - analysis in freezed mode (use probe trained on a single type of hiddens, say I+O, to analyze different types of hiddens): `python ./probing_survey/analysis.py --mode freezed --probe_path {...}`
       - analysis in specified mode (use probes trained from different hiddens types to analyze the corresponding hiddens types): `python ./probing_survey/analysis.py --mode specified` (specify the probes' paths in the code)
