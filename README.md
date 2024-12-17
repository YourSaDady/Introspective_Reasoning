# Introspective-Reasoning
   - Use classifier + intervention module + etc. to analyze and improve model reasoning...
### 0. Set up honest_llama and introspective-reasoning
   - `git clone git@github.com:likenneth/honest_llama.git`
   - `git clone git@github.com:YourSaDady/Introspective_Reasoning.git`
   - `chmod +x ./Introspective_Reasoning/setup.sh`
   - `./Introspective_Reasoning/setup.sh`
   - `cd honest_llama`
### 1. Set up the ITI environment
  - Follow the installation guide in https://github.com/likenneth/honest_llama
### 2. Build classifiers
  - Sample data splits from MATH-Shepherd: `python ./get_activations/prepare_math_shepherd.py`
  - Build hidden_states-label pairs for each data split: `./on-the-fly_probe_training.sh`
  - Train and evaluate classifiers: `./probes_comparable_taining.sh`
### 4. Build intervention modules
  - :wrench:
### 5. Analysis
  - :wrench:
