# Introspective-Reasoning
   - Use classifier + intervention module + etc. to analyze and improve model reasoning...

### 1. Set up the ITI environment
  - Follow the installation guide in https://github.com/likenneth/honest_llama
### 2. Build classifiers
  - Sample data splits from MATH-Shepherd: `get_activations/prepare_math_shepherd.py`
  - Build hidden_states-label pairs for each data split: `on-the-fly_probe_training.sh`
  - Train and evaluate classifiers: `probes_comparable_taining.sh`
### 4. Build intervention modules
  - :wrench:
### 5. Analysis
  - :wrench:
