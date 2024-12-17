from datasets import load_dataset
import sys

sys.path.append('../')

# Load the dataset
dataset = load_dataset('peiyi9979/Math-Shepherd')['train']

# Shuffle the dataset (you may specify a seed for reproducibility)
shuffled_dataset = dataset.shuffle(seed=42)

# Get the first 10,000 samples
sampled_dataset = shuffled_dataset.select(range(10000))

# Split the sampled dataset into 10 separate datasets
split_size = 1000  # 10,000 samples / 10 datasets
for i in range(10):
    # Get the slice for the current dataset
    split_data = sampled_dataset.select(range(i * split_size, (i + 1) * split_size))
    
    # Save the split dataset locally as a CSV file or any other format
    split_data.to_csv(f'../features/MATH-Shepherd_part_{i+1}.csv')