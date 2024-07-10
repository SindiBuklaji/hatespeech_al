import pandas as pd

# Load the full dataset
data = pd.read_csv("youtube_comments.csv", encoding='utf-8')

# Load the existing sample
existing_sample = pd.read_csv("sample_for_labeling.csv", encoding='utf-8')

# Exclude the existing sample from the full dataset
remaining_data = data[~data['Comment'].isin(existing_sample['Comment'])]

# Randomly sample 500 more comments from the remaining dataset
new_sample = remaining_data.sample(n=500, random_state=42)

# Combine the existing sample with the new sample
updated_sample = pd.concat([existing_sample, new_sample])

# Save the updated sample to a CSV file for manual labeling
updated_sample.to_csv("sample_for_labeling.csv", index=False, encoding='utf-8')
