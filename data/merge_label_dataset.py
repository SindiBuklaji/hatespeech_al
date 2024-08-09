import pandas as pd

# Load the CSV with the specific format
file_path1 = 'data/sample_for_labeling.csv'  # Change this to your actual file path
df1 = pd.read_csv(file_path1)

# Load the new CSV file after replacing values
file_path2 = 'SHAJ/cleaned_full_albanian_dataset.csv'  # Change this to your actual file path
df2 = pd.read_csv(file_path2, header=None, delimiter=';')

# Keep only the relevant columns from df2 and add missing columns
df2 = df2.iloc[:, :2]
df2.columns = ['Comment', 'Label']
df2['Video ID'] = ''  # Add empty 'Video ID' column
df2['Author'] = ''    # Add empty 'Author' column

# Reorder columns to match df1
df2 = df2[['Video ID', 'Author', 'Comment', 'Label']]

# Concatenate the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged dataset
output_path = 'data/merged_labeled_dataset.csv'
merged_df.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
