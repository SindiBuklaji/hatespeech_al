import pandas as pd

# Load the dataset
filename = "data/merged_labeled_dataset.csv"
data = pd.read_csv(filename, encoding='utf-8')

data.replace({'0.0': 0, '1.0': 1}, inplace=True)

# Identify rows with invalid labels (not 0 or 1)
invalid_rows = data[~data['Label'].isin([0, 1])]

# Print rows with invalid labels
if not invalid_rows.empty:
    print("Rows with invalid labels:")
    print(invalid_rows)
else:
    print("All labels are valid (0 or 1).")

# Save the cleaned dataset
cleaned_filename = "data/merged_labeled_dataset.csv"
data.to_csv(cleaned_filename, index=False)

print(f"Cleaned data saved to {cleaned_filename}")
