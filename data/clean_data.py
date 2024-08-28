import pandas as pd
import re

# Read the CSV file with UTF-8 encoding
data = pd.read_csv("youtube_comments.csv", encoding='utf-8')

# Define a basic list of Albanian stopwords manually
albanian_stopwords = set([
    'a', 'as', 'bë', 'bën', 'bëri', 'bënim', 'bëni', 'bënë', 'cila', 'cilat', 'cili', 'cilit', 'cilët',
    'dhe', 'e', 'edhe', 'gjatë', 'i', 'ia', 'iha', 'iho', 'isha', 'ishe', 'ishin', 'ishit', 'jam', 'janë',
    'je', 'jemi', 'jeni', 'ka', 'kam', 'kemi', 'keni', 'kish', 'kisha', 'kishin', 'kishit', 'kishte',
    'kjo', 'kjo', 'këto', 'këtë', 'këtij', 'këto', 'me', 'me', 'më', 'mi', 'mos', 'nga', 'një', 'nuk',
    'pa', 'pse', 'që', 'si', 'ta', 'te', 'ti', 'tij', 'tillë', 'tilla', 'të', 'u', 'unë', 'veç', 'veçse',
    'vetëm', 'çfarë', 'çdo', 'çka'
])

# Define the preprocessing function to make text lowercase
def preprocess_text_albanian(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'@\w+|http\S+', '', text)
        text = re.sub(r'[^a-zA-ZçÇëË ]', '', text)
        text = ' '.join([word for word in text.split() if word not in albanian_stopwords])
        text = ' '.join(text.split())
    else:
        text = ""
    return text

# Apply preprocessing to comments
data['Comment'] = data['Comment'].apply(preprocess_text_albanian)

# Drop rows where the 'Comment' column is empty after cleaning
data = data[data['Comment'].str.strip() != ""]

data.to_csv("youtube_comments.csv", index=False, encoding='utf-8')

updated_data = pd.read_csv("youtube_comments.csv", encoding='utf-8')

# Display the first few rows to check the preprocessing
print(data[['Comment']].head().to_string().encode('ascii', 'ignore').decode('ascii'))
