from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Sample comments
comments = [
    "avokati ditka gjitha çfar ndodhur pseee kaq von oooooavokat",
    "bravo artan futia gote kokes",
    "mire ishalla aqw",
    "zotria për qind flet drejt",
    "bravo zoti beqirijan gjitha verteta ato thuameta eshte super hajdut"
]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the comments to TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(comments)

# Convert the TF-IDF matrix to a DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Calculate IDF manually
idf_values = tfidf_vectorizer.idf_
idf_df = pd.DataFrame(idf_values, index=tfidf_vectorizer.get_feature_names_out(), columns=["IDF"])

# Select the words of interest
words_of_interest = ["bravo", "avokati", "gjitha", "drejt", "hajdut"]

# Extract TF values for the specific comments
tf_comment_2 = tfidf_df.loc[1, words_of_interest]
tf_comment_5 = tfidf_df.loc[4, words_of_interest]

# Compute TF (Term Frequency) values
tf_values_2 = tf_comment_2 / tf_comment_2.sum()
tf_values_5 = tf_comment_5 / tf_comment_5.sum()

# Create the result table
result_table = pd.DataFrame({
    "Word": words_of_interest,
    "TF (Comment 2)": tf_values_2,
    "TF (Comment 5)": tf_values_5,
    "IDF": idf_df.loc[words_of_interest]["IDF"],
    "TF-IDF (Comment 2)": tf_comment_2,
    "TF-IDF (Comment 5)": tf_comment_5
})

# Reset the index to add the words as a column
result_table.reset_index(drop=True, inplace=True)

# Fill NaN values with zeros (for words not present in a comment)
result_table = result_table.fillna(0)

# Display the result table
print(result_table)
