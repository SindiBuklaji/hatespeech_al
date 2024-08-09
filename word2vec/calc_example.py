from gensim.models import Word2Vec
import pandas as pd

# Define the comments
comments = [
    "avokati ditka gjitha çfar ndodhur pseee kaq von oooooavokat",
    "bravo artan futia gote kokes",
    "mire ishalla aqw",
    "zotria për qind flet drejt",
    "bravo zoti beqirijan gjitha verteta ato thuameta eshte super hajdut"
]

# Tokenize the comments into sentences of words
tokenized_comments = [comment.lower().split() for comment in comments]

# Initialize and train the Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_comments, vector_size=50, window=5, min_count=1, sg=0, workers=4)

# Example: Get the word vector for "bravo"
word_vector_bravo = word2vec_model.wv['bravo']
print(f"Word Vector for 'bravo':\n{word_vector_bravo}")

# Example: Find the most similar words to "bravo"
similar_words = word2vec_model.wv.most_similar('bravo')
print(f"Most similar words to 'bravo':\n{similar_words}")
