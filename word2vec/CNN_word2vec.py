import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import classification_report

# Load the labeled data
labeled_data = pd.read_csv("data/sample_for_labeling.csv", encoding='utf-8')

# Tokenize the comments
labeled_data['tokenized'] = labeled_data['Comment'].apply(lambda x: x.split())

# Train a Word2Vec model
w2v_model = Word2Vec(sentences=labeled_data['tokenized'], vector_size=100, window=5, min_count=1, workers=4)

# Save the Word2Vec model
w2v_model.save("word2vec_model.bin")

# Load the Word2Vec model
w2v_model = Word2Vec.load("word2vec_model.bin")

# Tokenize the comments using Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(labeled_data['Comment'])
sequences = tokenizer.texts_to_sequences(labeled_data['Comment'])

# Get the word index from the tokenizer
word_index = tokenizer.word_index

# Create an embedding matrix
embedding_dim = 100  # Same as your Word2Vec vector size
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

for word, i in word_index.items():
    if word in w2v_model.wv.key_to_index:
        embedding_matrix[i] = w2v_model.wv[word]

# Pad sequences to ensure uniform input length
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)

# Convert labels to numerical format
y = labeled_data['Label'].values

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert labels to categorical after the split
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_test_classes, y_pred_classes))
