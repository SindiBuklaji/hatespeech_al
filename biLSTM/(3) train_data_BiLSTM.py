import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the labeled data
labeled_data = pd.read_csv("sample_for_labeling.csv", encoding='utf-8')

# Prepare the tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(labeled_data['Comment'])
X = tokenizer.texts_to_sequences(labeled_data['Comment'])

# Pad sequences to ensure uniform input size
X = pad_sequences(X, maxlen=300)  # Adjust maxlen as needed

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labeled_data['Label'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=300))  # Adjust input_dim and output_dim as needed
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Save the model
model.save('bilstm_model.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
'''

# Load the unlabeled data
unlabeled_data = pd.read_csv("youtube_comments.csv", encoding='utf-8')

# Preprocess the unlabeled data
X_unlabeled = tokenizer.texts_to_sequences(unlabeled_data['Comment'])
X_unlabeled = pad_sequences(X_unlabeled, maxlen=300)  # Ensure the same maxlen as used for training

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('bilstm_model.h5')

# Predict labels for the unlabeled data
predicted_labels = (model.predict(X_unlabeled) > 0.5).astype("int32")

# Add predicted labels to the unlabeled data
unlabeled_data['predicted_label'] = predicted_labels

# Save the data with predicted labels
unlabeled_data.to_csv("youtube_comments_with_predictions.csv", index=False, encoding='utf-8')

# Display the first few rows to verify
print(unlabeled_data.head())

'''
