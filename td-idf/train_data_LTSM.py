from sre_parse import Tokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import joblib
from collections import Counter

# Load the labeled data
labeled_data = pd.read_csv("data/sample_for_labeling.csv", encoding='utf-8')

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(labeled_data['Comment'])
X = tokenizer.texts_to_sequences(labeled_data['Comment'])
X = pad_sequences(X, maxlen=500)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labeled_data['Label'])

# Check class distribution
counter = Counter(y)
print(counter)

# Find the smallest class size
min_class_size = min(counter.values())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42, k_neighbors=min(min_class_size - 1, 5))
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("LSTM Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and tokenizer
model.save('lstm_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
