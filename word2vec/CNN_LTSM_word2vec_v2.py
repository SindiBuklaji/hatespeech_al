import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import io
import sys

# Load the labeled data
labeled_data = pd.read_csv("data/merged_labeled_dataset.csv", encoding='utf-8')

# Tokenize the comments
labeled_data['tokenized'] = labeled_data['Comment'].apply(lambda x: x.split())

# Train a Word2Vec model
w2v_model = Word2Vec(sentences=labeled_data['tokenized'], vector_size=100, window=5, min_count=1, workers=4)

# Save the Word2Vec model
w2v_model.save("word2vec_model.bin")

# Load the Word2Vec model
w2v_model = Word2Vec.load("word2vec_model.bin")

# Function to get the average word2vec vector for a comment
def get_avg_word2vec(comment, model, num_features):
    words = comment.split()
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    for word in words:
        if word in model.wv.key_to_index:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# Generate embeddings for all comments
X = np.array([get_avg_word2vec(comment, w2v_model, 100) for comment in labeled_data['Comment']])
y = labeled_data['Label'].values  # Convert to a NumPy array

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Ensure labels are long for classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # Ensure labels are long for classification

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple CNN model using PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * ((input_dim - 2) // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()

# Define an LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # Reshape for LSTM
        out = self.fc(self.relu(out[:, -1, :]))
        return self.sigmoid(out).squeeze()


# Define hyperparameters
input_dim = X_train.shape[1]  # Number of features (100 from Word2Vec)
hidden_dim = 64  # Number of hidden units
output_dim = 1  # Single output for binary classification

# Initialize the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = SimpleCNN(input_dim, hidden_dim, output_dim).to(device)
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, train_loader, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()  # Ensure labels are floats for BCELoss
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()  # Ensure labels are floats for BCELoss
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Train and evaluate CNN model
print("Training CNN Model...")
train_model(cnn_model, cnn_optimizer, train_loader)
print("Evaluating CNN Model...")
evaluate_model(cnn_model, test_loader)

# Train and evaluate LSTM model
print("\nTraining LSTM Model...")
train_model(lstm_model, lstm_optimizer, train_loader)
print("Evaluating LSTM Model...")
evaluate_model(lstm_model, test_loader)

