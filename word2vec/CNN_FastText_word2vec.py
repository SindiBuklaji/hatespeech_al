### TESTING PURPOSES 

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

# Custom stdout reconfiguration for UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load pre-trained FastText embeddings for Albanian
fasttext_path = 'cc.sq.300.vec'
word2vec = KeyedVectors.load_word2vec_format(fasttext_path, binary=False)

# Define the CNN model with pre-trained embeddings
class CNNModel(torch.nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.conv1 = torch.nn.Conv1d(embedding_matrix.shape[1], 128, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(128 * (max_len // 2), num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# Function to load and prepare the dataset
def load_dataset(filename):
    data = pd.read_csv(filename, encoding='utf-8')
    texts = data['Comment'].values
    labels = data['Label'].astype(int).values
    return texts, labels

# Function to create the embedding matrix
def create_embedding_matrix(word_index, word2vec, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))  # Random initialization for unknown words
    return embedding_matrix

# Function to prepare data loaders
def prepare_dataloaders(texts, labels, batch_size, max_words, max_len, embedding_dim):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    word_index = tokenizer.word_index
    embedding_matrix = create_embedding_matrix(word_index, word2vec, embedding_dim)
    
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
    
    return train_loader, val_loader, test_loader, embedding_matrix

# Function to train the model
def train_model(model, train_loader, val_loader, epochs, learning_rate):
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = torch.nn.BCELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                loss = loss_fn(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions = model(x_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.squeeze().cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Parameters
    filename = "data/sample_for_labeling.csv"
    batch_size = 32
    max_words = 10000
    max_len = 100
    embedding_dim = 300  # I assume FastText embeddings of 300 dimensions
    num_classes = 1
    epochs = 10
    learning_rate = 0.001

    # Load and prepare data
    texts, labels = load_dataset(filename)
    train_loader, val_loader, test_loader, embedding_matrix = prepare_dataloaders(texts, labels, batch_size, max_words, max_len, embedding_dim)

    # Initialize the model
    model = CNNModel(embedding_matrix, num_classes)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, learning_rate)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    '''
        # Plotting the loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')
    plt.show()

    # Save the model state
    torch.save(model.state_dict(), 'cnn_model.pt')
    '''
