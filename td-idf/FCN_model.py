import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import sys

'''# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''

# Custom stdout reconfiguration for UTF-8 encoding support
sys.stdout.reconfigure(encoding='utf-8')

# Define the Fully Connected Network (FCN) model for TF-IDF features
class FCNModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Function to load and prepare the dataset
def load_dataset(filename):
    labeled_data = pd.read_csv(filename, encoding='utf-8')
    texts = labeled_data['Comment'].values
    labels = labeled_data['Label'].astype(int).values

    return texts, labels

# Function to prepare data loaders using TF-IDF vectorization
def prepare_dataloaders(texts, labels, batch_size):
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_features = vectorizer.fit_transform(texts).toarray()

    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: np.random.seed(42))
    
    return train_loader, val_loader, test_loader, vectorizer

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
    num_classes = 1
    epochs = 10
    learning_rate = 0.001

    # Load and prepare data
    texts, labels = load_dataset(filename)
    train_loader, val_loader, test_loader, vectorizer = prepare_dataloaders(texts, labels, batch_size)

    # Initialize the model
    input_dim = len(vectorizer.get_feature_names_out())
    model = FCNModel(input_dim, num_classes)

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
    torch.save(model.state_dict(), 'fcn_model.pt')
    '''
