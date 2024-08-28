import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

data = pd.read_csv('merged_labeled_dataset.csv')
device  = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

def tokenize_texts_in_batches(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokenized_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt', 
            max_length=128  
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized_batch)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.concatenate(embeddings, axis=0)
    
embeddings = tokenize_texts_in_batches(data['Comment'].tolist(), batch_size=16)
print(f'Embeddings shape: {embeddings.shape}')

pca = PCA(n_components=50)  
pca_embeddings = pca.fit_transform(embeddings)

print(f'PCA-reduced embeddings shape: {pca_embeddings.shape}')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    pca_embeddings, data['Label'], test_size=0.2, random_state=42, stratify=data['Label']
)

print("Class distribution before balancing:", y_train.value_counts())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after balancing:", y_train_resampled.value_counts())

# Scale the features
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Function to perform GridSearchCV and print results
def perform_grid_search(estimator, param_grid, X_train, y_train, X_test, y_test, model_name):
    grid_search = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: ", grid_search.best_params_)
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_:.2f}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\n{model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    

# Define parameter grid for Logistic Regression
param_grid_log_reg = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear'],
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced']  # Add class weights to the grid search
}

# Perform Grid Search for Logistic Regression
perform_grid_search(LogisticRegression(), param_grid_log_reg, X_train_resampled_scaled, y_train_resampled, X_test_scaled, y_test, "Logistic Regression")


# Define parameter grid for SVC
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear'],
    'max_iter': [2500, 3000]
}

# Perform Grid Search for SVC
perform_grid_search(SVC(probability=True), param_grid_svc, X_train_resampled_scaled, y_train_resampled, X_test_scaled, y_test, "Support Vector Classifier")

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True],
    'class_weight': ['balanced']  # Helps handle imbalance
}

# Perform Grid Search for Random Forest
perform_grid_search(RandomForestClassifier(), param_grid_rf, X_train_resampled_scaled, y_train_resampled, X_test_scaled, y_test, "Random Forest Classifier")

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * ((input_dim - 2) // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_dim = X_train_resampled.shape[1]
hidden_dim = 64  #adjust this
output_dim = len(np.unique(y_train_resampled))

#initialize the model
cnn_model = SimpleCNN(input_dim, hidden_dim, output_dim).to(device)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = cnn_model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
    cnn_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = cnn_model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve: Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Learning Curve: Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#evaluate the model
cnn_model.eval()
y_pred  = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = cnn_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_test, y_pred)
print(f'CNN Accuracy: {accuracy}')
print(classification_report(y_test, y_pred, zero_division=0))
