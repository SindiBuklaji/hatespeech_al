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

X_train, X_test, y_train, y_test = train_test_split(
    pca_embeddings, data['Label'], test_size=0.2, random_state=42, stratify=data['Label']
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


  # Define a function to perform GridSearchCV and print results
def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from GridSearch
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred, zero_division=0))
    

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_resampled, y_train_resampled)


y_pred = logistic_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# Logistic Regression with GridSearchCV
logistic_param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Type of regularization
    'solver': ['liblinear', 'saga']  # Solver options
}

perform_grid_search(LogisticRegression(max_iter=1000), logistic_param_grid, 
                    X_train_resampled, y_train_resampled, X_test, y_test, "Logistic Regression")


svc_model = SVC(kernel='linear',random_state=42)
svc_model.fit(X_train_resampled, y_train_resampled)

y_pred=svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVC Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# SVC with GridSearchCV
svc_param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'kernel': ['linear', 'rbf'],  # Kernel types
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

perform_grid_search(SVC(random_state=42), svc_param_grid, 
                    X_train_resampled, y_train_resampled, X_test, y_test, "SVC")

random_forest=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
random_forest.fit(X_train_resampled, y_train_resampled)

y_pred=random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

rf_param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'class_weight': ['balanced', 'balanced_subsample']  # Handle class imbalance
}

perform_grid_search(RandomForestClassifier(random_state=42), rf_param_grid, 
                    X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest")


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
num_epochs = 30  #change the number of epochs
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
