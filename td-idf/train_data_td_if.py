import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Load the labeled data
labeled_data = pd.read_csv("data/merged_labeled_dataset.csv", encoding='utf-8')

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(labeled_data['Comment']).toarray()
y = labeled_data['Label']

# Check the distribution of classes
print("Class distribution before SMOTE:", y.value_counts())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the distribution of classes after SMOTE
print("Class distribution after SMOTE:", pd.Series(y_resampled).value_counts())

# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)


# Define the parameter grid for Logistic Regression
param_grid_log_reg = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear'],
    'max_iter': [100, 200, 300]
}

# Perform Grid Search for Logistic Regression
grid_search_log_reg = GridSearchCV(LogisticRegression(), param_grid_log_reg, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_log_reg.fit(X_train, y_train)

print("Best parameters for Logistic Regression: ", grid_search_log_reg.best_params_)
print("Best cross-validation accuracy for Logistic Regression: {:.2f}".format(grid_search_log_reg.best_score_))

# Evaluate the best Logistic Regression model
best_log_reg_model = grid_search_log_reg.best_estimator_
y_pred_log_reg = best_log_reg_model.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Define the parameter grid for SVC
param_grid_svc = {
    'svc__C': [0.1, 1, 10],  # Regularization parameter
    'svc__kernel': ['linear'],  # Kernel type
    'svc__max_iter': [2500, 5000]  # Maximum iterations
}

# Create a pipeline with StandardScaler and SVC
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('svc', SVC(probability=True))
])

# Perform Grid Search for SVC
grid_search_svc = GridSearchCV(pipeline, param_grid_svc, cv=5, n_jobs=-1, scoring='accuracy', refit=True)
grid_search_svc.fit(X_train, y_train)

print("Best parameters for SVC: ", grid_search_svc.best_params_)
print("Best cross-validation accuracy for SVC: {:.2f}".format(grid_search_svc.best_score_))

# Evaluate the best SVC model
best_svc_model = grid_search_svc.best_estimator_
y_pred_svc = best_svc_model.predict(X_test)
print("Support Vector Classifier (SVC)")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))

### Define the parameter grid for Random Forest
param_grid_rf = {
    'randomforestclassifier__n_estimators': [50, 100, 200],  # Limit the number of trees
    'randomforestclassifier__max_depth': [None, 10, 20, 30],  # Smaller set of depths
    'randomforestclassifier__min_samples_split': [2, 5, 10],  # Reduce options for splits
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],  # Limit leaf nodes
    'randomforestclassifier__bootstrap': [True]  # Use bootstrap sampling
}

# Create a pipeline with StandardScaler and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('randomforestclassifier', RandomForestClassifier(random_state=42))
])

# Perform Grid Search for Random Forest
grid_search_rf = GridSearchCV(pipeline, param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy', refit=True)
grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
print("Best cross-validation accuracy for Random Forest: {:.2f}".format(grid_search_rf.best_score_))

# Evaluate the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#try bayesian optimization

bayesian_opt = BayesSearchCV(
    RandomForestClassifier(),
    param_grid_rf,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)
bayesian_opt.fit(X_train, y_train)

print("Best parameters for Random Forest: ", bayesian_opt.best_params_)
print("Best cross-validation accuracy for Random Forest: {:.2f}".format(bayesian_opt.best_score_))

# Predict probabilities
y_pred_prob = grid_search_rf.predict_proba(X_test)

# Compute ROC curve and ROC area for class 0 and class 1
fpr_0, tpr_0, _ = roc_curve(y_test, y_pred_prob[:, 0], pos_label=0)
fpr_1, tpr_1, _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
roc_auc_0 = auc(fpr_0, tpr_0)
roc_auc_1 = auc(fpr_1, tpr_1)

# Plot ROC curves
plt.figure()
plt.plot(fpr_0, tpr_0, color='blue', lw=2, label='ROC curve for class 0 (area = %0.2f)' % roc_auc_0)
plt.plot(fpr_1, tpr_1, color='red', lw=2, label='ROC curve for class 1 (area = %0.2f)' % roc_auc_1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', format = 'png')
plt.show()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a simple CNN model using PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * ((input_dim - 2) // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define hyperparameters
input_dim = X_train.shape[1]  # Number of features (2000 from TF-IDF)
hidden_dim = 64  # Number of hidden units
output_dim = len(np.unique(y_train))  # Number of classes (binary classification)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = SimpleCNN(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training loop
num_epochs = 30  # Number of epochs
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

# Final evaluation
cnn_model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = cnn_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())

print("CNN Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define an LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # Reshape for LSTM
        out = self.fc(self.relu(out[:, -1, :]))
        return out

# Define hyperparameters
input_dim = X_train.shape[1]  # Number of features (2000 from TF-IDF)
hidden_dim = 64  # Number of hidden units
output_dim = len(np.unique(y_train))  # Number of classes (binary classification)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop
num_epochs = 30  # Number of epochs
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    lstm_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = lstm_model(X_batch)
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
    lstm_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = lstm_model(X_batch)
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

# Final evaluation
lstm_model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = lstm_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())

print("LSTM Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


