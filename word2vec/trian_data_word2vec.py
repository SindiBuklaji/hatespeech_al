import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

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
y = labeled_data['Label']

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

### Define the parameter grid for Logistic Regression
param_grid_log_reg = {
    'C': [0.5, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [600, 700, 800]
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

### Define the paramter grid for SVC
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'max_iter': [3000, 4000]
}

# Perform Grid Search for SVC
grid_search_svc = GridSearchCV(SVC(probability=True), param_grid_svc, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)

print("Best parameters for SVC: ", grid_search_svc.best_params_)
print("Best cross-validation accuracy for SVC: {:.2f}".format(grid_search_svc.best_score_))

# Evaluate the best SVC model
best_svc_model = grid_search_svc.best_estimator_
y_pred_svc = best_svc_model.predict(X_test)
print("Support Vector Classifier (SVC)")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

### Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10. 20],
    'min_samples_leaf': [1, 2, 4, 5],
    'bootstrap': [True, False]
}

# Perform Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=2, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
print("Best cross-validation accuracy for Random Forest: {:.2f}".format(grid_search_rf.best_score_))

# Evaluate the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
