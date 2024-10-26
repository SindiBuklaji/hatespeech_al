### TESTING PURPOSES

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
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

### Define the parameter search space for Logistic Regression
param_space_log_reg = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'solver': Categorical(['liblinear', 'lbfgs']),
    'max_iter': Integer(100, 1000)
}

# Perform Bayesian Search for Logistic Regression
bayes_search_log_reg = BayesSearchCV(LogisticRegression(), param_space_log_reg, n_iter=30, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
bayes_search_log_reg.fit(X_train, y_train)

print("Best parameters for Logistic Regression: ", bayes_search_log_reg.best_params_)
print("Best cross-validation accuracy for Logistic Regression: {:.2f}".format(bayes_search_log_reg.best_score_))

# Evaluate the best Logistic Regression model
best_log_reg_model = bayes_search_log_reg.best_estimator_
y_pred_log_reg = best_log_reg_model.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))
joblib.dump(best_log_reg_model, 'bayesian_logistic_regression_model_w2v.pkl')

### Define the parameter search space for SVC
param_space_svc = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf']),
    'max_iter': Integer(1000, 10000)
}

# Perform Bayesian Search for SVC
bayes_search_svc = BayesSearchCV(SVC(probability=True), param_space_svc, n_iter=30, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
bayes_search_svc.fit(X_train, y_train)

print("Best parameters for SVC: ", bayes_search_svc.best_params_)
print("Best cross-validation accuracy for SVC: {:.2f}".format(bayes_search_svc.best_score_))

# Evaluate the best SVC model
best_svc_model = bayes_search_svc.best_estimator_
y_pred_svc = best_svc_model.predict(X_test)
print("Support Vector Classifier (SVC)")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
joblib.dump(best_svc_model, 'bayesian_svc_model_w2v.pkl')

### Define the parameter search space for Random Forest
param_space_rf = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 8),
    'bootstrap': Categorical([True, False])
}

# Perform Bayesian Search for Random Forest
bayes_search_rf = BayesSearchCV(RandomForestClassifier(), param_space_rf, n_iter=30, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
bayes_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest: ", bayes_search_rf.best_params_)
print("Best cross-validation accuracy for Random Forest: {:.2f}".format(bayes_search_rf.best_score_))

# Evaluate the best Random Forest model
best_rf_model = bayes_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
joblib.dump(best_rf_model, 'bayesian_random_forest_model_w2v.pkl')

# Collect results
results = []

# Logistic Regression results
log_reg_results = {
    "Model": "Logistic Regression",
    "Best Parameters": bayes_search_log_reg.best_params_,
    "Best Cross-Validation Accuracy": bayes_search_log_reg.best_score_,
    "Test Accuracy": accuracy_score(y_test, y_pred_log_reg),
    "Precision": precision_recall_fscore_support(y_test, y_pred_log_reg, average='weighted')[0],
    "Recall": precision_recall_fscore_support(y_test, y_pred_log_reg, average='weighted')[1],
    "F1-Score": precision_recall_fscore_support(y_test, y_pred_log_reg, average='weighted')[2]
}
results.append(log_reg_results)

# SVC results
svc_results = {
    "Model": "Support Vector Classifier (SVC)",
    "Best Parameters": bayes_search_svc.best_params_,
    "Best Cross-Validation Accuracy": bayes_search_svc.best_score_,
    "Test Accuracy": accuracy_score(y_test, y_pred_svc),
    "Precision": precision_recall_fscore_support(y_test, y_pred_svc, average='weighted')[0],
    "Recall": precision_recall_fscore_support(y_test, y_pred_svc, average='weighted')[1],
    "F1-Score": precision_recall_fscore_support(y_test, y_pred_svc, average='weighted')[2]
}
results.append(svc_results)

# Random Forest results
rf_results = {
    "Model": "Random Forest Classifier",
    "Best Parameters": bayes_search_rf.best_params_,
    "Best Cross-Validation Accuracy": bayes_search_rf.best_score_,
    "Test Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')[0],
    "Recall": precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')[1],
    "F1-Score": precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')[2]
}
results.append(rf_results)

# Create DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file (optional)
results_df.to_csv('bayesian_model_comparison_results_w2v.csv', index=False)

# Display the DataFrame
results_df
