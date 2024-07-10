import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 4))
X = tfidf_vectorizer.fit_transform(labeled_data['Comment']).toarray()
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
joblib.dump(best_log_reg_model, 'bayesian_logistic_regression_model_tfidf.pkl')

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
joblib.dump(best_svc_model, 'bayesian_svc_model_tfidf.pkl')

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
joblib.dump(best_rf_model, 'bayesian_random_forest_model_tfidf.pkl')