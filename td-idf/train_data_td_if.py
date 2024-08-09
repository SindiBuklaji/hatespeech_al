import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load the labeled data
labeled_data = pd.read_csv("data/merged_labeled_dataset.csv", encoding='utf-8')

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 4))
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
print(classification_report(y_test, y_pred_svc))



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

# Load the unlabeled data
unlabeled_data = pd.read_csv("youtube_comments.csv", encoding='utf-8')

# Feature extraction using the previously fitted TF-IDF vectorizer
X_unlabeled = tfidf_vectorizer.transform(unlabeled_data['Comment']).toarray()

# Scale the features using the previously fitted scaler
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Load the best model (choose the one with the highest accuracy)
best_model = joblib.load('logistic_regression_model_scaled.pkl')  # Replace with the best model you have found

# Predict labels for the unlabeled data
predicted_labels = best_model.predict(X_unlabeled_scaled)

# convert to integers
predicted_labels = predicted_labels.astype(int)

# Add predicted labels to the unlabeled data
unlabeled_data['predicted_label'] = predicted_labels

# Save the data with predicted labels
unlabeled_data.to_csv("youtube_comments_with_predictions_logistic_reg.csv", index=False, encoding='utf-8')

# Display the first few rows to verify
print(unlabeled_data.head())
