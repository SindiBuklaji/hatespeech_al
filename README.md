# Online Hate Speech and Misogyny Detection in the Albanian Language

## Project Overview
This repository contains the code and data for the research project aimed at detecting hate speech and misogyny in the Albanian language. The project explores various text analysis algorithms, including traditional machine learning models and deep learning approaches, to identify and classify hate speech effectively. The study focuses on the application of different text vectorization techniques such as TF-IDF, Word2Vec, and BERT embeddings and evaluating their results respecively. 

## Table of Contents
1. [Data Description](#data-description)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Performance Metrics](#performance-metrics)
6. [How to Run](#how-to-run)
7. [Future Work](#future-work)
8. [Contributors](#contributors)

## Data Description
- **Dataset:** The dataset (`merged_labeled_dataset.csv`) contains user comments in the Albanian language, manually annotated with labels indicating whether the comment contains hate speech or not.
- **Classes:** The dataset includes two classes:
  - `0`: Non-hateful comments
  - `1`: Hateful comments, including misogyny

## Preprocessing
The preprocessing steps include:
1. Removal or stop-words, punctuation, links, mentions, etc. 
2. Tokenization using the different techniques (BERT Embeddings, TD-IDF, Word2Vec)
3. PCA is applied to reduce the dimensionality of the BERT embeddings.
4. SMOTE is sed to balance the dataset by oversampling the minority class.
5. Features are scaled using StandardScaler.

The general preprocessing code is available under the `data` directory, while the tokenization is done seperately and can be found specifically in the `bert`, `td-idf` and `word2vec` directories. 

## Models
The following models were trained and evaluated:
- **Traditional Machine Learning Models:**
  - Logistic Regression
  - Random Forest
  - Support Vector Classifier (SVC)
- **Deep Learning Models:**
  - Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)

These models were trained using the following vectorization techniques:
- **TF-IDF**
- **Word2Vec**
- **BERT Embeddings**

Model training scripts are located in the respective tokenization directories.

## Performance Metrics
The models were evaluated using various performance metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC Curve and AUC**

The best performing model was the Random Forest with Word2Vec embeddings, achieving an accuracy of 92.5% and an F1-score of 93%.

## How to Run
1. **Clone the repository:** 
