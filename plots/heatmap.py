import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Models and Embeddings
models = ['Logistic Regression', 'SVC', 'Random Forest', 'CNN', 'LSTM']
embeddings = ['TF-IDF', 'Word2Vec', 'BERT']

# Accuracy data for each model and embedding type
accuracy_data = np.array([
    [0.79, 0.64, 0.45],  # Logistic Regression
    [0.69, 0.62, 0.46],  # SVC
    [0.89, 0.93, 0.34],  # Random Forest
    [0.86, 0.72, 0.72],  # CNN
    [0.88, 0.75, 0.75]   # LSTM
])

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(accuracy_data, annot=True, fmt=".2f", cmap="Oranges", xticklabels=embeddings, yticklabels=models)

plt.title('Model F1-score Across Embeddings')
plt.xlabel('Embedding Type')
plt.ylabel('Model')
plt.savefig('plots/heatmap__f1-score.png', format='png')
plt.show()
