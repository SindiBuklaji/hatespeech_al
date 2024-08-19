import matplotlib.pyplot as plt
import numpy as np

# Models and Embedding Types
models = ['CNN', 'LSTM']
embedding_types = ['TF-IDF', 'Word2Vec', 'BERT']

# Accuracy, Precision, Recall, F1-score for each model and embedding type
metrics = {
    'Accuracy': [
        [0.86, 0.88, 0.73],  # CNN
        [0.88, 0.88, 0.73],  # LSTM
    ],
    'Precision': [
        [0.90, 0.90, 0.71],  # CNN
        [0.88, 0.88, 0.78],  # LSTM
    ],
    'Recall': [
        [0.80, 0.80, 0.78],  # CNN
        [0.88, 0.88, 0.63],  # LSTM
    ],
    'F1-score': [
        [0.85, 0.85, 0.74],  # CNN
        [0.88, 0.88, 0.70],  # LSTM
    ]
}

def plot_comparison(metrics, models, embedding_types):
    n_models = len(models)
    n_embeddings = len(embedding_types)
    bar_width = 0.2
    index = np.arange(n_models)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Across Embeddings')

    for i, (metric, data) in enumerate(metrics.items()):
        ax = axes[i//2, i%2]
        for j, embedding in enumerate(embedding_types):
            ax.bar(index + j * bar_width, [data[model_idx][j] for model_idx in range(n_models)], 
                   bar_width, label=embedding)

        ax.set_title(metric)
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(models)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_comparison(metrics, models, embedding_types)
plt.savefig('plots/bar_plot.png', format = 'png') 