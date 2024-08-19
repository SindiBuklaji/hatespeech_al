import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Define models and metrics
models = ['Random Forest', 'CNN', 'LSTM', 'SVC', 'Logistic Regression']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# Performance data for each model (this is example data, replace with actual metrics)
data = {
    'Random Forest': [0.89, 0.89, 0.90, 0.89],
    'CNN': [0.86, 0.85, 0.80, 0.85],
    'LSTM': [0.88, 0.88, 0.88, 0.88],
    'SVC': [0.64, 0.71, 0.49, 0.58],
    'Logistic Regression': [0.79, 0.80, 0.78, 0.79],
}

# Number of variables
num_vars = len(metrics)

# Set up the radar chart
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each model
for model in models:
    values = data[model]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.25)

# Add labels for each metric
plt.xticks(angles[:-1], metrics)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Model Performance Comparison')
plt.show()

plt.savefig('plots/radar_rf.png', format='png')