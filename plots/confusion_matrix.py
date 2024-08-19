import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Class distribution
N0 = 9110  # Number of samples in Class 0
N1 = 9110  # Number of samples in Class 1 (after balancing)

# Directory to save plots
output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

# Model Metrics
models_metrics = {
    'Logistic Regression': {'precision': [0.92, 0.33], 'recall': [0.67, 0.73]},
    'SVC': {'precision': [0.92, 0.34], 'recall': [0.68, 0.74]},
    'Random Forest': {'precision': [0.88, 0.22], 'recall': [0.43, 0.73]},
    'CNN': {'precision': [0.71, 0.75], 'recall': [0.78, 0.68]},
    'LSTM': {'precision': [0.78, 0.69], 'recall': [0.63, 0.83]}
}

# Calculate confusion matrix for each model
for model_name, metrics in models_metrics.items():
    precision_0, precision_1 = metrics['precision']
    recall_0, recall_1 = metrics['recall']

    # Calculate TP, FN, FP, TN for Class 0
    TP_0 = recall_0 * N0
    FN_0 = N0 - TP_0
    FP_0 = (TP_0 / precision_0) - TP_0
    TN_0 = N1 - FP_0

    # Calculate TP, FN, FP, TN for Class 1
    TP_1 = recall_1 * N1
    FN_1 = N1 - TP_1
    FP_1 = (TP_1 / precision_1) - TP_1
    TN_1 = N0 - FP_1

    # Round the values to the nearest integer
    TP_0, FN_0, FP_0, TN_0 = np.round([TP_0, FN_0, FP_0, TN_0]).astype(int)
    TP_1, FN_1, FP_1, TN_1 = np.round([TP_1, FN_1, FP_1, TN_1]).astype(int)

    # Create confusion matrix
    confusion_matrix = np.array([[TP_0, FN_0], [FP_0, TN_0]])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Oranges", cbar=False, 
                xticklabels=["Class 0", "Class 1"], 
                yticklabels=["Class 0", "Class 1"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save the plot
    plot_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"Confusion matrix for {model_name} saved as {plot_filename}")