from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_confusion_matrices(prediction_files, class_labels, title='Confusion Matrix'):
    """
    Plots confusion matrices for different models based on their prediction results.

    Parameters:
    - prediction_files: A dictionary where keys are descriptive model names,
                        and values are paths to CSV files containing the actual and predicted labels.
    - class_labels: List of labels for the classification classes.

    Example of prediction_files:
    {
        'Race-Specific Model - Race 1': 'path/to/race_1_model_predictions.csv',
        ...
    }
    """
    for model_name, file_path in prediction_files.items():
        df = pd.read_csv(file_path)
        actual = df['Actual']
        predicted = df['Predicted']

        # Generate the confusion matrix
        cm = confusion_matrix(actual, predicted, labels=class_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.title(f'{title}: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # Adding accuracy score to the plot
        accuracy = np.trace(cm) / float(np.sum(cm))
        plt.annotate(f'Accuracy: {accuracy:.2f}', xy=(0.05, 1.05), xycoords='axes fraction')

        plt.show()


def main():

    # Assuming you have a list or array of unique race identifiers used in your dataset
    unique_races = [1, 2, 3, 4]  # Example list, replace with actual race IDs used in your dataset

    metric_files = {
        "base_model": "../results/metrics/classification/HetHieTL/base_model_metrics.csv",
        "intermediate_model": "../results/metrics/classification/HetHieTL/intermediate_model_metrics_tl.csv",
    }

    class_labels = ['Low', 'Moderate', 'High', 'Very High']

    # Dynamically add race-specific models
    for race_id in unique_races:
        for model_name in ['RandomForest', 'GBC', 'XGBoost', 'CatBoost']:  # Add or remove model names as needed
            key_name = f'Race-Specific Model - {model_name} - Race {race_id}'
            file_path = f'../results/metrics/classification/HetHieTL/{model_name}_race_{race_id}_metrics.csv'
            metric_files[key_name] = file_path

    plot_confusion_matrices(metric_files, class_labels)


if __name__ == "__main__":
    main()
