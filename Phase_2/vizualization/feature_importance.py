import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_feature_importances(feature_importance_files, top_n_features=10, title='Feature Importances Comparison'):
    """
    Plots a comparison of feature importances for different models.

    Parameters:
    - feature_importance_files: A dictionary where keys are descriptive model names,
                                and values are paths to CSV files containing the feature importances.
    - top_n_features: The number of top features to display in the plot.

    Example of feature_importance_files:
    {
        'Base Model': 'path/to/base_model_feature_importances.csv',
        'Race-Specific Model - Race 1': 'path/to/race_1_model_feature_importances.csv',
        ...
    }
    """
    plt.figure(figsize=(15, top_n_features * 0.5 * len(feature_importance_files)))

    for idx, (model_name, file_path) in enumerate(feature_importance_files.items(), start=1):
        df = pd.read_csv(file_path)
        df.sort_values(by='Importance', ascending=False, inplace=True)
        df_top = df.head(top_n_features)

        plt.subplot(len(feature_importance_files), 1, idx)
        sns.barplot(data=df_top, x='Importance', y='Feature', palette='coolwarm')
        plt.title(f'{title}: {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')

    plt.tight_layout()
    plt.show()


def main():

    # Assuming you have a list or array of unique race identifiers used in your dataset
    unique_races = [1, 2, 3, 4]  # Example list, replace with actual race IDs used in your dataset

    metric_files = {
        "base_model": "../results/metrics/classification/HetHieTL/AST/base_model_metrics.csv",
        "intermediate_model": "../results/metrics/classification/HetHieTL/intermediate_model_metrics_tl.csv",
    }

    # Dynamically add race-specific models
    for race_id in unique_races:
        for model_name in ['RandomForest', 'GBC', 'XGBoost', 'CatBoost']:  # Add or remove model names as needed
            key_name = f'Race-Specific Model - {model_name} - Race {race_id}'
            file_path = f'../results/metrics/classification/HetHieTL/{model_name}_race_{race_id}_metrics.csv'
            metric_files[key_name] = file_path


if __name__ == "__main__":
    main()
