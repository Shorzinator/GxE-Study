import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_performance_comparison(metric_files, title='Model Performance Comparison'):
    """
    Plots a comparison of model performances based on accuracy metrics.

    Parameters:
    - metric_files: A dictionary where keys are descriptive model names,
                    and values are paths to CSV files containing the metrics.

    Example of metric_files:
    {
        'Base Model': 'path/to/base_model_metrics.csv',
        'Intermediate Model': 'path/to/intermediate_model_metrics_tl.csv',
        'Race-Specific Model - Race 1': 'path/to/race_1_model_metrics.csv',
        ...
    }
    """
    performance_data = []

    # Load metrics and aggregate into a single DataFrame
    for model_name, file_path in metric_files.items():
        df = pd.read_csv(file_path)
        df['Model'] = model_name  # Add a column to identify the model
        performance_data.append(df)

    # Concatenate all dataframes
    performance_df = pd.concat(performance_data, ignore_index=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=performance_df, x='Model', y='Accuracy', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.tight_layout()

    # Show plot
    plt.show()


def main():
    unique_races = [1, 2, 3, 4]  # Example list, replace with actual race IDs used in your dataset

    # Metrics with Transfer Learning
    metric_files_with_tl = {
        "Base Model (With TL)": "../results/metrics/classification/HetHieTL/base_model_metrics_with_tl.csv",
        "Intermediate Model (With TL)": "../results/metrics/classification/HetHieTL/"
                                        "AST/BaseModel - RanFor/intermediate_model_metrics_with_tl.csv",
    }

    # Metrics without Transfer Learning
    metric_files_without_tl = {
        "Base Model (Without TL)": "../results/metrics/classification/HetHieTL/base_model_metrics_without_tl.csv",
        "Intermediate Model (Without TL)": "../results/metrics/classification/HetHieTL/AST/BaseModel - RanFor/"
                                           "intermediate_model_metrics_without_tl.csv",
    }

    # Optionally, you can add race-specific models here if needed
    for race_id in unique_races:
        for model_name in ['RandomForest', 'GBC', 'XGBoost', 'CatBoost']:
            key_with_tl = f'Race-Specific Model (With TL) - {model_name} - Race {race_id}'
            key_without_tl = f'Race-Specific Model (Without TL) - {model_name} - Race {race_id}'
            file_path_with_tl = (f'../results/metrics/classification/HetHieTL/{model_name}_race_{race_id}'
                                 f'_metrics_with_tl.csv')
            file_path_without_tl = (f'../results/metrics/classification/HetHieTL/{model_name}_race_{race_id}'
                                    f'_metrics_without_tl.csv')
            metric_files_with_tl[key_with_tl] = file_path_with_tl
            metric_files_without_tl[key_without_tl] = file_path_without_tl

    # Plotting comparisons
    plot_model_performance_comparison(metric_files_with_tl, 'Model Performance Comparison (With Transfer Learning)')
    plot_model_performance_comparison(metric_files_without_tl,
                                      'Model Performance Comparison (Without Transfer Learning)')


if __name__ == "__main__":
    main()
