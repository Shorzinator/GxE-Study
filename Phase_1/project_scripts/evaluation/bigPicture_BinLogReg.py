import os

import matplotlib.pyplot as plt
import pandas as pd

from Phase_1.project_scripts import get_path_from_root

# Load the data
files = [
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_2_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_3_vs_4_binary_SMOTE_GCV_KF_IT.csv")
]

dfs = [pd.read_csv(file) for file in files]
dfs_test = [df[df['type'] == 'test_metrics'] for df in dfs]

# Overall Performance
overall_performance = pd.concat(dfs_test).groupby('interaction').mean()


# Most Influential Interactions
# (This will give a ranking based on the average metric values across the three binary classifications)
def rank_interactions(metric):
    top_interactions = overall_performance[metric].nlargest(5)
    print(f"Top 5 interactions for {metric} across all binary problems:\n", top_interactions)


for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:  # Modify this list as needed
    rank_interactions(metric)


# Visualization: Averaged metrics across all binary problems
def plot_avg_metric(metric):
    plt.figure(figsize=(14, 6))
    overall_performance[metric].sort_values().plot(kind='barh', color='skyblue')
    plt.title(f"Averaged {metric} across all binary problems", fontsize=15)
    plt.xlabel(metric, fontsize=13)
    plt.ylabel("Interactions", fontsize=13)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:  # Modify this list as needed
    plot_avg_metric(metric)

# ... [You can add more analyses or visualizations as needed]
