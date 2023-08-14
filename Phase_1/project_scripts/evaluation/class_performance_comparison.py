import os

import matplotlib.pyplot as plt
import pandas as pd

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Load datasets
data_AST_1_vs_4 = pd.read_csv(get_path_from_root("results", "one_vs_all", "logistic_regression_results",
                                                 "metrics", "AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv"))
data_AST_2_vs_4 = pd.read_csv(get_path_from_root("results", "one_vs_all", "logistic_regression_results",
                                                 "metrics", "AST_2_vs_4_binary_SMOTE_GCV_KF_IT.csv"))
data_AST_3_vs_4 = pd.read_csv(get_path_from_root("results", "one_vs_all", "logistic_regression_results",
                                                 "metrics", "AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv"))


def compute_weighted_averages_simplified(dataset):
    """
    Compute the weighted average of metrics for predicting classes based on the available columns.
    """
    class_prefixes = sorted(list(set(col.split("_")[0] for col in dataset.columns if "." in col)))
    avgs = {}
    for class_prefix in class_prefixes:
        metrics = [col for col in dataset.columns if col.startswith(class_prefix)]
        avg = dataset[metrics].mean(axis=1).mean()
        avgs[f"Average for Class {class_prefix}"] = avg
    return avgs

# Compute weighted averages
averages_1_vs_4 = compute_weighted_averages_simplified(data_AST_1_vs_4)
averages_2_vs_4 = compute_weighted_averages_simplified(data_AST_2_vs_4)
averages_3_vs_4 = compute_weighted_averages_simplified(data_AST_3_vs_4)
final_accuracy = (data_AST_1_vs_4['Accuracy'].mean() + data_AST_2_vs_4['Accuracy'].mean() + data_AST_3_vs_4['Accuracy'].mean()) / 3

final_metrics_simplified = {
    **averages_1_vs_4,
    **averages_2_vs_4,
    **averages_3_vs_4,
    "Overall Accuracy": final_accuracy
}

# Visualization
classes = list(final_metrics_simplified.keys())
values = list(final_metrics_simplified.values())
plt.figure(figsize=(10, 6))
bars = plt.bar(classes, values, color=['blue', 'green', 'red', 'cyan', 'yellow'])
plt.ylabel('Performance Metric')
plt.title('Performance Comparison across Classes')
plt.ylim(0, 1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=10)


# Save the visualization
save_path_dir = get_path_from_root("results", "figures", "gxe_comparison_plots")
if not os.path.exists(save_path_dir):
    os.makedirs(save_path_dir)
plt.tight_layout()
plt.savefig(os.path.join(save_path_dir, "performance_comparison.png"))
plt.show()
