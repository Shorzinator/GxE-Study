import os

import matplotlib.pyplot as plt
import pandas as pd

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Load the output files
multinomial_file = os.path.join(get_path_from_root("results", "multi_class", "logistic_regression_results", "metrics"),
                                "AST_multinomial_SMOTE_noGCV_noKF_IT.csv")
binary_files = [
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_2_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_3_vs_4_binary_SMOTE_GCV_KF_IT.csv")
]

multinomial_df = pd.read_csv(multinomial_file)
binary_dfs = [pd.read_csv(file) for file in binary_files]

# Metrics we are interested in
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']


# Plotting function
def plot_comparison(metric, binary_df, multinomial_df, title, save_path, class_suffix):
    plt.figure(figsize=(10, 6))

    # Determine the correct column name
    if metric == 'Accuracy':
        column_name = metric
    elif class_suffix in ['macro avg', 'weighted avg']:
        column_name = f"{class_suffix}_{metric}"
    else:
        column_name = f"{class_suffix}_{metric}"

    # Extract metric values for binary and multinomial methods
    bin_value = binary_df[binary_df['type'] == 'test_metrics'][column_name].values[0]
    multi_value = multinomial_df[multinomial_df['type'] == 'test_metrics'][column_name].values[0]

    plt.scatter(['BinLogReg'], [bin_value], color='blue', label='BinLogReg', s=100)
    plt.scatter(['MultLogReg'], [multi_value], color='red', label='MultLogReg', s=100)

    # Annotate the values
    plt.text('BinLogReg', bin_value + 0.01, f"{bin_value:.2f}", ha='center', va='bottom', fontsize=10)
    plt.text('MultLogReg', multi_value + 0.01, f"{multi_value:.2f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel(metric, fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path))
    plt.close()


# Generate plots
class_suffixes = ['1.0', '2.0', '3.0']
for metric in metrics:
    for i, (binary_df, suffix) in enumerate(zip(binary_dfs, class_suffixes)):
        save_path = os.path.join(get_path_from_root("results", "model_comparison_plots"))
        plot_comparison(metric, binary_df, multinomial_df, f"AST {i + 1}v4 - {metric} Comparison", save_path, suffix)

print("All comparison plots saved successfully!")
