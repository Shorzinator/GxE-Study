import os

import matplotlib.pyplot as plt
import pandas as pd

from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

MODEL_NAME = "logistic_regression"


# Define a function to compute and plot differences
def plot_differences(file_with_IT, file_without_IT, title_suffix):
    # Load the metrics
    with_IT = pd.read_csv(get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results", "metrics",
                                             f"{file_with_IT}.csv"))
    without_IT = pd.read_csv(get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results", "metrics",
                                                f"{file_without_IT}.csv"))

    # Specify the metrics to compare
    metrics_to_compare = ['Accuracy', 'Custom_Metric', 'Matthews Correlation Coefficient',
                          '1.0_Precision', '1.0_Recall', '1.0_F1-Score',
                          '3.0_Precision', '3.0_Recall', '3.0_F1-Score']

    # Adjust metrics if required
    if "2_vs_3" in file_with_IT or "2_vs_3" in file_without_IT:
        metrics_to_compare[3:6] = ['2.0_Precision', '2.0_Recall', '2.0_F1-Score']

    # Calculate the average metrics for test data in the file with interaction terms
    with_IT_avg = with_IT[with_IT['type'] == 'test_metrics'][metrics_to_compare].mean()

    # Grab the test metrics from the file without interaction terms
    without_IT_test = without_IT[without_IT['type'] == 'test_metrics'].iloc[0]

    # Compute the differences using the average metrics
    differences = with_IT_avg - without_IT_test[metrics_to_compare]

    # Plotting the differences with enhancements
    plt.figure(figsize=(14, 7))

    # Use different colors for positive and negative differences
    bar_colors = ['limegreen' if diff > 0 else 'tomato' for diff in differences]

    bars = plt.bar(differences.index, differences.values, color=bar_colors, width=0.6)

    # Annotations directly on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (0.005 if yval > 0 else -0.015),
                 f"{yval:.3f}", ha='center', va='center', fontsize=9, fontweight='bold')

    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(
        f"Difference in Average Metrics: With Interaction Terms vs Without Interaction Terms ({title_suffix})",
        fontsize=15, fontweight='bold')
    plt.ylabel("Difference", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    path = get_path_from_root("results", "evaluation", "comparison")
    ensure_directory_exists(path)
    plt.savefig(os.path.join(path, f"difference_plot_{title_suffix}.png"))  # Save the plot
    plt.show()


# Plot differences for all SUT tasks
plot_differences("SUT_1_vs_3", "SUT_1_vs_3_noIT", "SUT_1_vs_3")
plot_differences("SUT_2_vs_3", "SUT_2_vs_3_noIT", "SUT_2_vs_3")
