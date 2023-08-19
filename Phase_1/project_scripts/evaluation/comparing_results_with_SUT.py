import os.path

import matplotlib.pyplot as plt
import pandas as pd
from Phase_1.project_scripts.utility.path_utils import get_path_from_root


def refined_compare_metrics(df_with_sut_path, df_without_sut_path, metric1, metric2, title):
    """
    Refined comparison and visualization of two metrics between datasets with and without SUT.

    Args:
    - df_with_sut_path: Path to the DataFrame with SUT as a feature.
    - df_without_sut_path: Path to the DataFrame without SUT as a feature.
    - metric1: First metric to compare (e.g., "Accuracy").
    - metric2: Second metric to compare (e.g., "Custom_Metric").
    - title: Title for the plot.
    """

    # Load the dataframes
    df_with_sut = pd.read_csv(df_with_sut_path)
    df_without_sut = pd.read_csv(df_without_sut_path)

    # Exclude rows with interaction terms involving SUT
    df_with_sut = df_with_sut[~df_with_sut['interaction'].str.contains("SubstanceUseTrajectory")]

    # Extract interactions and metrics
    interactions = df_with_sut['interaction'].unique()
    metric1_with_sut_values = df_with_sut.groupby('interaction')[metric1].mean().values
    metric1_without_sut_values = df_without_sut.groupby('interaction')[metric1].mean().values
    metric2_with_sut_values = df_with_sut.groupby('interaction')[metric2].mean().values
    metric2_without_sut_values = df_without_sut.groupby('interaction')[metric2].mean().values

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))

    # Bar plot for metric1
    bars1 = ax[0].bar(interactions, metric1_with_sut_values, width=0.4, label='With SUT', align='center')
    bars2 = ax[0].bar(interactions, metric1_without_sut_values, width=0.4, label='Without SUT', align='edge')
    ax[0].set_title(f"{metric1} Comparison {title}")
    ax[0].set_ylabel(metric1)
    ax[0].legend()
    ax[0].tick_params(axis='x', rotation=45)

    # Annotate bars with metric values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax[0].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    # Bar plot for metric2
    bars3 = ax[1].bar(interactions, metric2_with_sut_values, width=0.4, label='With SUT', align='center')
    bars4 = ax[1].bar(interactions, metric2_without_sut_values, width=0.4, label='Without SUT', align='edge')
    ax[1].set_title(f"{metric2} Comparison {title}")
    ax[1].set_ylabel(metric2)
    ax[1].legend()
    ax[1].tick_params(axis='x', rotation=45)

    # Annotate bars with metric values
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax[1].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    plt.tight_layout()
    save_path = get_path_from_root("results", "figures", "model_comparison_plots",
                                   "Comparing results with and without SUT")
    if not os.path.exists(save_path) :
        os.makedirs(save_path)

    plt.savefig(f"{save_path}//Comparing metrics {title}.png")
    plt.show()


path_w_SUT = get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics",
                                "without Race", "with SUT", "AST_3_vs_4_with_SUT.csv")
path_wo_SUT = get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics",
                                 "without Race", "AST_3_vs_4_without_SUT.csv")
# Compare the metrics for AST 1 v 4 configuration using the refined function
refined_compare_metrics(path_w_SUT, path_wo_SUT,
                        "Accuracy", "Custom_Metric", "for AST 3 vs 4 Configuration")
