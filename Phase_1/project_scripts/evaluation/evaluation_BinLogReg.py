import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Phase_1.project_scripts import get_path_from_root

# Define the file paths
files = [
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_2_vs_4_binary_SMOTE_GCV_KF_IT.csv"),
    os.path.join(get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics"),
                 "AST_3_vs_4_binary_SMOTE_GCV_KF_IT.csv")
]

output_dir_plots_ranked = os.path.join(
    get_path_from_root("results", "figures", "gxe_interaction_plots", "BinLogReg_plots"))

# Load the data
dfs = [pd.read_csv(file) for file in files]

# Filter for 'test_metrics'
dfs_test = [df[df['type'] == 'test_metrics'] for df in dfs]


# Plotting function with dynamic top 5 extraction and rank annotations
def plot_ranked_top5(metric, df1, df2, df3, title_suffix='', save_dir=output_dir_plots_ranked):
    # Combine the dataframes and extract top 5 interactions based on metric's value
    combined_df = pd.concat([df1, df2, df3], axis=0)
    top5_df = combined_df.nlargest(5, metric)
    top_interactions = top5_df['interaction'].unique()

    # Assign ranks based on the metric value
    top5_df['rank'] = top5_df[metric].rank(method='first', ascending=False).astype(int)
    rank_dict = top5_df.set_index('interaction')['rank'].to_dict()

    # Filter the datasets by these top interactions
    df1 = df1[df1['interaction'].isin(top_interactions)]
    df2 = df2[df2['interaction'].isin(top_interactions)]
    df3 = df3[df3['interaction'].isin(top_interactions)]

    # Plot
    plt.figure(figsize=(14, 6))
    bar_width = 0.25
    r1 = np.arange(len(df1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    bars1 = plt.bar(r1, df1[metric], color='b', width=bar_width, edgecolor='grey', label='1v4 Case')
    bars2 = plt.bar(r2, df2[metric], color='r', width=bar_width, edgecolor='grey', label='2v4 Case')
    bars3 = plt.bar(r3, df3[metric], color='g', width=bar_width, edgecolor='grey', label='3v4 Case')

    # Add rank annotations inside the bars
    all_bars = [bars1, bars2, bars3]
    datasets = [df1, df2, df3]
    for idx, bars in enumerate(all_bars):
        df = datasets[idx]
        for i, bar in enumerate(bars):
            interaction = df['interaction'].iloc[i]
            rank = rank_dict[interaction]
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{rank}', ha='center',
                     va='center', fontsize=10, color='white', fontweight='bold')

    plt.title(f'{metric} for {title_suffix}', fontweight='bold', fontsize=15)
    plt.xlabel('Interactions', fontweight='bold', fontsize=13)
    plt.ylabel(metric, fontweight='bold', fontsize=13)
    plt.xticks([r + bar_width for r in range(len(df1))], df1['interaction'], rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{metric}_{title_suffix}.png'))
    plt.close()


# List of metrics available in the datasets (excluding non-metric columns)
metrics = list(set(dfs_test[0].columns) & set(dfs_test[1].columns) & set(dfs_test[2].columns))
metrics = [m for m in metrics if m not in ['type', 'interaction', 'Model', 'Target']]

# Generate and save plots for each metric
for metric in metrics:
    plot_ranked_top5(metric, *dfs_test, title_suffix='Ranked Top Interactions for 4.0')

print("All ranked top 5 plots saved successfully!")
