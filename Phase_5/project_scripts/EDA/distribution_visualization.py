import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Phase_5.project_scripts.utility.data_loader import load_data
from Phase_5.project_scripts.utility.path_utils import get_path_from_root

# Setting up directories
output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "distribution_visualization")

# Load data
df = load_data()

# List of numerical columns (excluding target variable for now)
numerical_cols = df.select_dtypes(['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('AntisocialTrajectory')  # Assuming 'AntisocialTrajectory' is your target variable

for col in numerical_cols:
    plt.figure(figsize=(12, 7))
    sns.histplot(df[col].dropna(), bins=30, kde=True, edgecolor='k', alpha=0.6)

    # Mean and median lines
    plt.axvline(df[col].mean(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(df[col].median(), color='b', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    # Adjusting position for mean and median text so they don't overlap
    mean_position = df[col].mean() if df[col].mean() < df[col].median() else df[col].mean() * 1.1
    median_position = df[col].median() if df[col].mean() < df[col].median() else df[col].median() * 1.1

    plt.text(mean_position, max_ylim * 0.9, 'Mean: {:.2f}'.format(df[col].mean()), color='r')
    plt.text(median_position, max_ylim * 0.8, 'Median: {:.2f}'.format(df[col].median()), color='b')

    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plot_filename = os.path.join(output_dir_plots, f"{col}_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

print(f"Distribution plots saved in {output_dir_plots}.")
