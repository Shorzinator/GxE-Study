import os
import matplotlib.pyplot as plt
import seaborn as sns
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

sns.set(style="whitegrid")

# Setting up directories
output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "boxplot_visualization")

# Load data
df = load_data()

# Numeric columns (excluding target variable if it's numeric)
numerical_cols = df.select_dtypes(['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('AntisocialTrajectory', errors='ignore')
# Assuming 'AntisocialTrajectory' is your target variable, if not, replace the string

for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[col].dropna())
    plt.title(f'Boxplot for {col}')
    plt.xlabel(col)
    plot_filename = os.path.join(output_dir_plots, f"{col}_boxplot.png")
    plt.savefig(plot_filename)
    plt.close()

print(f"Boxplots saved in {output_dir_plots}.")
