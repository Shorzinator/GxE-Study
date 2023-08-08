import os
import matplotlib.pyplot as plt
import seaborn as sns
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

sns.set(style="whitegrid", palette="pastel")

# Setting up directories
output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "categorical_feature_analysis")

# Load data
df = load_data()

# Selecting the 'Race' column
col = 'Race'
if col in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.ylabel(col)
    plt.xlabel('Count')
    plot_filename = os.path.join(output_dir_plots, f"{col}_countplot.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"'{col}' feature plot saved in {output_dir_plots}.")
else:
    print(f"Column '{col}' not found in the dataset.")
