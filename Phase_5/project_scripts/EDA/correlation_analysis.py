import os
import seaborn as sns
import matplotlib.pyplot as plt
from Phase_5.project_scripts.utility.data_loader import load_data
from Phase_5.project_scripts.utility.path_utils import get_path_from_root

# Setting up directories
output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "correlation_analysis")

# Load data
df = load_data()

# Numeric columns (excluding target variable if it's numeric)
numerical_cols = df.select_dtypes(['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('AntisocialTrajectory', errors='ignore')
# Assuming 'AntisocialTrajectory' is your target variable, if not, replace the string

# Compute correlations
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plot_filename = os.path.join(output_dir_plots, "correlation_matrix.png")
plt.tight_layout()
plt.savefig(plot_filename)
plt.close()

print(f"Correlation matrix plot saved in {plot_filename}.")
