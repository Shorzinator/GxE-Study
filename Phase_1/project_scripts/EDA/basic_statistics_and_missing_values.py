import pandas as pd
import os
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Setting up directories
output_dir_csv = get_path_from_root("results", "eda_results")
output_dir_plots = get_path_from_root("results", "figures", "eda_plots")

# Check and create directories if they don't exist
if not os.path.exists(output_dir_csv):
    os.makedirs(output_dir_csv)

if not os.path.exists(output_dir_plots):
    os.makedirs(output_dir_plots)

# Load data
df = load_data()

# Basic Statistics
basic_stats = df.describe(include='all').transpose()

# Save to CSV
basic_stats_file = os.path.join(output_dir_csv, "basic_statistics.csv")
basic_stats.to_csv(basic_stats_file)

# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})

# Save to CSV
missing_values_file = os.path.join(output_dir_csv, "missing_values.csv")
missing_df.to_csv(missing_values_file)

print(f"Basic statistics saved to {basic_stats_file}.")
print(f"Missing values analysis saved to {missing_values_file}.")

# In future, any plots generated during EDA can be saved in output_dir_plots.
