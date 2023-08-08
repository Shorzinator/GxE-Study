import pandas as pd
import matplotlib.pyplot as plt
from Phase_5.project_scripts.utility.data_loader import load_data
from Phase_5.project_scripts.utility.path_utils import get_path_from_root
import os

# Load data
df = load_data()

output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "contingency_table")

# Cross-tabulation
predictors = df.drop(columns=['AntisocialTrajectory', 'SubstanceUseTrajectory'])
target = 'AntisocialTrajectory'     # Replace with the desired column name

for predictor in predictors:
    ct = pd.crosstab(df[predictor], df[target])

    print(ct)

    if predictor in df.columns and target in df.columns :
        # Visualize the crosstab
        plt.figure(figsize=(12, 7))
        ct.plot.bar(stacked=True, figsize=(12, 7))
        plt.title(f'{predictor} vs {target} Distribution')
        plt.ylabel('Count')
        plt.xlabel(predictor)
        plt.tight_layout()
        plt.legend(title=target)
        plot_filename = os.path.join(output_dir_plots, f"{predictor}_{target}-contingency_table.png")
        plt.savefig(plot_filename)  # Save the plot to a file
        plt.show()
        print(f"Contingency plot saved in {output_dir_plots}.")
    else:
        print(f"'{predictor}' or '{target}' not found in the dataset.")
