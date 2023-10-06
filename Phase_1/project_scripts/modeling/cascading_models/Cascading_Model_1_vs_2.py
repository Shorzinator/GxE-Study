import os

import matplotlib.pyplot as plt
import pandas as pd

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Paths to the results
MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results", "metrics")
GRAPH_OUTPUT_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results", "graphs")

if not os.path.exists(GRAPH_OUTPUT_DIR):
    os.makedirs(GRAPH_OUTPUT_DIR)


# Load the performance metrics from CSV files
def load_performance_data(filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, filename))


# Create a difference bar plot for Model 1 vs Model 2 performance metrics
def plot_difference_performance(data1, data2, model1_name, model2_name):
    labels = list(data1.columns)
    differences = data1.iloc[0].values - data2.iloc[0].values

    plt.figure(figsize=(12, 7))
    plt.bar(labels, differences, color=['red' if diff < 0 else 'green' for diff in differences])

    for i, diff in enumerate(differences):
        plt.text(i, diff + 0.001, f"{diff:.3f}", ha='center', va='bottom', fontsize=10)

    plt.title(f'Difference in Scores: {model1_name} - {model2_name}')
    plt.ylabel('Difference in Score')
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, f'difference_performance_{model1_name}_vs_{model2_name}.png'))
    plt.close()


def main():
    model1_names = [
        'AntisocialTrajectory_1_vs_4_Model1',
        'AntisocialTrajectory_2_vs_4_Model1',
        'AntisocialTrajectory_3_vs_4_Model1'
    ]

    model2_names = [
        'AntisocialTrajectory_1_vs_4_Model2',
        'AntisocialTrajectory_2_vs_4_Model2',
        'AntisocialTrajectory_3_vs_4_Model2'
    ]

    for model1_name, model2_name in zip(model1_names, model2_names):
        data1 = load_performance_data(f"{model1_name}_performance.csv")
        data2 = load_performance_data(f"{model2_name}_performance.csv")

        plot_difference_performance(data1, data2, model1_name, model2_name)


if __name__ == "__main__":
    main()
