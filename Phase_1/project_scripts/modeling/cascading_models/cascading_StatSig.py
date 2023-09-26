import os

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Paths to the results
MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results", "metrics")


# Load the performance metrics from CSV files
def load_performance_data(filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, filename))


def check_statistical_significance(data1, data2, model1_name, model2_name):
    for column in data1.columns:
        metric1 = data1[column].values
        metric2 = data2[column].values

        # Paired t-test
        t_stat, t_p_val = ttest_rel(metric1, metric2)
        print(f"{column} - Paired t-test: t = {t_stat:.3f}, p = {t_p_val:.3f}")

        # Wilcoxon signed-rank test
        w_stat, w_p_val = wilcoxon(metric1, metric2)
        print(f"{column} - Wilcoxon signed-rank test: statistic = {w_stat}, p = {w_p_val:.3f}")
        print("-" * 50)


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

        print(f"Testing statistical significance for {model1_name} vs {model2_name}:\n")
        check_statistical_significance(data1, data2, model1_name, model2_name)


if __name__ == "__main__":
    main()
