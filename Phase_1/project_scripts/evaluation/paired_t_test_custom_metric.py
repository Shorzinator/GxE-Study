import pandas as pd
from scipy.stats import ttest_rel
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Load the datasets for 1 vs 4 configuration
path_1 = get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics",
                            "without Race", "with SUT", "AST_3_vs_4_with_SUT.csv")
path_2 = get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics",
                            "without Race", "AST_3_vs_4_without_SUT.csv")

ast_1_vs_4_with_sut = pd.read_csv(path_1)
ast_1_vs_4_no_sut = pd.read_csv(path_2)

# Filter out rows with interaction terms involving SUT
ast_1_vs_4_with_sut_filtered = ast_1_vs_4_with_sut[~ast_1_vs_4_with_sut['interaction'].str.contains("SubstanceUseTrajectory")]

# Ensure the orders match - this might not be necessary but is a precaution
ast_1_vs_4_with_sut_filtered = ast_1_vs_4_with_sut_filtered.sort_values(by='interaction')
ast_1_vs_4_no_sut = ast_1_vs_4_no_sut.sort_values(by='interaction')

# Extract Accuracy values for the 1 vs 4 configuration
accuracy_with_sut = ast_1_vs_4_with_sut_filtered['Custom_Metric'].values
accuracy_without_sut = ast_1_vs_4_no_sut['Custom_Metric'].values

# Perform paired t-test
t_stat, p_value = ttest_rel(accuracy_with_sut, accuracy_without_sut)

print(f"t_stat: {t_stat}\n", f"p_value: {p_value}")
