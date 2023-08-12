import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

# Using path_utils to get the path to the evaluation folder
RESULTS_DIR = get_path_from_root("results", "one_vs_all", "logistic_regression_results", "metrics")

# Load the datasets
ast_1_vs_4_df = pd.read_csv(f'{RESULTS_DIR}/AST_1_vs_4_binary_SMOTE_GCV_KF_IT.csv')
ast_2_vs_4_df = pd.read_csv(f'{RESULTS_DIR}/AST_2_vs_4_binary_SMOTE_GCV_KF_IT.csv')
ast_3_vs_4_df = pd.read_csv(f'{RESULTS_DIR}/AST_3_vs_4_binary_SMOTE_GCV_KF_IT.csv')

# Metrics of interest
metrics_of_interest = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]

# Extract metrics for each dataset
ast_1_vs_4_metrics = ast_1_vs_4_df.loc[ast_1_vs_4_df['type'] == 'test_metrics', metrics_of_interest].mean()
ast_2_vs_4_metrics = ast_2_vs_4_df.loc[ast_2_vs_4_df['type'] == 'test_metrics', metrics_of_interest].mean()
ast_3_vs_4_metrics = ast_3_vs_4_df.loc[ast_3_vs_4_df['type'] == 'test_metrics', metrics_of_interest].mean()

# Data preparation for visualization
data = {
    'Metrics': metrics_of_interest * 3,
    'Score': list(ast_1_vs_4_metrics) + list(ast_2_vs_4_metrics) + list(ast_3_vs_4_metrics),
    'Dataset': ['AST 1v4'] * 5 + ['AST 2v4'] * 5 + ['AST 3v4'] * 5
}

# Convert to DataFrame
df_plot = pd.DataFrame(data)

# Create dot plots
plt.figure(figsize=(10, 7))
sns.pointplot(x='Metrics', y='Score', hue='Dataset', data=df_plot, join=False, palette='viridis')
plt.title('Metrics for AST datasets using Logistic Regression')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
