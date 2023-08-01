# Comparing noIT, oneIT, allIT

import pandas as pd

# Load the dataframes
df_all_interactions = pd.read_csv('AST_KNN_SMOTE_allInteractionTerms.csv')
df_one_interaction = pd.read_csv('AST_KNN_SMOTE_oneInteractionTerm.csv')
df_no_interactions = pd.read_csv('AST_KNN_SMOTE_noInteractionTerms.csv')

# Add new column indicating the type of interaction term used
df_all_interactions['Interaction_Terms'] = 'All'
df_one_interaction['Interaction_Terms'] = 'One'
df_no_interactions['Interaction_Terms'] = 'None'

# Concatenate the dataframes
df_all = pd.concat([df_all_interactions, df_one_interaction, df_no_interactions])

# Save the combined results
df_all.to_csv('All_Model_Results.csv', index=False)

# Load the results from the CSV file
results = pd.read_csv('All_Model_Results.csv')

# Metrics list
metrics = ['Precision', 'Recall', 'F1', 'ROC_AUC', 'Accuracy', 'Log_Loss', 'Custom_Score']

# Find the best interaction term for each model based on each metric
best_interaction_per_model_per_metric = {}
for metric in metrics:
    best_interaction_per_model_per_metric[metric] = results.loc[results.groupby('Model')[metric].idxmax()]

# Display
print("Best Model Per Metric:")
for metric, df in best_interaction_per_model_per_metric.items():
    print(f"\nFor {metric}:")
    print(df[['Model', 'Interaction_Terms', metric]].sort_values(metric, ascending=False))
