import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# The list of models
models = ["lr", "lgbm", "xgb", "rf", "gb"]

# Initialize an empty DataFrame to store all results
all_results = pd.DataFrame()

# Loop through each model and read its combined results CSV
for model in models:
    model_results = pd.read_csv(
        f"C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_{model}\\Without IT\\combined_results_RandomizedSearch_SMOTEEN_StratifiedKFold_{model}.csv",
        index_col=0)
    model_results["Model"] = model  # Add a column to indicate the model
    all_results = pd.concat([all_results, model_results], axis=0)

# Reset index for the combined DataFrame
all_results = all_results.reset_index().rename(columns={"index": "Outcome"})

# Calculate average metrics across outcomes
avg_all_results = all_results.groupby(['Model', 'Outcome']).mean().reset_index()

avg_all_results.to_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\average_model_metrics.csv", index=False)

# Metrics to plot
metrics = ['Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score', 'Train ROC AUC',
           'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score', 'Test ROC AUC']

# Print the mean values
print(avg_all_results)

# Plotting
for metric in metrics:
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=avg_all_results, x='Model', y=metric, hue='Outcome', palette='viridis')
    plt.title(f'Comparison of {metric} for Different Models')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.legend(title='Outcome')

    # Show the value on the bar graphs
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')
    plt.show()
