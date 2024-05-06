import matplotlib.pyplot as plt
import pandas as pd

# Example data based on your descriptions
data = {
    'Race': ['1', '2', '3', '4', '5']*2,  # Same races for both models
    'Model': ['RF']*5 + ['XGB']*5,
    'Test Accuracy': [
        # RF Test Accuracies for Code 2 (insert your RF data here)
        0.7964, 0.8228, 0.7260, 0.8421, 0.7700,
        # XGB Test Accuracies for Code 2
        0.7888, 0.8148, 0.7329, 0.8080, 0.7752
    ],
    'Test ROC AUC': [
        # RF Test ROC AUC for Code 2 (insert your RF data here)
        0.8355, 0.8369, 0.7685, 0.7861, 0.8327,
        # XGB Test ROC AUC for Code 2
        0.8374, 0.8336, 0.7727, 0.8190, 0.8156
    ]
}

df = pd.DataFrame(data)

# Define the races to show on x-axis
race_labels = ['1', '2', '3', '4', '5']

# Plotting Test Accuracy
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test Accuracy'], marker='o', linestyle='-', label=f'{label} - Test Accuracy')
plt.title('Comparison of Test Accuracy by Race for RF and XGB')
plt.xlabel('Race')
plt.ylabel('Test Accuracy')
plt.xticks(race_labels)
plt.legend(title='Model & Metric')
plt.grid(True)
plt.show()

# Plotting Test ROC AUC
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test ROC AUC'], marker='o', linestyle='-', label=f'{label} - Test ROC AUC')
plt.title('Comparison of Test ROC AUC by Race for RF and XGB')
plt.xlabel('Race')
plt.ylabel('Test ROC AUC')
plt.xticks(race_labels)
plt.legend(title='Model & Metric')
plt.grid(True)
plt.show()
