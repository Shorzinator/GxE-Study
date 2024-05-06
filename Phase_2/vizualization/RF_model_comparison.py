import matplotlib.pyplot as plt
import pandas as pd

# Updated example data based on your descriptions
data = {
    'Race': [1, 2, 3, 4, 5]*2,  # Only two sets of races for Code 1 and Code 2
    'Model': ['Old Data']*5 + ['New Data']*5,  # Labels indicate old vs. new data
    'Test Accuracy': [
        0.7927, 0.7967, 0.7500, 0.8127, 0.7855,  # Code 1 (Old Data)
        0.7964, 0.8228, 0.7260, 0.8421, 0.7700,  # Code 2 (New Data)
    ],
    'Test ROC AUC': [
        0.7356, 0.7516, 0.7089, 0.6929, 0.7411,  # Code 1 (Old Data)
        0.8355, 0.8369, 0.7685, 0.7861, 0.8327,  # Code 2 (New Data)
    ]
}

df = pd.DataFrame(data)

# Define the races to show on x-axis
race_labels = [1, 2, 3, 4, 5]

# Plotting Test Accuracy
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test Accuracy'], marker='o', label=label)
plt.title('Test Accuracy Comparison by Race - RF')
plt.xlabel('Race')
plt.ylabel('Test Accuracy')
plt.xticks(race_labels)  # Set x-ticks to show only races 1, 2, 3, 4, 5
plt.legend(title='Model Code')
plt.grid(True)
plt.show()

# Plotting Test ROC AUC
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test ROC AUC'], marker='o', label=label)
plt.title('Test ROC AUC Comparison by Race - RF')
plt.xlabel('Race')
plt.ylabel('Test ROC AUC')
plt.xticks(race_labels)  # Set x-ticks to show only races 1, 2, 3, 4, 5
plt.legend(title='Model Code')
plt.grid(True)
plt.show()
