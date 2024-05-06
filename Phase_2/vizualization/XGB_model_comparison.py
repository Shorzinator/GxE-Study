import matplotlib.pyplot as plt
import pandas as pd

# Data Preparation
data = {
    'Race': ['1', '2', '3', '4', '5']*2,
    'Model': ['Old Data']*5 + ['New Data']*5,
    'Test Accuracy': [
        0.5927, 0.6675, 0.6935, 0.7291, 0.7129,  # Code 1
        0.7888, 0.8148, 0.7329, 0.8080, 0.7752   # Code 2
    ],
    'Test ROC AUC': [
        0.6673, 0.6835, 0.6620, 0.6880, 0.7181,  # Code 1
        0.8374, 0.8336, 0.7727, 0.8190, 0.8156   # Code 2
    ]
}

df = pd.DataFrame(data)

# Define the races to show on x-axis
race_labels = ['1', '2', '3', '4', '5']

# Plotting Test Accuracy
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test Accuracy'], marker='o', label=f'{label} - Test Accuracy')
plt.title('Test Accuracy Comparison by Race - XGB')
plt.xlabel('Race')
plt.ylabel('Test Accuracy')
plt.xticks(race_labels)
plt.legend(title='Model & Metric')
plt.grid(True)
plt.show()

# Plotting Test ROC AUC
plt.figure(figsize=(10, 6))
for label, df_group in df.groupby('Model'):
    plt.plot(df_group['Race'], df_group['Test ROC AUC'], marker='o', label=f'{label} - Test ROC AUC')
plt.title('Test ROC AUC Comparison by Race - XGB')
plt.xlabel('Race')
plt.ylabel('Test ROC AUC')
plt.xticks(race_labels)
plt.legend(title='Model & Metric')
plt.grid(True)
plt.show()
