import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\Phase_2\\preprocessed_data\\with_resampling\\without_PGS\\AST_new\\y_val_new_AST.csv'
data = pd.read_csv(file_path)

# Count the occurrences of each unique outcome
outcome_counts = data['AntisocialTrajectory'].value_counts()

# Plot the distribution of outcomes
plt.figure(figsize=(10, 6))
outcome_counts.plot(kind='bar')
plt.title('Distribution of Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the plot
plt.show()
