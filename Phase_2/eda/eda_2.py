import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# Load the dataset
file_path = '/Users/shouryamaheshwari/Desktop/GxE-Study/data/raw/Data_GxE_on_EXT_trajectories_new.csv'
df = pd.read_csv(file_path)

# Simple Class Distribution (Bar Graph)
class_distribution = df['AntisocialTrajectory'].value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Class Distribution in AntisocialTrajectory')
plt.ylabel('Count')
plt.xlabel('Class')
plt.savefig('../results/raw_new_data/Class Distribution in AntisocialTrajectory.png')
plt.show()

# Visualizing Class Distribution Against Key Features
# Select a couple of key features for visualization
features = FEATURES  # Example features
features.remove("Is_Male")

for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='AntisocialTrajectory', y=feature, data=df)
    plt.title(f'Distribution of {feature} Across AST Classes')
    plt.savefig(f'../results/raw_new_data/Distribution of {feature} Across AST Classes.png')
    plt.show()
