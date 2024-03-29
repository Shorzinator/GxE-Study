import pandas as pd
import matplotlib.pyplot as plt

from utility.data_loader import load_data_new

# Load your dataset
data_new = load_data_new()

# Calculate the sample distribution percentages for the 'Race' variable
race_distribution_sample = data_new['Race'].value_counts(normalize=True) * 100


# Plot the race distribution from the sample
def plot_race_distribution(race_distribution):
    fig, ax = plt.subplots(figsize=(8, 6))
    race_distribution.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Race Distribution in Sample')
    ax.set_xlabel('Race Category')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels(['White', 'Black/African', 'Native American', 'Asian', 'Other/Multiracial'],
                       rotation=0)
    plt.savefig("")
    plt.show()


# Run the function to generate the plot
plot_race_distribution(race_distribution_sample)
