import matplotlib.pyplot as plt
import seaborn as sns

from utility.data_loader import load_data

df = load_data()


# Define a function to annotate the bars with counts
def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')


# Filter out NaN values from the 'Race' column
race_data = df.dropna(subset=['Race'])

# Define the category labels for Race
race_labels = {1: 'White', 2: 'Black/African', 3: 'Native', 4: 'Asian', 5: 'Other'}

# Plot for Race distribution with meaningful x-axis labels
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Race', data=race_data, palette='viridis')
ax.set_xticklabels([race_labels.get(x) for x in sorted(race_data['Race'].unique())])
plt.title('Distribution of Race Categories', fontsize=20)
plt.xlabel('Race', labelpad=10, fontsize=14)
plt.ylabel('Count per Category', fontsize=14)
annotate_bars(ax)  # Annotate bars with counts
plt.xticks(rotation=0)  # Rotate labels to avoid overlap
plt.savefig("../results/raw_new_data/race_distribution.png")
plt.show()
