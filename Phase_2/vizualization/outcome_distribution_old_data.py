import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utility.data_loader import load_data, load_data_old

data = load_data()


# Define a function to annotate the bars with counts
def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')


# Common color scheme
color_palette = 'viridis'

# Define the category labels for AntisocialTrajectory and SubstanceUseTrajectory
antisocial_labels = {1: 'High Decline', 2: 'Moderate', 3: 'Adolescence-Peak', 4: 'Low'}
substance_use_labels = {1: 'High Use', 2: 'Low Use', 3: 'Typical Use'}

# Adjusted plot for AntisocialTrajectory distribution
plt.figure(figsize=(10, 6))
ax1 = sns.countplot(x='AntisocialTrajectory', data=data, palette=color_palette)
ax1.set_xticklabels([antisocial_labels.get(x) for x in sorted(data['AntisocialTrajectory'].dropna().unique())])
plt.title('Distribution of AntisocialTrajectory Categories in Phase 2')
plt.xlabel('Antisocial Trajectory')
plt.ylabel('Count per Category')
annotate_bars(ax1)  # Annotate bars with counts
plt.savefig("../results/raw_new_data/distribution_AST")
plt.show()

# Adjusted plot for SubstanceUseTrajectory distribution
plt.figure(figsize=(10, 6))
ax2 = sns.countplot(x='SubstanceUseTrajectory', data=data, palette=color_palette)
ax2.set_xticklabels([substance_use_labels.get(x) for x in sorted(data['SubstanceUseTrajectory'].dropna().unique())])
plt.title('Distribution of SubstanceUseTrajectory Categories in Phase 2')
plt.xlabel('Substance Use Trajectory')
plt.ylabel('Count per Category')
annotate_bars(ax2)  # Annotate bars with counts
plt.savefig("../results/raw_new_data/distribution_SUT")
plt.show()
