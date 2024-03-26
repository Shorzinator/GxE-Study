import matplotlib.pyplot as plt
import numpy as np

resampling = "with"

# Define race categories and their corresponding accuracies
races = np.array([1.0, 2.0, 3.0, 4.0])

if resampling == "with":
    # Results from the approach of evaluating within a single model (approach 1)
    training_accuracies_approach_1 = np.array([0.8045, 0.7755, 0.7939, 0.8387])
    validation_accuracies_approach_1 = np.array([0.8112, 0.7739, 0.8039, 0.7658])
    testing_accuracies_approach_1 = np.array([0.8088, 0.7697, 0.7547, 0.8182])

    # Results from the approach of modeling each race using a separate model instance (approach 2)
    training_accuracies_approach_2 = np.array([0.8045, 0.7917, 0.7939, 0.8387])
    validation_accuracies_approach_2 = np.array([0.8112, 0.7287, 0.8039, 0.7748])
    testing_accuracies_approach_2 = np.array([0.8088, 0.7580, 0.7547, 0.8283])

else:
    # Results from the approach of evaluating within a single model (approach 1)
    training_accuracies_approach_1 = np.array([0.8045, 0.7755, 0.7939, 0.8387])
    validation_accuracies_approach_1 = np.array([0.8112, 0.7739, 0.8039, 0.7658])
    testing_accuracies_approach_1 = np.array([0.8088, 0.7697, 0.7547, 0.8182])

    # Results from the approach of modeling each race using a separate model instance (approach 2)
    training_accuracies_approach_2 = np.array([0.8045, 0.7917, 0.7939, 0.8387])
    validation_accuracies_approach_2 = np.array([0.8112, 0.7287, 0.8039, 0.7748])
    testing_accuracies_approach_2 = np.array([0.8088, 0.7580, 0.7547, 0.8283])

fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Titles for each plot
titles = ['Training Accuracies by Race', 'Validation Accuracies by Race', 'Testing Accuracies by Race']

# Set of colors for differentiation
colors = ['#1f77b4', '#ff7f0e']

# Accuracies for each plot
accuracies_sets = [
    (training_accuracies_approach_1, training_accuracies_approach_2),
    (validation_accuracies_approach_1, validation_accuracies_approach_2),
    (testing_accuracies_approach_1, testing_accuracies_approach_2)
]

# Plot each set of accuracies
for i, (ax, accuracies, title) in enumerate(zip(axs, accuracies_sets, titles)):
    approach_1, approach_2 = accuracies

    # Approach 1
    ax.plot(races, approach_1, 'o-', color=colors[0], label='Approach 1')

    # Approach 2
    ax.plot(races, approach_2, 's--', color=colors[1], label='Approach 2')

    ax.set_title(title)
    ax.set_xlabel('Race')
    ax.set_ylabel('Accuracy')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

# Set common labels
axs[1].set_ylabel('Accuracy')
axs[2].set_xlabel('Race')
plt.xticks(races, [f'Race {int(r)}' for r in races])

plt.tight_layout()
plt.show()
