import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
scenarios = ['With Resampling\n(Race Separate)', 'Without Resampling\n(Race Separate)',
             'With Resampling\n(Race Single)', 'Without Resampling\n(Race Single)']
mean_validation_accuracies = [0.7796427066985588, 0.7796427066985588, 0.7993939393939394, 0.7993939393939394]
mean_training_accuracies = [0.8072216803112549, 0.8072216803112549, 0.7995959595959596, 0.7995959595959596]
mean_testing_accuracies = [0.7874602078845097, 0.7874602078845097, 0.7995154451847365, 0.7995154451847365]

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(scenarios, mean_validation_accuracies, label='Mean Validation Accuracy', marker='o')
ax.plot(scenarios, mean_training_accuracies, label='Mean Training Accuracy', marker='s')
ax.plot(scenarios, mean_testing_accuracies, label='Mean Testing Accuracy', marker='^')

# Titles and labels
plt.title('Detailed Model Performance Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Scenario')
plt.ylim(0.77, 0.81)  # Adjust the y-axis to zoom in on the data range of interest
plt.grid(True)
plt.legend()

# Show plot with markers
plt.show()
