import pickle
import matplotlib.pyplot as plt
import numpy as np


# Comparing the base and interim performance between RF and XGB

# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Load your models
base_model_rf = load_model('../results/models/classification/AST/BaseModel - RanFor/base_model.pkl')
intermediate_model_rf = load_model('../results/models/classification/AST/BaseModel - '
                                   'RanFor/intermediate_model.pkl')
base_model_xgb = load_model('../results/models/classification/AST/BaseModel - XGB/base_model_wPGS.pkl')
intermediate_model_xgb = load_model('../results/models/classification/AST/BaseModel - '
                                    'XGB/intermediate_model_wRace_wPGS.pkl')

# Assuming all models use the same set of features
feature_names = ['Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth',
                 'SubstanceUseTrajectory', 'Is_Male', 'InFamilyCluster']  # Replace with your actual feature names


# Updated function to plot feature importances with scaling
def plot_feature_importances(importances, title, feature_names, ax, base_color, highlight_color):
    # Scale importances so that the sum equals 1
    scaled_importances = importances / np.sum(importances)

    # Sort features by importance and get indices of the top 3
    top_3_indices = np.argsort(scaled_importances)[-3:]

    y_pos = np.arange(len(feature_names))
    for i, (y, imp) in enumerate(zip(y_pos, scaled_importances)):
        # Use a different color for top 3 features
        bar_color = highlight_color if i in top_3_indices else base_color
        bar = ax.barh(y, imp, align='center', color=bar_color, edgecolor='black')
        # Shift annotation to the left and add it
        width = bar[0].get_width()
        offset = ax.get_xlim()[1] * 0.02  # Offset to shift annotation to the left
        label_x_pos = width - offset
        ax.text(label_x_pos, y, f'{width:.3f}', va='center', ha='right', color='black', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel('Scaled Importance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.grid(True)  # Add horizontal gridlines


# Continue with plotting, using the updated function and specifying the base and highlight colors
base_color = 'skyblue'
highlight_color = 'darkorange'

# Accessing and scaling feature importances for RF and XGB models
importances_rf_base = base_model_rf.feature_importances_
importances_rf_intermediate = intermediate_model_rf.feature_importances_
importances_xgb_base = base_model_xgb.get_booster().get_score(importance_type='weight')
importances_xgb_intermediate = intermediate_model_xgb.get_booster().get_score(importance_type='weight')

# Example usage for one of the plots:
fig, ax = plt.subplots(figsize=(7, 4))
plot_feature_importances(importances_rf_base, 'RF Base Model Feature Importances', feature_names, ax, base_color,
                         highlight_color)
plt.tight_layout()
plt.show()

# Convert and scale XGB importances
importances_xgb_base_scaled = np.array([importances_xgb_base.get(f, 0) for f in feature_names])
importances_xgb_intermediate_scaled = np.array([importances_xgb_intermediate.get(f, 0) for f in feature_names])

# Creating figures for RF and XGBoost models
fig_rf, axs_rf = plt.subplots(1, 2, figsize=(14, 8))
fig_xgb, axs_xgb = plt.subplots(1, 2, figsize=(14, 8))

# Plotting feature importances with scaling
plot_feature_importances(importances_rf_base, 'RF Base Model Feature Importances', feature_names, axs_rf[0],
                         base_color, highlight_color)
plot_feature_importances(importances_rf_intermediate, 'RF Intermediate Model Feature Importances', feature_names,
                         axs_rf[1], base_color, highlight_color)
plot_feature_importances(importances_xgb_base_scaled, 'XGB Base Model Feature Importances', feature_names, axs_xgb[0],
                         base_color, highlight_color)
plot_feature_importances(importances_xgb_intermediate_scaled, 'XGB Intermediate Model Feature Importances',
                         feature_names, axs_xgb[1], base_color, highlight_color)

# Adjust layout and display the plots
for fig in [fig_rf, fig_xgb]:
    fig.tight_layout(pad=3.0)
    for ax in fig.axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.show()

# Save the Random Forest feature importances figure
fig_rf.savefig('rf_feature_importances.png', bbox_inches='tight')

# Save the XGBoost feature importances figure
fig_xgb.savefig('xgb_feature_importances.png', bbox_inches='tight')


