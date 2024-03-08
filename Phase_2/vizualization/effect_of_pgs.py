import pickle
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Function to plot feature importances
def plot_feature_importances(importances, title, ax):
    y_pos = np.arange(len(importances))
    ax.barh(y_pos, list(importances.values()), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importances.keys())
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)


# 2. Effect of PGS
def plot_effect_of_pgs(ax):
    model_wo_pgs = load_model('../results/models/classification/AST/BaseModel - XGB/base_model_woPGS.pkl')
    model_w_pgs = load_model('../results/models/classification/AST/BaseModel - XGB/base_model_wPGS.pkl')

    importances_wo_pgs = model_wo_pgs.get_booster().get_score(importance_type='weight')
    importances_w_pgs = model_w_pgs.get_booster().get_score(importance_type='weight')

    plot_feature_importances(importances_wo_pgs, 'Without PGS', ax[0])
    plot_feature_importances(importances_w_pgs, 'With PGS', ax[1])


# Main function to call plotting functions
def main():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_effect_of_pgs(ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
