import pickle
import matplotlib.pyplot as plt
import numpy as np


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


def plot_model_evolution(ax):
    base_model = load_model('../results/models/classification/HetHieTL/AST/BaseModel - XGB/base_model_woPGS.pkl')
    intermediate_wo_tl = load_model('../results/models/classification/HetHieTL/AST/BaseModel - '
                                    'XGB/intermediate_model_wRace_woPGS_woTL.pkl')
    intermediate_w_tl = load_model('../results/models/classification/HetHieTL/AST/BaseModel - '
                                   'XGB/intermediate_model_wRace_woPGS_wTL.pkl')

    importances_base = base_model.get_booster().get_score(importance_type='weight')
    importances_wo_tl = intermediate_wo_tl.get_booster().get_score(importance_type='weight')
    importances_w_tl = intermediate_w_tl.get_booster().get_score(importance_type='weight')

    plot_feature_importances(importances_base, 'Base Model', ax[0])
    plot_feature_importances(importances_wo_tl, 'Intermediate without TL', ax[1])
    plot_feature_importances(importances_w_tl, 'Intermediate with TL', ax[2])


# Main function to call plotting functions
def main():

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_model_evolution(ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
