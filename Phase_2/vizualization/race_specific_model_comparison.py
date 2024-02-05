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


# 4. Race-Specific Models
def plot_race_specific_models(ax, races=[1.0, 2.0, 3.0, 4.0]):
    for i, race in enumerate(races):
        model_path = f'../results/models/classification/HetHieTL/AST/XGBClassifier/XGBoost_wPGS_race_{race}.pkl'
        model = load_model(model_path)
        importances = model.get_booster().get_score(importance_type='weight')
        plot_feature_importances(importances, f'Race {race}', ax[i])


# Main function to call plotting functions
def main():
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    plot_race_specific_models(ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
