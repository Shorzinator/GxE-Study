from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

from Phase_2.model_scripts.model_utils import evaluate_model


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), name="new_train"):
    """
    Generate a simple plot of the test and training learning curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    plt.savefig(f"../results/modeling/{name}_learning_curve")
    plt.show()


def plot_feature_importance(model, feature_names, X_train, name="new_train"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(15, 7))  # Adjusted size
    plt.title("Feature Importances", fontsize=16)
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=12)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"../results/modeling/{name}_feature_importance")
    plt.show()


# Function to train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    return model


# Main function to run the model training and evaluation
def main(target_variable):
    if target_variable == "AntisocialTrajectory":
        X_train_old = load_data("../preprocessed_data/AST_old/X_train_old_AST.csv")
        X_test_old = load_data("../preprocessed_data/AST_old/X_test_old_AST.csv")
        y_train_old = load_data("../preprocessed_data/AST_old/y_train_old_AST.csv")
        y_test_old = load_data("../preprocessed_data/AST_old/y_test_old_AST.csv")

        X_train_new = load_data("../preprocessed_data/AST_new/X_train_new_AST.csv")
        X_test_new = load_data("../preprocessed_data/AST_new/X_test_new_AST.csv")
        y_train_new = load_data("../preprocessed_data/AST_new/y_train_new_AST.csv")
        y_test_new = load_data("../preprocessed_data/AST_new/y_test_new_AST.csv")

    else:
        X_train_old = load_data("../preprocessed_data/SUT_old/X_train_old_SUT.csv")
        X_test_old = load_data("../preprocessed_data/SUT_old/X_test_old_SUT.csv")
        y_train_old = load_data("../preprocessed_data/SUT_old/y_train_old_SUT.csv")
        y_test_old = load_data("../preprocessed_data/SUT_old/y_test_old_SUT.csv")

        X_train_new = load_data("../preprocessed_data/SUT_new/X_train_new_SUT.csv")
        X_test_new = load_data("../preprocessed_data/SUT_new/X_test_new_SUT.csv")
        y_train_new = load_data("../preprocessed_data/SUT_new/y_train_new_SUT.csv")
        y_test_new = load_data("../preprocessed_data/SUT_new/y_test_new_SUT.csv")

    # Reshape y_train_old and y_train_new to be 1-dimensional
    y_train_old = y_train_old.values.ravel()
    y_train_new = y_train_new.values.ravel()

    # Train the model on the old data
    model_old = train_model(X_train_old, y_train_old)
    r2_old = evaluate_model(model_old, X_test_old, y_test_old)
    print(f'R-squared on Old Dataset for {target_variable}: {r2_old}')

    # Plot learning curve for the old data model
    # plot_learning_curve(model_old, f"Learning Curve for {target_variable} (Old Data)", X_train_old, y_train_old,
    #                     cv=5, name="old_train")

    # Plot feature importance for the old data model
    # plot_feature_importance(model_old, X_train_old.columns, X_train_old, name="old_train")

    # Transfer Learning: Use the model trained on old data as a starting point for the new data
    model_new = deepcopy(model_old)
    model_new.fit(X_train_new, y_train_new)  # Continue training on the new dataset

    # Evaluate the transferred model on the new test set
    r2_new = evaluate_model(model_new, X_test_new, y_test_new)
    print(f'R-squared on New Dataset for {target_variable}: {r2_new}')

    # Plot learning curve for the new data model
    # plot_learning_curve(model_new, f"Learning Curve for {target_variable} (New Data)", X_train_new, y_train_new,
    #                     cv=5)
    # plot_feature_importance(model_new, X_train_new.columns, X_train_new)


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
