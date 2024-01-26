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


# Function to train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
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

    # Transfer Learning: Use the model trained on old data as a starting point for the new data
    model_new = deepcopy(model_old)
    model_new.fit(X_train_new, y_train_new)  # Continue training on the new dataset

    # Evaluate the transferred model on the new test set
    r2_new = evaluate_model(model_new, X_test_new, y_test_new)
    print(f'R-squared on New Dataset for {target_variable}: {r2_new}')


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
