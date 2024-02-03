from copy import deepcopy

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tqdm import tqdm
import xgboost as xgb

from Phase_2.model_scripts.Regression.model_utils import evaluate_model


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to load the data splits
def load_data_splits(target_variable, pgs_1="with", pgs_2="without"):
    if target_variable == "AntisocialTrajectory":
        X_train_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_old/X_train_old_AST.csv")
        X_test_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_old/X_test_old_AST.csv")
        y_train_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_old/y_train_old_AST.csv")
        y_test_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_old/y_test_old_AST.csv")

        X_train_new = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_new/X_train_new_AST.csv")
        X_test_new = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_new/X_test_new_AST.csv")
        y_train_new = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_new/y_train_new_AST.csv")
        y_test_new = load_data(f"../../preprocessed_data/{pgs_2}_PGS/AST_new/y_test_new_AST.csv")

    else:
        X_train_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/SUT_old/X_train_old_SUT.csv")
        X_test_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/SUT_old/X_test_old_SUT.csv")
        y_train_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/SUT_old/y_train_old_SUT.csv")
        y_test_old = load_data(f"../../preprocessed_data/{pgs_2}_PGS/SUT_old/y_test_old_SUT.csv")

        X_train_new = load_data(f"../../preprocessed_data/{pgs_1}_PGS/SUT_new/X_train_new_SUT.csv")
        X_test_new = load_data(f"../../preprocessed_data/{pgs_1}_PGS/SUT_new/X_test_new_SUT.csv")
        y_train_new = load_data(f"../../preprocessed_data/{pgs_1}_PGS/SUT_new/y_train_new_SUT.csv")
        y_test_new = load_data(f"../../preprocessed_data/{pgs_1}_PGS/SUT_new/y_test_new_SUT.csv")

    return X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old


# Function to train a Random Forest model

def train_model(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, max_depth=30, bootstrap=False, max_features='log2',
                                      min_samples_split=8, random_state=42)
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'SVR':
        model = SVR()
    elif model_type == "GBM":
        model = GradientBoostingRegressor()
    elif model_type == "XGBoost":
        model = xgb()
    # Add more models here as needed
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_race_specific_models(X_train_new, y_train_new, X_test_new, y_test_new, race_column):
    race_models = {}
    r2_scores = {}
    for race in tqdm(X_train_new[race_column].unique(), desc="Training and evaluating race-specific models"):
        if race == 5:  # Skip the 'Other' category
            continue

        race_X_train = X_train_new[X_train_new[race_column] == race].drop(columns=race_column)
        race_y_train = y_train_new[X_train_new[race_column] == race]
        race_X_test = X_test_new[X_test_new[race_column] == race].drop(columns=race_column)
        race_y_test = y_test_new[X_test_new[race_column] == race]

        # Choose the model type based on race or other criteria
        model_type = 'XGBoost' if race == 1 \
            else 'RandomForest' if race == 2 \
            else 'SVR' if race == 3 \
            else 'SVR'
        race_model = train_model(race_X_train, race_y_train, model_type=model_type)
        race_models[race] = race_model

        # Evaluate the model
        r2_score = evaluate_model(race_model, race_X_test, race_y_test)
        r2_scores[race] = r2_score
        print(f'R-squared for race {race} using {model_type}: {r2_score}')

    return race_models, r2_scores


def main(target_variable, race_column="Race"):
    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable))

    # Flatten y_train and y_test if they are not already 1-dimensional
    y_train_new = y_train_new.values.ravel()
    y_test_new = y_test_new.values.ravel()
    y_train_old = y_train_old.values.ravel()
    y_test_old = y_test_old.values.ravel()

    # Step 1: Train base model on old data
    base_model = train_model(X_train_old, y_train_old, model_type='RandomForest')
    r2_base = evaluate_model(base_model, X_test_old, y_test_old)
    print(f"R-squared for base model: {r2_base}")

    # Step 2: Apply transfer learning with new data (excluding race)
    intermediate_model = deepcopy(base_model)
    intermediate_model.fit(X_train_new.drop(columns=race_column), y_train_new)
    r2_intermediate = evaluate_model(intermediate_model, X_test_new.drop(columns=race_column), y_test_new)
    print(f"R-squared for intermediate model (excluding race): {r2_intermediate}")

    # Step 3: Train and evaluate final models for each race using new data
    final_models, r2_scores = train_and_evaluate_race_specific_models(X_train_new, y_train_new, X_test_new, y_test_new,
                                                                      race_column)
    print(f"R-squared for final model: {r2_scores}")


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
