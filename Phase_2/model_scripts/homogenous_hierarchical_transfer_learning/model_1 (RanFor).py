from copy import deepcopy

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from Phase_2.model_scripts.model_utils import evaluate_model, tune_random_forest


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
def train_model(X_train, y_train):
    # Tuned parameters: n_estimators=650, max_depth=30, bootstrap=False, max_features='log2',
    #                                       min_samples_split=8, random_state=42
    model = RandomForestRegressor(n_estimators=650, random_state=42, max_depth=30, bootstrap=False, max_features="log2",
                                  min_samples_split=8)
    model.fit(X_train, y_train)
    return model


# Function to train race-specific models
def train_race_specific_models(X_train, y_train, race_column, base_model_params):
    race_models = {}
    for race in tqdm(X_train[race_column].unique(), desc="Training race-specific models"):
        print(f"Training model for race: {race}")  # Troubleshooting print statement
        if race == 5:  # Skip the 'Other' category
            continue
        race_X_train = X_train[X_train[race_column] == race].drop(columns=race_column)
        race_y_train = y_train[X_train[race_column] == race]

        # Initialize the model with base model's parameters
        race_model = RandomForestRegressor(**base_model_params)
        race_model.fit(race_X_train, race_y_train)
        race_models[race] = race_model
    return race_models


# Function to evaluate race-specific models
def evaluate_race_specific_models(race_models, X_test, y_test, race_column):
    r2_scores = {}
    for race, model in race_models.items():
        print(f"Evaluating model for race: {race}")  # Troubleshooting print statement

        race_X_test = X_test[X_test[race_column] == race].drop(columns=race_column)
        if race_X_test.empty:
            print(f"No samples for race {race} in the test set.")
            continue
        race_y_test = y_test[X_test[race_column] == race]
        r2 = evaluate_model(model, race_X_test, race_y_test)
        r2_scores[race] = r2
        print(f'R-squared for race {race}: {r2}')
    return r2_scores


# Function to train race-specific models using transfer learning
def transfer_learn_race_models(X_train_old, y_train_old, X_train_new, y_train_new, race_column):
    # Train a base model on the old data
    base_model = train_model(X_train_old, y_train_old)

    # Initialize race-specific models with the trained base model's parameters
    race_models = {}
    for race in X_train_new[race_column].unique().sort():
        race_X_train_new = X_train_new[X_train_new[race_column] == race].drop(columns=race_column)
        race_y_train_new = y_train_new[X_train_new[race_column] == race]

        # Clone the base model for each race and fine-tune it on the new race-specific data
        race_model = deepcopy(base_model)
        race_model.fit(race_X_train_new, race_y_train_new)
        race_models[race] = race_model

    return race_models


# Main function to run the model training and evaluation
def main(target_variable, race_column="Race"):
    # Load data
    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable))

    # Flatten y_train and y_test if they are not already 1-dimensional
    y_train_new = y_train_new.values.ravel()
    y_test_new = y_test_new.values.ravel()
    y_train_old = y_train_old.values.ravel()
    y_test_old = y_test_old.values.ravel()

    # Step 1: Train base model on old data
    base_model = train_model(X_train_old, y_train_old)

    # print("Tuning RandomForestRegressor on old data...")
    # initial_model = RandomForestRegressor(random_state=42)  # Initialize the base model to be tuned
    # base_model = tune_random_forest(initial_model, X_train_old, y_train_old)
    # print("Best base model parameters found:", base_model.get_params())

    # Evaluate the base model
    r2_base = evaluate_model(base_model, X_test_old, y_test_old)
    print(f"R-squared for base model: {r2_base}")

    # Step 2: Apply transfer learning with new data (excluding race)
    intermediate_model = deepcopy(base_model)
    intermediate_model.fit(X_train_new.drop(columns=race_column), y_train_new)

    # Evaluate intermediate model on the new test set (excluding race)
    r2_intermediate = evaluate_model(intermediate_model, X_test_new.drop(columns=race_column), y_test_new)
    print(f"R-squared for intermediate model (excluding race): {r2_intermediate}")

    # Step 3: Train final models for each race using new data
    final_models = {}
    for race in X_train_new[race_column].unique():
        race_X_train = X_train_new[X_train_new[race_column] == race].drop(columns=race_column)
        race_y_train = y_train_new[X_train_new[race_column] == race]

        # Initialize the final model for this race with base model's structure
        final_model = deepcopy(base_model)
        final_model.fit(race_X_train, race_y_train)
        final_models[race] = final_model

        # Evaluate the final model for this race on new test set
        race_X_test = X_test_new[X_test_new[race_column] == race].drop(columns=race_column)
        race_y_test = y_test_new[X_test_new[race_column] == race]
        r2_final = evaluate_model(final_model, race_X_test, race_y_test)
        print(f"R-squared for final model (race {race}): {r2_final}")


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
