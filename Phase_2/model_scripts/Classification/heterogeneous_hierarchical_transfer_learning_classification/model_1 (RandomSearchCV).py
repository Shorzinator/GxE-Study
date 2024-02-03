from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Phase_2.model_scripts.model_utils import evaluate_model, random_search_tuning


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to load the data splits
def load_data_splits(target_variable, pgs_1="with", pgs_2="without"):
    if target_variable == "AntisocialTrajectory":
        X_train_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_old/X_train_old_AST.csv")
        X_test_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_old/X_test_old_AST.csv")
        y_train_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_old/y_train_old_AST.csv")
        y_test_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_old/y_test_old_AST.csv")

        X_train_new = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_new/X_train_new_AST.csv")
        X_test_new = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_new/X_test_new_AST.csv")
        y_train_new = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_new/y_train_new_AST.csv")
        y_test_new = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/AST_new/y_test_new_AST.csv")

    else:
        X_train_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/SUT_old/X_train_old_SUT.csv")
        X_test_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/SUT_old/X_test_old_SUT.csv")
        y_train_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/SUT_old/y_train_old_SUT.csv")
        y_test_old = load_data(f"../../../preprocessed_data/{pgs_2}_PGS/SUT_old/y_test_old_SUT.csv")

        X_train_new = load_data(f"../../../preprocessed_data/{pgs_1}_PGS/SUT_new/X_train_new_SUT.csv")
        X_test_new = load_data(f"../../../preprocessed_data/{pgs_1}_PGS/SUT_new/X_test_new_SUT.csv")
        y_train_new = load_data(f"../../../preprocessed_data/{pgs_1}_PGS/SUT_new/y_train_new_SUT.csv")
        y_test_new = load_data(f"../../../preprocessed_data/{pgs_1}_PGS/SUT_new/y_test_new_SUT.csv")

    return X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old


# Updated function to train and evaluate race-specific models with real-time hyperparameter tuning
def train_and_evaluate_race_specific_models(X_train_new, y_train_new, X_test_new, y_test_new, race_column):
    race_models = {}
    race_best_params = {}
    performance_metrics = {}

    # Define search spaces for each model
    search_spaces = {
        'RandomForest': {
            'n_estimators': range(200, 1000, 50),
            'max_depth': [None] + list(range(1, 101, 10)),
            'min_samples_split': range(2, 11, 2),
            'min_samples_leaf': range(1, 11, 2),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        },
        'SVC': {
            'C': [10 ** i for i in range(-6, 7)],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + [10 ** i for i in range(-9, 4)],
        },
        'GBM': {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
            'max_depth': [3, 5, 10, 20, 50],
            'min_samples_split': range(2, 11, 2),
            'min_samples_leaf': range(1, 11, 2),
            'subsample': [0.5, 0.75, 1.0],
            'max_features': ['sqrt', 'log2', None],
        },
        'XGBoost': {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
            'max_depth': [3, 5, 10, 20, 50],
            'min_child_weight': range(1, 11, 2),
            'subsample': [0.5, 0.75, 1.0],
            'colsample_bytree': [0.5, 0.75, 1.0],
            'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
            'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
        },
        'CatBoost': {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128, 254],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        }
    }

    for race in X_train_new[race_column].unique():

        print()

        if race == 5:  # Skip the 'Other' category
            continue

        race_X_train = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        race_y_train = y_train_new[X_train_new[race_column] == race]
        race_X_test = X_test_new[X_test_new[race_column] == race].drop(columns=[race_column])
        race_y_test = y_test_new[X_test_new[race_column] == race]

        # Map labels to start from 0 if necessary
        unique_labels = np.unique(race_y_train)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        inv_label_mapping = {i: label for label, i in label_mapping.items()}  # For inverse mapping after prediction

        race_y_train_mapped = np.vectorize(label_mapping.get)(race_y_train)

        # Define models to be used
        models = {
            # 'RandomForest': RandomForestClassifier(),
            # 'GBM': GradientBoostingClassifier(),
            # 'XGBoost': XGBClassifier(eval_metric='logloss'),
            'CatBoost': CatBoostClassifier(verbose=False, one_hot_max_size=4),
        }

        # Iterate through models and search spaces
        for model_name, model in models.items():
            print(f"Training {model_name} for race {race}\n")

            # Without Tuning
            # predictions = model.predict(race_X_test)

            # With Tuning
            best_model, best_params = random_search_tuning(model, model_name, search_spaces, race_X_train,
                                                           race_y_train)
            predictions = best_model.predict(race_X_test)

            # Map predictions back to original labels
            predictions = np.vectorize(inv_label_mapping.get)(predictions)
            performance = accuracy_score(race_y_test, predictions)

            # Evaluate the best model
            print(f'Best {model_name} for race {race}: {model}')
            print(f'Performance: {performance}')

            # Store the best model, parameters, and performance
            # race_models[(race, model_name)] = model
            race_models[(race, model_name)] = best_model
            race_best_params[(race, model_name)] = best_params
            performance_metrics[(race, model_name)] = performance

    return race_models, race_best_params, performance_metrics


def train_base_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=30, bootstrap=False, max_features='log2',
                                   min_samples_split=8, random_state=42)
    model.fit(X_train, y_train)

    return model


def main(target_variable, race_column="Race"):
    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable))

    # Flatten y_train and y_test
    y_train_new, y_test_new, y_train_old, y_test_old = [
        y.values.ravel() for y in (y_train_new, y_test_new, y_train_old, y_test_old)
    ]

    # Step 1: Train base model on old data
    base_model_tuned = RandomForestClassifier(n_estimators=100, max_depth=30, bootstrap=False, max_features='log2',
                                              min_samples_split=8, random_state=42)
    # base_model_tuned = tune_random_forest(base_model, X_train_old, y_train_old)
    base_model_tuned.fit(X_train_old, y_train_old)

    base_model_accuracy = evaluate_model(base_model_tuned, X_test_old, y_test_old)
    print(f"Accuracy for base model: {base_model_accuracy}")

    # Step 2: Retrain base model on new data (excluding race) as an intermediate model
    intermediate_model = deepcopy(base_model_tuned)
    intermediate_model.fit(X_train_new.drop(columns=[race_column]), y_train_new)
    intermediate_accuracy = evaluate_model(intermediate_model, X_test_new.drop(columns=[race_column]), y_test_new)
    print(f"Accuracy for intermediate model (excluding race): {intermediate_accuracy}")

    # Step 3: Train and evaluate race-specific models
    final_models, race_best_params, performance_metrics = train_and_evaluate_race_specific_models(
        X_train_new, y_train_new, X_test_new, y_test_new, race_column)
    print(f"Performance metrics for final models: {performance_metrics}")


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
