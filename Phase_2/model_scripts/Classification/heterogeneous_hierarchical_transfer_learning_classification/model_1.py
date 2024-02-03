from copy import deepcopy

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier
from skopt import BayesSearchCV

from Phase_2.model_scripts.Regression.standard_models.model_utils import evaluate_model, tune_random_forest


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
            'n_estimators': (10, 1000),
            'max_depth': (1, 100),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        },
        'LogisticRegression': {
            'C': (1e-6, 1e+6, 'log-uniform'),
        },
        'SVC': {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': (1e-6, 1e+1, 'log-uniform'),
        },
        'GBM': {
            'n_estimators': (10, 1000),
            'learning_rate': (0.001, 1.0, 'log-uniform'),
            'max_depth': (3, 50),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'subsample': (0.5, 1.0),
            'max_features': ['sqrt', 'log2', None],
        },
        'XGBoost': {
            'n_estimators': (10, 1000),
            'learning_rate': (0.001, 1.0, 'log-uniform'),
            'max_depth': (3, 50),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'colsample_bynode': (0.5, 1.0),
            'reg_alpha': (0, 1.0),
            'reg_lambda': (1e-9, 100.0, 'log-uniform'),
        },
    }

    for race in tqdm(X_train_new[race_column].unique(), desc="Training and evaluating race-specific models"):
        if race == 5:  # Skip the 'Other' category
            continue

        race_X_train = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        race_y_train = y_train_new[X_train_new[race_column] == race]
        race_X_test = X_test_new[X_test_new[race_column] == race].drop(columns=[race_column])
        race_y_test = y_test_new[X_test_new[race_column] == race]

        # Define models to be used
        models = {
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'SVC': SVC(probability=True),
            'GBM': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        }

        # Iterate through models and search spaces
        for model_name, model in models.items():
            print(f"Training {model_name} for race {race}")
            opt = BayesSearchCV(
                model,
                search_spaces[model_name],
                n_iter=30,  # Number of parameter settings sampled
                cv=StratifiedKFold(3),
                n_jobs=-1
            )

            # Fit the model
            opt.fit(race_X_train, race_y_train)

            # Best model and parameters
            best_model = opt.best_estimator_
            best_params = opt.best_params_

            # Evaluate the best model
            performance = evaluate_model(best_model, race_X_test, race_y_test)
            print(f'Best {model_name} for race {race}: {best_model}')
            print(f'Performance: {performance}')

            # Store the best model, parameters, and performance
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
    base_model = RandomForestClassifier(n_estimators=100, max_depth=30, bootstrap=False, max_features='log2',
                                        min_samples_split=8, random_state=42)
    base_model_tuned = tune_random_forest(base_model, X_train_old, y_train_old)
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