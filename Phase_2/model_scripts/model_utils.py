import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split


# Function to split the data into training and testing sets
# def split_data(df, target):
#     # Split the data into training and testing sets with stratification
#     X_train, X_test, y_train, y_test = train_test_split(
#         df.drop(columns=[target]),
#         df[target],
#         test_size=0.2,
#         random_state=42,
#         stratify=df[target])
#
#     return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


# The split_data function to include a validation set
def split_data(df, target):
    # Split the data into training+validation and testing sets with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target])

    # Further split the training+validation set into separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # This will split the training+validation into 60% training and 20% validation
        random_state=42,
        stratify=y_train_val)

    return pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test), \
        pd.DataFrame(y_train), pd.DataFrame(y_val), pd.DataFrame(y_test)


# Function to evaluate model
# def evaluate_model(model, X_test, y_test, algo_type="regression"):
#     predictions = model.predict(X_test)
#     return r2_score(y_test, predictions)


def evaluate_model(model, X_test, y_test, algo_type="classification"):
    # Get the predicted probabilities for each class
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


# Function to perform randomized search on RandomForestRegressor
def tune_random_forest(model, X_train, y_train):
    # Define the parameter space to explore
    param_distributions = {
        'n_estimators': np.arange(300, 1001, 50),
        'max_depth': [None] + list(np.arange(10, 101, 10)),
        'min_samples_split': np.arange(2, 21, 2),
        'min_samples_leaf': np.arange(1, 21, 2),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Initialize the base model
    rf = model

    # Initialize RandomizedSearchCV
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV to the data
    rf_random.fit(X_train, y_train)

    best_model = rf_random.best_estimator_
    best_params = rf_random.best_params_

    # Return the best estimator
    return best_model, best_params


def random_search_tuning(model, params, race_X_train, race_y_train):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=3,
        verbose=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    random_search.fit(race_X_train, race_y_train)

    # Best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to load the data splits
def load_data_splits(target_variable, pgs_old="with", pgs_new="without", resampling="with"):

    suffix = "AST" if target_variable == "AntisocialTrajectory" else "SUT"
    X_train_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_train_old_{suffix}.csv")
    X_test_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_test_old_{suffix}.csv")
    X_val_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_val_old_{suffix}.csv")
    y_train_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_train_old_{suffix}.csv")
    y_test_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_test_old_{suffix}.csv")
    y_val_old = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_val_old_{suffix}.csv")

    X_train_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_train_new_{suffix}.csv")
    X_test_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_test_new_{suffix}.csv")
    X_val_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_val_new_{suffix}.csv")
    y_train_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_train_new_{suffix}.csv")
    y_test_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_test_new_{suffix}.csv")
    y_val_new = load_data(f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_val_new_{suffix}.csv")

    return (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
            y_train_old, y_val_old, y_test_old)


# Define a function or mapping to determine transfer strategy
def get_transfer_strategy(base_model_type, target_model_type):
    # Example mappings, to be expanded based on actual compatibility
    direct_transfer_compatible = {
        ('RandomForestClassifier', 'XGBClassifier'): True,
        ('RandomForestClassifier', 'GradientBoostingClassifier'): True,
        ('XGBClassifier', 'XGBClassifier'): True,
        ('XGBClassifier', 'GradientBoostingClassifier'): True,

        # Add more pairs as needed
    }
    return direct_transfer_compatible.get((base_model_type, target_model_type), False)


def search_spaces():
    # Define search spaces for each model
    search_spaces = {
        'LogisticRegression': {
            'C': np.logspace(-4, 4, 20),  # Exploring a wide range of regularization strengths
            'penalty': ['l2', 'none'],  # 'l2' is commonly used; 'none' for no regularization
            # 'penalty': ['elasticnet'],  # 'l2' is commonly used; 'none' for no regularization
            'solver': ['lbfgs', 'sag', 'saga'],  # Solvers that support multiclass problems and 'l2' or no penalty
            'max_iter': list(range(100, 5001, 100)),
            # 'l1_ratio': np.linspace(0, 1, 10)  # Only if you include 'elasticnet' in 'penalty'
        },
        'RandomForest': {
            'n_estimators': np.arange(300, 1001, 50),
            'max_depth': [None] + list(np.arange(10, 101, 10)),
            'min_samples_split': np.arange(2, 21, 2),
            'min_samples_leaf': np.arange(1, 21, 2),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'GBC': {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
            'max_depth': [3, 5, 10, 20, 50],
            'min_samples_split': range(2, 11, 2),
            'min_samples_leaf': range(1, 11, 2),
            'subsample': [0.5, 0.75, 1.0],
            'max_features': ['sqrt', 'log2', None],
        },
        'XGBClassifier': {
            'n_estimators': (100, 1000),
            'learning_rate': (0.01, 0.2),
            'max_depth': (3, 10),
            'min_child_weight': (0, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0, 1),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10),
            'max_delta_step': (0, 5),
            'colsample_bylevel': (0.5, 1.0),
        },
        'CatBoost': {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128, 254],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        },
        'DecisionTree': {
            'max_depth': (1, 30),  # Continuous space, but you'll need to round to integer values when using
            'min_samples_split': (0.01, 0.2),  # Represented as a fraction of the total number of samples
            'min_samples_leaf': (0.01, 0.1),  # Represented as a fraction of the total number of samples
        },
        'LightGBM': {
            'num_leaves': np.linspace(20, 400, 10).astype(int),  # More granular range, still need to round to integers
            'learning_rate': np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=10),  # Log-uniform distribution
            'min_child_samples': np.linspace(5, 200, 10).astype(int),  # Broader range, round to integers
            'subsample': np.linspace(0.5, 1.0, 20),  # More granular control
            'colsample_bytree': np.linspace(0.5, 1.0, 10),
            'max_depth': np.linspace(-1, 15, 5).astype(int),  # Including max_depth, round to integers
            'min_split_gain': np.linspace(0.0, 1.0, 10),
            'reg_alpha': np.linspace(0.0, 1.0, 10),
            'reg_lambda': np.linspace(0.0, 1.0, 10),
            # 'max_bin': np.linspace(200, 300, 10)  # Round to integers
        },
    }

    return search_spaces
