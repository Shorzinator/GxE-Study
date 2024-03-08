import os
import statistics
from joblib import dump

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split


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


def random_search_tuning(model, params, race_X_train, race_y_train, cv=3, model_path=None, param_path=None,
                         script_name=None, model_type="base"):

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=cv,
        verbose=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    random_search.fit(race_X_train, race_y_train.ravel())

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    file_name = f"{model_type}_{script_name}"
    model_file_path = os.path.join(model_path, f'{file_name}_best_model.joblib')
    params_file_path = os.path.join(param_path, f'{file_name}_best_params.joblib')

    # Save the best model and parameters
    # dump(random_search.best_estimator_, model_file_path)
    # dump(random_search.best_params_, params_file_path)

    return best_model, best_params


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


def evaluate_overfitting(train_accuracy, val_accuracy, y_train_true, y_train_pred, y_val_true, y_val_pred):
    """
    Evaluate the model for overfitting using training and validation accuracies, and F1 scores.

    :param train_accuracy: Accuracy of the model on the training data.
    :param val_accuracy: Accuracy of the model on the validation data.
    :param y_train_true: True labels for the training data.
    :param y_train_pred: Predicted labels for the training data.
    :param y_val_true: True labels for the validation data.
    :param y_val_pred: Predicted labels for the validation data.
    :return: Dictionary containing overfitting evaluation results.
    """

    # Calculate F1 scores for training and validation sets
    f1_train = f1_score(y_train_true, y_train_pred, average='macro')
    f1_val = f1_score(y_val_true, y_val_pred, average='macro')

    # Calculate the difference in F1 scores and accuracies
    f1_diff = f1_train - f1_val
    acc_diff = train_accuracy - val_accuracy

    # Define thresholds for differences that would indicate overfitting
    # These are heuristic values and could be adjusted based on domain knowledge and empirical evidence
    f1_threshold = 0.2
    acc_threshold = 0.2

    # Check for overfitting
    is_overfitting = f1_diff > f1_threshold or acc_diff > acc_threshold

    # Compile results into a dictionary
    results = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'f1_train': f1_train,
        'f1_val': f1_val,
        'f1_diff': f1_diff,
        'acc_diff': acc_diff,
        'is_overfitting': is_overfitting
    }

    return results['is_overfitting']


def get_mapped_data(y_train_old, y_val_old, y_test_old, y_train_new, y_val_new, y_test_new):
    # Map labels to start from 0
    label_mapping_old = {label: i for i, label in enumerate(np.unique(y_train_old))}
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}

    y_train_old_mapped = np.vectorize(label_mapping_old.get)(y_train_old)
    y_val_old_mapped = np.vectorize(label_mapping_old.get)(y_val_old)
    y_test_old_mapped = np.vectorize(label_mapping_old.get)(y_test_old)

    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)
    y_test_new_mapped = np.vectorize(label_mapping_old.get)(y_test_new)

    # Converting the arrays to be 1-D
    y_train_old_mapped = y_train_old_mapped.ravel()
    y_val_old_mapped = y_val_old_mapped.ravel()
    y_test_old_mapped = y_test_old_mapped.ravel()

    y_train_new_mapped = y_train_new_mapped.ravel()
    y_val_new_mapped = y_val_new_mapped.ravel()
    y_test_new_mapped = y_test_new_mapped.ravel()

    return (y_train_old_mapped, y_val_old_mapped, y_test_old_mapped, y_train_new_mapped, y_val_new_mapped,
            y_test_new_mapped)


def prep_data_for_TL(base_model, X_train_new, X_val_new, X_test_new, race_column):
    # Enhance new data with predicted probabilities from the base model
    base_model_probs_train = base_model.predict_proba(X_train_new.drop(columns=[race_column]))
    X_train_new_enhanced = np.hstack([X_train_new.drop(columns=[race_column]), base_model_probs_train])

    base_model_probs_val = base_model.predict_proba(X_val_new.drop(columns=[race_column]))
    X_val_new_enhanced = np.hstack([X_val_new.drop(columns=[race_column]), base_model_probs_val])

    base_model_probs_test = base_model.predict_proba(X_test_new.drop(columns=[race_column]))
    X_test_new_enhanced = np.hstack([X_test_new.drop(columns=[race_column]), base_model_probs_test])

    # Reintroduce 'Race' for race-specific modeling for both training and validation enhanced sets
    X_train_new_enhanced = pd.DataFrame(X_train_new_enhanced)
    X_train_new_enhanced[race_column] = X_train_new[race_column].values

    X_val_new_enhanced = pd.DataFrame(X_val_new_enhanced)
    X_val_new_enhanced[race_column] = X_val_new[race_column].values

    X_test_new_enhanced = pd.DataFrame(X_test_new_enhanced)
    X_test_new_enhanced[race_column] = X_test_new[race_column].values

    return X_train_new_enhanced, X_val_new_enhanced, X_test_new_enhanced


def perform_cross_validation(model, X_train, y_train, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_acc = []
    fold_train_acc = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model.fit(X_train_fold, y_train_fold.ravel())

        # Evaluate on the validation fold
        val_accuracy = accuracy_score(y_val_fold, model.predict(X_val_fold))
        fold_val_acc.append(val_accuracy)

        train_accuracy = accuracy_score(y_train_fold, model.predict(X_train_fold))
        fold_train_acc.append(train_accuracy)

    return model, fold_val_acc, fold_train_acc


def prep_data_for_race_model(X_train_new_enhanced, y_train_new_mapped, X_val_new_enhanced, y_val_new_mapped,
                             X_test_new_enhanced, y_test_new_mapped, race, race_column):

    X_train_race = X_train_new_enhanced[X_train_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_train_race = y_train_new_mapped[X_train_new_enhanced[race_column] == race].ravel()

    X_val_race = X_val_new_enhanced[X_val_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_val_race = y_val_new_mapped[X_val_new_enhanced[race_column] == race].ravel()

    X_test_race = X_test_new_enhanced[X_test_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_test_race = y_test_new_mapped[X_test_new_enhanced[race_column] == race].ravel()

    return X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, params=None, tune=False,
                             check_overfitting=False, race=None, model_type="base", cv=10, resampling="with",
                             script_name=None, outcome="AntisocialTrajectory"):
    """
    Train and evaluate a model with optional hyperparameter tuning and cross-validation.

    Parameters:
    - model: The model to be trained.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - params: Parameter grid for hyperparameter tuning.
    - tune: Whether to perform hyperparameter tuning.
    - check_overfitting: Whether to check if the model is overfitting or not
    - model_type: Type of model ("base" or "final").
    - race: The race identifier for race-specific final models, if applicable.
    - n_splits: Number of splits for cross-validation.
    """

    model_name = f"{model_type} model" + (f" (race {race})" if race else "")

    tag = "AST" if outcome == "AntisocialTrajectory" else "SUT"
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models", "classification", tag)
    param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "param", "classification", tag)

    if tune:
        model, best_params = random_search_tuning(model, params, X_train, y_train, cv=cv, model_path=model_path,
                                                  param_path=param_path, script_name=script_name, model_type=model_type)
        print(f"Best Parameters for {model_name} {resampling} resampling: \n{best_params} \n")

        model.fit(X_train, y_train)

        model_val_accuracy = accuracy_score(y_val, model.predict(X_val))
        print(f"Accuracy for {model_name} on validation set {resampling} resampling: {model_val_accuracy}")

        model_train_accuracy = accuracy_score(y_train, model.predict(X_train))
        print(f"Accuracy for {model_name} on training set {resampling} resampling: {model_train_accuracy}\n")

        model_test_accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy for {model_name} on testing set {resampling} resampling: {model_test_accuracy}\n")

    else:
        model.fit(X_train, y_train)

        model_val_accuracy = accuracy_score(y_val, model.predict(X_val))
        print(f"Accuracy for {model_name} on validation set {resampling} resampling: {model_val_accuracy}")

        model_train_accuracy = accuracy_score(y_train, model.predict(X_train))
        print(f"Accuracy for {model_name} on training set {resampling} resampling: {model_train_accuracy}")

        model_test_accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy for {model_name} on testing set {resampling} resampling: {model_test_accuracy}\n")

    # Check if the model being evaluated is overfitting on the outcome currently under consideration or not.
    if check_overfitting:

        overfitting_results = evaluate_overfitting(
            train_accuracy=model_train_accuracy,
            val_accuracy=model_val_accuracy,
            y_train_true=y_train,
            y_train_pred=model.predict(X_train),
            y_val_true=y_val,
            y_val_pred=model.predict(X_val)
        )

        print(f"Overfitting Evaluation Results for {model_name} {resampling} resampling: {overfitting_results}", "\n")

    return model


def get_model_instance(model_name):
    if model_name == "LogisticRegression":
        return LogisticRegression(multi_class="multinomial", random_state=42)
    elif model_name == "RandomForest":
        return RandomForestClassifier(random_state=42)
    # Add more models as needed
    else:
        raise ValueError(f"Model {model_name} is not supported.")


# Function to load the data splits
def load_data_splits(target_variable, pgs_old="with", pgs_new="with", resampling="without"):
    suffix = "AST" if target_variable == "AntisocialTrajectory" else "SUT"
    X_train_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_train_old_{suffix}.csv")
    X_test_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_test_old_{suffix}.csv")
    X_val_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_val_old_{suffix}.csv")
    y_train_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_train_old_{suffix}.csv")
    y_test_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_test_old_{suffix}.csv")
    y_val_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_val_old_{suffix}.csv")

    X_train_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_train_new_{suffix}.csv")
    X_test_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_test_new_{suffix}.csv")
    X_val_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_val_new_{suffix}.csv")
    y_train_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_train_new_{suffix}.csv")
    y_test_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_test_new_{suffix}.csv")
    y_val_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_val_new_{suffix}.csv")

    return (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
            y_train_old, y_val_old, y_test_old)


def search_spaces():
    # Define search spaces for each model
    search_spaces = {
        'LogisticRegression': {
            # 'penalty': ['l2', 'elasticnet', None], # Including all types of penalties
            'penalty': ['l2'],  # Including all types of penalties
            'C': np.logspace(-5, 5, 50),  # A wider range and more values for regularization strength
            'solver': ['newton-cg', 'lbfgs'],  # Including all solvers
            'max_iter': list(range(100, 30001, 50)),  # More iterations range with finer steps
            'multi_class': ['multinomial', 'ovr'],  # All strategies for handling multiple classes
            # 'l1_ratio': np.linspace(0, 1, 20),  # Relevant for 'elasticnet' penalty, more granular range
            'fit_intercept': [True, False],  # Whether to include an intercept term or not
            'class_weight': [None, 'balanced'],  # Whether to use balanced class weights or not
            # for other cases
            'warm_start': [True, False],  # Whether to reuse the solution of the previous call as initialization
            'tol': np.logspace(-6, -1, 20),  # Tolerance for stopping criteria
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
            'criterion': ['gini', 'entropy'],  # Criterion for measuring the quality of a split
            'splitter': ['best', 'random'],  # Strategy used to choose the split at each node
            'max_depth': [None] + list(range(1, 31)),  # Maximum depth of the tree (None means unlimited)
            'min_samples_split': [2, 3, 4, 5] + list(np.linspace(0.01, 0.2, 20)),  # Minimum number of samples
            # required to split an internal node
            'min_samples_leaf': [1, 2, 3, 4, 5] + list(np.linspace(0.01, 0.1, 10)),  # Minimum number of samples
            # required to be at a leaf node
            'max_features': ['sqrt', 'log2', None] + list(np.linspace(0.1, 1.0, 10)),  # Number of features
            # to consider when looking for the best split
            'max_leaf_nodes': [None] + list(range(10, 101, 10)),  # Maximum number of leaf nodes
            'min_impurity_decrease': np.linspace(0, 0.2, 10),  # Threshold for early stopping in tree growth
            'class_weight': [None, 'balanced'],  # Weights associated with classes
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
