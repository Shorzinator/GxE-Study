import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from Phase_1.config import COMBINED, IT
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_interaction_terms(df, feature_pairs):
    """
    Generate interaction terms for specified feature pairs iteratively.

    Args:
    :param df: (pd.DataFrame) Original dataset.
    :param feature_pairs: (list of tuples) List of tuples where each tuple contains feature columns
                          for which interaction terms are to be generated.
    :return: df (pd.DataFrame): Dataset with added interaction terms.
    """

    logger.info(f"Generating interaction term for features: {feature_pairs}\n")

    # Create the interaction term
    interaction_column_name = f"{feature_pairs[0]}_x_{feature_pairs[1]}"
    df[interaction_column_name] = df[feature_pairs[0]] * df[feature_pairs[1]]

    # logger.info(f"Updated columns are: {df.columns}\n")

    return df


def save_results(model_name, target, type_of_classification, model_type, results, dir):
    """
    Save the results in a structured directory and file.
    :param dir: model_dir or metrics_dir
    :param model_type: multi_class or one_vs_rest
    :param type_of_classification: multinomial, binary, etc.
    :param model_name: Name of the model (e.g., "xgboost")
    :param target: Target variable (either "AST" or "SUT")
    :param results: The results data (a dictionary)
    """
    logger.info("Saving results ...\n")

    # Flatten the results dictionary
    flat_results = {}
    for split, metrics in results.items():
        for metric, value in metrics.items():
            flat_results[f"{split}_{metric}"] = value

    # Convert the flattened dictionary to a dataframe
    results_df = pd.DataFrame([flat_results])

    dir_path = dir

    # Check and create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Define the path for saving
    if model_type == "multi_class":
        results_file = os.path.join(dir_path, f"{target}_results_{type_of_classification}_{COMBINED}_{IT}.csv")
    else:  # For one_vs_all, keep the original naming convention
        results_file = os.path.join(dir_path, f"{target}_results_{type_of_classification}_{COMBINED}_{IT}.csv")

    # Save to CSV
    results_df.to_csv(results_file, index=False)


def calculate_metrics(y_true, y_pred, model_name, target, type):
    """
    Calculate metrics for the multinomial model predictions.
    :param type: test or train
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param model_name: Name of the model
    :param target: Target column name
    :return: A dictionary containing the calculated metrics
    """

    # Log unique classes for validation
    # logger.info(f"Unique classes in true labels for {type}: {set(y_true)}")
    # logger.info(f"Unique classes in predicted labels for {type}: {set(y_pred)}")

    # Check if predictions are all one class
    if len(set(y_pred)) == 1:
        logger.warning(f"All predictions are of class {y_pred[0]}.")

    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "Target": target,
        "Accuracy": accuracy
    }

    # Extract metrics for all classes
    for cls, cls_report in report.items():
        if isinstance(cls_report, dict):  # To ensure we're processing a class
            metrics[f"{cls}_Precision"] = cls_report['precision']
            metrics[f"{cls}_Recall"] = cls_report['recall']
            metrics[f"{cls}_F1-Score"] = cls_report['f1-score']

    return metrics


def train_model(model, X_train, y_train, save_model=False, model_name=None):
    """
    Train the model and optionally save it.
    """
    model.fit(X_train, y_train)

    if save_model and model_name:
        # Save the model using joblib
        model_path = get_path_from_root("results", "one_vs_all", model_name, "models", f"{model_name}.pkl")
        joblib.dump(model, model_path)

    return model


def perform_grid_search(model, X, y, param_grid, cv=None):
    """
    Perform grid search with cross-validation and return the best estimator.

    Args:
    - model: The estimator.
    - X: Feature data.
    - y: Target data.
    - param_grid: Grid of hyperparameters for search.
    - cv: Cross-validator. If None, default KFold with 5 splits is used.

    Returns:
    - The best estimator from the grid search.
    """

    logger.info("Starting Grid Search ...\n")

    if cv is None:
        cv = KFold(n_splits=5)  # Default cross-validator

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def cross_validate_model(model, X, y, cv=None):
    """
    Perform cross-validation and return the mean score.

    Args:
    - model: The estimator.
    - X: Feature data.
    - y: Target data.
    - cv: Cross-validator. If None, default KFold with 5 splits is used.

    Returns:
    - Mean cross-validation score.
    """
    if cv is None:
        cv = KFold(n_splits=5)  # Default cross-validator

    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    return np.mean(scores)
