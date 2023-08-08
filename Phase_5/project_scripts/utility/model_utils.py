import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from Phase_5.project_scripts.utility.path_utils import get_path_from_root


def save_results(model_name, target, comparison_class, results):
    """
    Save the results in a structured directory and file.
    :param model_name: Name of the model (e.g., "xgboost")
    :param target: Target variable (either "AST" or "SUT")
    :param comparison_class: The class being compared against (e.g., "1_vs_4")
    :param results: The results data (a dictionary)
    """

    # Define the directory structure
    dir_path = get_path_from_root("results", "one_vs_all", f"{model_name}_results")

    # Check and create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Convert results dictionary to dataframe and save as CSV
    results_df = pd.DataFrame([results])
    results_file = os.path.join(dir_path, f"{target}_{comparison_class}_results.csv")
    results_df.to_csv(results_file, index=False)


def calculate_metrics(y_actual, y_pred, model_name, target, comparison_class):
    """
    Evaluate the given model on the test set and save the results.
    :param y_actual: True values for both train and test (provided one after other)
    :param y_pred: Predicted values for both train and test (provided one after other)
    :param model_name: Name of the model (e.g., "xgboost")
    :param target: Target variable (either "AST" or "SUT")
    :param comparison_class: The class being compared against (e.g., "1_vs_4")
    :return: None
    """

    # Calculate metrics
    accuracy = accuracy_score(y_actual, y_pred)
    class_report = classification_report(y_actual, y_pred, output_dict=True)
    precision = precision_score(y_actual, y_pred, average="weighted")

    # Store results
    results = {
        "Model": model_name,
        "Target": target,
        "Comparison": comparison_class,
        "Accuracy": accuracy,
        "Precision": precision,
    }

    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                results[f"{label}_{metric_name}"] = metric_value

    # Save the results
    save_results(model_name, target, comparison_class, results)


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



def perform_grid_search(model, X, y, param_grid):
    """
    Perform grid search with cross-validation and return the best estimator.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return the mean score.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    return np.mean(scores)
