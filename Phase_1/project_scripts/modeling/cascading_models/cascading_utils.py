import logging
import os
import traceback

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, \
    roc_auc_score

from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")

# Subdirectories for metrics
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")
ensure_directory_exists(METRICS_DIR)
ensure_directory_exists(GRAPH_DIR)


def compute_and_plot_shap_values(model, X_train, X_test, feature_name=None, outcome_names=None):
    logging.info("Starting SHAP analysis...\n")

    try:
        # Initialize the explainer
        logging.info("Initializing the explainer...\n")
        explainer = shap.KernelExplainer(model.predict, X_train.sample(100))

        # Compute SHAP values for the test set
        logging.info("Compute SHAP values for the test set...\n")
        shap_values = explainer.shap_values(X_test)

        # Collect average SHAP values for each outcome
        avg_shap_values = [np.mean(values) for values in shap_values]

        # Plotting the consolidated graph
        plt.figure(figsize=(10, 6))
        plt.bar(outcome_names, avg_shap_values, color='skyblue')
        plt.xlabel("Outcomes")
        plt.ylabel("Average SHAP Value")
        plt.title(f"Average SHAP Values for Feature: {feature_name}")
        for i, v in enumerate(avg_shap_values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()

        # Save the consolidated plot
        filename = os.path.join(GRAPH_DIR, f"consolidated_shap_values.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error computing or plotting SHAP values: {e}")
        print(traceback.format_exc())


def save_performance_metrics_csv(metrics, target):
    """
    Save the performance metrics to a CSV file

    :param metrics: Dict, performance metrics
    :param target: str, target variable name
    """
    filename = f"{target}_performance.csv"
    filepath = os.path.join(METRICS_DIR, filename)

    # Convert dictionary to DataFrame for easier CSV saving
    metrics_df = pd.DataFrame([metrics])

    # Save to CSV
    metrics_df.to_csv(filepath, index=False)

    logger.info(f"Performance metrics saved to {filepath}...\n")


def evaluate_model(predictions, y_true):
    """
    Evaluate model performance on given predictions and true labels.

    :param predictions: array-like, model predictions
    :param y_true: array-like, true labels
    :return: dict, containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average="micro")
    auc = roc_auc_score(y_true, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "auc": auc
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")

    return metrics


def evaluate_regression_model(predictions, y_true):
    """
    Evaluate regression model performance on given predictions and true labels.

    :param predictions: Array-like, model predictions
    :param y_true: Array-like, true labels
    :return: Dict, containing evaluation metrics
    """

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2
    }

    return metrics
