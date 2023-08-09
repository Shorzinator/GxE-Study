import logging
import os

import joblib
from sklearn.linear_model import LogisticRegression

# Using your utility functions and other functions you've already created
from Phase_1.project_scripts.preprocessing import imputation_pipeline, preprocess_multinomial, scaling_pipeline, \
    split_data
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, save_results
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, X_test, y_train, y_test, preprocessor, model_dir):
    """
    Train the multinomial logistic regression model and return metrics.

    Args:
    - X_train, X_test: feature data for training and testing
    - y_train, y_test: target data for training and testing
    - preprocessor: preprocessing pipeline (imputation, scaling, etc.)
    - model_dir: directory where trained model will be saved

    Returns:
    - train_metrics, test_metrics: Metrics for train and test predictions
    """

    logger.info("Training the multinomial logistic regression model...")

    # Define the model
    mlr_model = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='saga', n_jobs=-1)

    # Train the model
    mlr_model.fit(X_train, y_train)

    # Saving the model
    joblib.dump(mlr_model, os.path.join(model_dir, "multinomial_logistic_regression_model.pkl"))

    # Make predictions
    y_pred_train = mlr_model.predict(X_train)
    y_pred_test = mlr_model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train, "logistic_regression", "Multinomial", "Train")
    test_metrics = calculate_metrics(y_test, y_pred_test, "logistic_regression", "Multinomial", "Test")

    return train_metrics, test_metrics


if __name__ == "__main__":
    # Load data
    df = load_data()

    # Preprocess data
    data, outcome = preprocess_multinomial(df, "AntisocialTrajectory")

    # Split data
    X_train, X_test, y_train, y_test = split_data(data, outcome)

    # Applying imputation
    impute = imputation_pipeline(X_train)
    X_train = impute.fit_transform(X_train)
    X_test = impute.transform(X_test)

    # Applying scaling
    scaler = scaling_pipeline(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Establish the model-specific directories
    model_name = "logistic_regression"
    results_dir = get_path_from_root("results", "multi_class", f"{model_name}_results")

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Subdirectories for model and metrics
    model_dir = os.path.join(results_dir, "models")
    metrics_dir = os.path.join(results_dir, "metrics")

    # Ensure the subdirectories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Train the model and get metrics
    train_metrics, test_metrics = train_model(X_train, X_test, y_train, y_test, preprocessor, model_dir)

    # Save results
    save_results(model_name, "AST", "Multinomial", {"train": train_metrics, "test": test_metrics})

    logger.info("Multinomial Logistic Regression Completed.")
