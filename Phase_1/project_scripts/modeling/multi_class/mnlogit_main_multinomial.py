import json
import logging
import os
import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Using your utility functions and other functions you've already created
from Phase_1.project_scripts.preprocessing import balance_data, imputation_pipeline, preprocess_multinomial, \
    scaling_pipeline, split_data
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, save_results
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

warnings.filterwarnings("ignore", "l1_ratio parameter is only used when penalty is 'elasticnet'. Got (penalty=l1)")
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, X_test, y_train, y_test, model_dir):
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
    # User-entered variable names for naming
    cross_validator = "KF"
    grid_searcher = "GCV"
    balancer = "SMOTE"

    combined = f"{balancer}_{grid_searcher}_{cross_validator}"

    logger.info("Training the multinomial logistic regression model...\n")

    # Define the model
    mlr_model = LogisticRegression(multi_class='multinomial', max_iter=5000, solver='saga', n_jobs=-1, tol=1e-4,
                                   class_weight="balanced")

    # Define the parameter grid
    param_grid = {
        'solver': ['saga'],
        'penalty': ['elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.5, 0.75],
        'class_weight': [None, 'balanced']
    }

    logger.info("Fitting the model...\n")

    """
    # Expand the parameter grid into a list of parameter combinations
    from itertools import product
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_score = -float('inf')
    best_params = None
    best_model = None

    for params in tqdm(param_combinations):
        mlr_model.set_params(**params)
        mlr_model.fit(X_train, y_train)
        score = mlr_model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_params = params
            best_model = mlr_model

    logger.info(f"Best score: {best_score}")
    logger.info(f"Best parameters: {best_params}")
    """

    # Now, when you create your GridSearchCV object, you add the following:
    grid_search = GridSearchCV(estimator=mlr_model, param_grid=param_grid,
                               scoring='accuracy', cv=5,
                               verbose=0, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Train the model
    # mlr_model.fit(X_train, y_train)

    # Saving the model
    joblib.dump(grid_search, os.path.join(model_dir, f"multinomial_logistic_regression_model_{combined}.pkl"))

    # Make predictions
    y_pred_train = grid_search.predict(X_train)
    y_pred_test = grid_search.predict(X_test)

    logger.info("Calculating Metrics...\n")

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train, "logistic_regression", "Multinomial", "train")
    test_metrics = calculate_metrics(y_test, y_pred_test, "logistic_regression", "Multinomial", "test")

    best_parameters = grid_search.best_params_
    with open(os.path.join(results_dir, f"best_parameters_{combined}.json"), 'w') as f:
        json.dump(best_parameters, f)

    best_estimator = grid_search.best_estimator_
    joblib.dump(best_estimator, os.path.join(model_dir, f"best_estimator_{combined}.pkl"))

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
    initial_size_train = len(X_train)
    initial_size_test = len(X_test)
    X_train_imputed = impute.fit_transform(X_train)
    X_test_imputed = impute.transform(X_test)
    logger.info(f"Rows before scaling X_train: {initial_size_train}. Rows after: {len(X_train_imputed)}.")
    logger.info(f"Rows before scaling X_test: {initial_size_test}. Rows after: {len(X_test_imputed)}.\n")

    X_train_imputed = pd.DataFrame(X_train_imputed)
    X_test_imputed = pd.DataFrame(X_test_imputed)

    # Applying scaling
    scaler = scaling_pipeline(X_train_imputed)
    initial_size_train = len(X_train_imputed)
    initial_size_test = len(X_test_imputed)
    X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)
    logger.info(f"Rows before scaling X_train: {initial_size_train}. Rows after: {len(X_train_imputed_scaled)}.")
    logger.info(f"Rows before scaling X_test: {initial_size_test}. Rows after: {len(X_test_imputed_scaled)}.\n")

    X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled)
    X_test_imputed_scaled = pd.DataFrame(X_test_imputed_scaled)

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

    print("Distribution before balancing:")
    print(y_train.value_counts(normalize=True), "\n")

    X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train)

    print("Distribution after balancing:")
    print(y_train_resampled.value_counts(normalize=True), "\n")

    # Train the model and get metrics
    train_metrics, test_metrics = train_model(X_train_resampled, X_test_imputed_scaled, y_train_resampled, y_test,
                                              model_dir)

    # Save results
    save_results(model_name, "AST", "Multinomial", {"train": train_metrics, "test": test_metrics})

    logger.info("Multinomial Logistic Regression Completed.")
