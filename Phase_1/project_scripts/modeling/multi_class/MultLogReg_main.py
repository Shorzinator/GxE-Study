import itertools
import logging
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Using your utility functions and other functions you've already created
from Phase_1.project_scripts.preprocessing import *
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.model_utils import add_interaction_terms, calculate_metrics
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, X_test, y_train, y_test, metric_dir, model_dir):
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

    logger.info("Training the multinomial logistic regression model...\n")

    # Define the model
    mlr_model = LogisticRegression(multi_class='multinomial', max_iter=10000, solver='saga', n_jobs=-1, tol=1e-5,
                                   class_weight="balanced", l1_ratio=0.4, penalty="elasticnet", C=0.5, fit_intercept=True)

    logger.info("Fitting the model...\n")

    # GridSearchCV
    """
    # Define the parameter grid
    param_grid = {
        'penalty': ['elasticnet'],
        'C': [0.5, 1, 2],
        'fit_intercept': [True, False],
        'l1_ratio': [0.4, 0.5, 0.6, 0.55, 0.45]
    }
    
    grid_search = GridSearchCV(estimator=mlr_model, param_grid=param_grid,
                               scoring='accuracy', cv=5,
                               verbose=0, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = grid_search.predict(X_train)
    y_pred_test = grid_search.predict(X_test)
    
        best_parameters = grid_search.best_params_
    results_path = os.path.join(results_dir, f"best_parameters_{COMBINED}.json")
    
    # Check if the file exists
    if os.path.exists(results_path):
        # Read the current content of the JSON file
        with open(results_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Append new results to the data
    data['run_{}'.format(len(data) + 1)] = best_parameters

    # Write the updated data back to the JSON file
    with open(results_path, 'w') as f:
        json.dump(data, f, indent=4)

    best_estimator = grid_search.best_estimator_
    joblib.dump(best_estimator, os.path.join(model_dir, f"best_estimator_{COMBINED}.pkl"))
    """

    # Train the model
    mlr_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = mlr_model.predict(X_train)
    y_pred_test = mlr_model.predict(X_test)

    # Saving the model
    # joblib.dump(grid_search, os.path.join(model_dir, f"multinomial_logistic_regression_{COMBINED}.pkl"))

    logger.info("Calculating Metrics...\n")

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train, "logistic_regression", "AST", "train")
    test_metrics = calculate_metrics(y_test, y_pred_test, "logistic_regression", "AST", "test")

    return train_metrics, test_metrics,


if __name__ == "__main__":
    logging.info(f"Starting the program ...")

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

    # Load data
    df = load_data()

    # Preprocess data
    df, outcome, features = preprocess_multinomial(df, "AntisocialTrajectory")

    # List of features to consider for interactions
    feature_pairs = list(itertools.combinations(features, 2))

    results = []  # To store results of each iteration
    train_metrics, test_metrics = {}, {}

    for feature_pair in feature_pairs:
        df_temp = df.copy()

        df_temp = add_interaction_terms(df_temp, feature_pair)

        # Split, train using df_temp, and get metrics
        X_train, X_test, y_train, y_test = split_data(df_temp, outcome)

        # Applying imputation
        impute = imputation_pipeline()
        X_train_imputed, X_test_imputed = imputation_applier(impute, X_train, X_test)

        # Applying scaling
        scaler = scaling_pipeline(X_train_imputed.columns.tolist())
        X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(scaler, X_train_imputed, X_test_imputed)

        # Balancing data
        # logger.info(f"Distribution before balancing:\n{y_train.value_counts(normalize=True)}\n")
        X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train)
        # logger.info(f"Distribution after balancing:\n{y_train_resampled.value_counts(normalize=True)}\n")

        # Training model and calculating performance
        train_metrics, test_metrics = train_model(X_train_resampled, X_test_imputed_scaled, y_train_resampled, y_test, metrics_dir, model_dir)

        # Append the results
        results.append({
            "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        })

        print(results)

    flattened_data = []

    for entry in results:
        interaction = entry['interaction']
        for key in ['train_metrics', 'test_metrics']:
            metrics = entry[key]
            flat_metrics = {'interaction': interaction, 'type': key}
            for metric, value in metrics.items():
                flat_metrics[metric] = value
            flattened_data.append(flat_metrics)

    df = pd.DataFrame(flattened_data)

    print(df.to_csv(
        "C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\Phase_1\\results\\multi_class\\logistic_regression_results\\metrics\\results.csv"))

    # Save results
    # save_results(model_name, "AST", "Multinomial", "multi-class", {"train": train_metrics, "test": test_metrics},
    #           metrics_dir)

    # logger.info("Multinomial Logistic Regression Completed.")
