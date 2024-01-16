import itertools
import json
import logging
import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import COMBINED, FEATURES as allFeatures, IT, TARGET_1
# Using your utility functions and other functions you've already created
from Phase_1.project_scripts.preprocessing import *
from utility.data_loader import load_data
from utility.model_utils import add_interaction_terms, \
    calculate_metrics, ensure_directory_exists, save_results, train_model
from utility.path_utils import get_path_from_root

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "random_forest"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "multinomial"


def main():
    logging.info(f"Starting {MODEL_NAME} regression ...")

    ensure_directory_exists(RESULTS_DIR)

    # Subdirectories for model and metrics
    model_dir = os.path.join(RESULTS_DIR, "models")
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")

    ensure_directory_exists(model_dir)
    ensure_directory_exists(metrics_dir)

    # Load data
    df = load_data()

    # Preprocess data
    logger.info("Starting data preprocessing for random forest classification ...\n")
    df, outcome = preprocess_multinomial(df, "AntisocialTrajectory")

    # List of features to consider for interactions
    feature_pairs = list(itertools.combinations(allFeatures, 2))

    results = []  # To store results of each iteration

    for feature_pair in feature_pairs:
        df_temp = df.copy()

        # Split, train using df_temp, and get metrics
        X_train, X_test, y_train, y_test = split_data(df_temp, outcome)

        # Applying imputation and one-hot encoding on training data
        impute = imputation_pipeline()
        X_train_imputed = imputation_applier(impute, X_train)

        # Generate interaction terms using the transformed column names for training data
        X_train_final = add_interaction_terms(X_train_imputed, feature_pair)

        # Capture transformed column names after preprocessing the training data
        transformed_columns = X_train_final.columns.tolist()

        # Applying imputation and one-hot encoding on testing data
        X_test_imputed = imputation_applier(impute, X_test)

        # Generate interaction terms using the transformed column names for testing data
        X_test_final = add_interaction_terms(X_test_imputed, feature_pair)
        X_test_final = pd.DataFrame(X_test_final)

        # Applying scaling
        scaler = scaling_pipeline(transformed_columns)
        X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(scaler, X_train_final, X_test_final)

        # Balancing data
        # logger.info(f"Distribution before balancing:\n{y_train.value_counts(normalize=True)}\n")
        X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train)
        # logger.info(f"Distribution after balancing:\n{y_train_resampled.value_counts(normalize=True)}\n")

        X_train_resampled = pd.DataFrame(X_train_resampled)

        # Defining parameter grid for grid search. To initiate grid search, comment out the second definition
        """
        param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
        """
        param_grid = None

        # Define model
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Training model
        logger.info(f"Training the {MODEL_NAME} model...\n")
        best_model = train_model(X_train_resampled, y_train_resampled, model, param_grid,
                                 "False", MODEL_NAME, model_dir, IT)

        logger.info("Predicting with the model...\n")

        # Predictions
        y_train_pred = best_model.predict(X_train_resampled)
        y_test_pred = best_model.predict(X_test_final)

        logger.info("Calculating Metrics...\n")

        # Calculate metrics
        train_metrics = calculate_metrics(y_train_resampled, y_train_pred, MODEL_NAME, TARGET_1, "train")
        test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "test")

        # If the param_grid is not commented out, grid search would run and hence this would run as well
        if param_grid:

            # Saving the best parameters
            best_parameters = best_model.best_params_
            results_path = os.path.join(model_dir, f"best_parameters_{COMBINED}_{IT}.json")

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

            # Saving the best estimator
            best_model.dump(best_model, os.path.join(model_dir, f"best_estimator_{COMBINED}_{IT}.pkl"))

        # Append the results
        results.append({
            "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        })

    logging.info(f"Saving results ...")
    save_results(TARGET_1, TYPE_OF_CLASSIFICATION, results, metrics_dir)
    logger.info("One-vs-all random forest classification completed.")


if __name__ == "__main__":
    main()
