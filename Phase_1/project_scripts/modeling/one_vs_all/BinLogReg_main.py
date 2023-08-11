import itertools
import logging
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

from Phase_1.config import FEATURES as allFeatures, TARGET_1
from Phase_1.project_scripts import get_path_from_root
# Importing preprocessing functions
from Phase_1.project_scripts.preprocessing.preprocessing import (balance_data, imputation_applier, imputation_pipeline,
                                                                 preprocess_ovr, scaling_applier, scaling_pipeline,
                                                                 split_data)
# Importing utility functions
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.model_utils import (add_interaction_terms, calculate_metrics,
                                                         ensure_directory_exists, save_results, train_model)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "logistic_regression"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "binary"


def main():
    logger.info("Starting one-vs-all logistic regression...")

    ensure_directory_exists(RESULTS_DIR)

    # Subdirectories for model and metrics
    model_dir = os.path.join(RESULTS_DIR, "models")
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")

    ensure_directory_exists(model_dir)
    ensure_directory_exists(metrics_dir)
    # Load data
    df = load_data()

    # Preprocess the data specific for OvR
    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    # List of features to consider for interactions
    feature_pairs = list(itertools.combinations(allFeatures, 2))

    results = []

    for key, (X, y) in datasets.items():
        logging.info(f"Starting model for {key} ...\n")

        for feature_pair in feature_pairs:
            # Splitting the data
            X_train, X_test, y_train, y_test = split_data(X, y)

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

            # Convert to DataFrame
            X_train_resampled_df = pd.DataFrame(X_train_resampled)

            # Assign column names
            X_train_resampled_df.columns = transformed_columns

            # Do the same for y, if y has more than one column.
            # If y is just a 1D array (single column), you can assign a single column name:
            y_train_resampled_df = pd.DataFrame(y_train_resampled)
            y_train_resampled_df.columns = ["AntisocialTrajectory"]

            # Defining parameter grid for grid search. To initiate grid search, comment out the second definition
            """
            param_grid = {
                'penalty': ['elasticnet'],
                'C': [0.5, 1, 2],
                'fit_intercept': [True, False],
                'l1_ratio': [np.arange(0.4, 0.8, 0.5)]
            }
            """
            param_grid = None

            # Training the model
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5)
            best_model = train_model(X_train_resampled, y_train_resampled, model, param_grid,
                                     "True", MODEL_NAME, model_dir)

            logger.info("Fitting the model...\n")

            # Predictions
            y_train_pred = best_model.predict(X_train_resampled)
            y_test_pred = best_model.predict(X_test_final)

            logger.info("Calculating Metrics...\n")

            # Calculating metrics
            train_metrics = calculate_metrics(y_train_resampled, y_train_pred, key, TARGET_1, "train")
            test_metrics = calculate_metrics(y_test, y_test_pred, key, TARGET_1, "test")

            # Saving the model
            # joblib.dump(grid_search, os.path.join(model_dir, f"binary_logistic_regression_{COMBINED}.pkl"))

            # Combining metrics
            results.append({
                "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            })
            logging.info("Printing results ...\n")
            print(results)

        save_results(TARGET_1, f"{key}_binary", results, metrics_dir)

        logger.info(f"Completed {key} classification.")

    logger.info("One-vs-all logistic regression completed.")


if __name__ == '__main__':
    main()
