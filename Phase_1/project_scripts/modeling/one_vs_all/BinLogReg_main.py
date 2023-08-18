import json
import logging
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

from Phase_1.config import TARGET_1
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import balance_data, imputation_applier, imputation_pipeline, \
    preprocess_ovr, scaling_applier, scaling_pipeline, split_data
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import add_interaction_terms, calculate_metrics, \
    ensure_directory_exists, train_model, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "logistic_regression"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "binary"


def main():
    logger.info("Starting one-vs-all logistic regression...")

    ensure_directory_exists(RESULTS_DIR)

    # Subdirectories for a model and metrics
    model_dir = os.path.join(RESULTS_DIR, "models")
    metrics_dir = os.path.join(RESULTS_DIR, "metrics\\without Race\\with SUT")

    ensure_directory_exists(model_dir)
    ensure_directory_exists(metrics_dir)

    # Load data
    df = load_data_old()

    # Preprocess the data specific for OvR
    datasets, features = preprocess_ovr(df, "AntisocialTrajectory")

    # List of features to consider for interactions
    # feature_pairs = list(itertools.combinations(features, 2))

    features = ["Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth", "Is_Male",
                "SubstanceUseTrajectory"]
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    for key, (X, y) in datasets.items():
        results = []

        logging.info(f"Starting model for {key} ...\n")

        for feature_pair in feature_pairs:
            # Split, train using df_temp, and get metrics
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
            X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled)

            # Balancing data
            # logger.info(f"Distribution before balancing:\n{y_train.value_counts(normalize=True)}\n")
            X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train, key)
            # logger.info(f"Distribution after balancing:\n{y_train_resampled.value_counts(normalize=True)}\n")

            X_train_resampled = pd.DataFrame(X_train_resampled)

            # Defining parameter grid for grid search. To initiate grid search, comment on the second definition
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

            best_model = train_model(X_train_resampled, y_train_resampled, model, param_grid, MODEL_NAME)

            # Predictions
            y_train_pred = best_model.predict(X_train_resampled)
            y_test_pred = best_model.predict(X_test_final)

            # Calculate metrics
            train_metrics = calculate_metrics(y_train_resampled, y_train_pred, MODEL_NAME, TARGET_1, "train")
            test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "test")

            # Saving the model
            # joblib.dump(grid_search, os.path.join(model_dir, f"multinomial_logistic_regression_{COMBINED}.pkl"))

            # If the param_grid is not commented out, grid search would run and hence this would run as well
            if param_grid:

                # Saving the best parameters
                best_parameters = best_model.best_params_
                results_path = os.path.join(model_dir, f"best_parameters.json")

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
                best_model.dump(best_model, os.path.join(model_dir, f"best_estimator.pkl"))

            # Append the results
            results.append({
                "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            })

        logging.info("Saving results ...\n")
        save_results(TARGET_1, f"{key}", results, metrics_dir)
        logger.info(f"Completed {key} classification.\n")

    logger.info("One-vs-all logistic regression completed.")


if __name__ == '__main__':
    main()
