import logging
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import *
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import ap_without_it, \
    ap_with_it, preprocess_ast_ovr, preprocess_sut_ovr
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import add_squared_terms, calculate_metrics, \
    ensure_directory_exists, save_results, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "logistic_regression"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "binary"


def save_processed_data(X, y, target, case, interaction, directory, feature_pair):
    """
    Save the processed datasets for each specific case before training.

    Args:
    - X: Processed features.
    - y: Processed target.
    - target: Either "AntisocialTrajectory" or "SubstanceUseTrajectory".
    - case: Specific case like "1_vs_3".
    - interaction: Boolean indicating if interaction terms were used.
    - directory: The directory where data should be saved.
    """
    # Creating a filename based on the target, case, and interaction.
    if target == "SubstanceUseTrajectory":
        target = "SUT"
    else:
        target = "AST"

    if interaction:
        filename = f"{target}_{case}_{feature_pair[0]}_x_{feature_pair[1]}_IT.csv"
    else:
        filename = f"{target}_{case}_{feature_pair[0]}_x_{feature_pair[1]}_noIT.csv"

    # Merging the features and target into a single dataframe.
    df_to_save = pd.concat([X, y], axis=1)

    # Saving the dataframe to the specified directory.
    save_path = os.path.join(directory, filename)
    df_to_save.to_csv(save_path, index=False)
    logger.info(f"Saved data for {target} {case}...\n")


def main(interaction, target):
    logger.info(f"Starting one-vs-all {MODEL_NAME}...")

    # Subdirectories for metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    processed_data_dir = os.path.join(RESULTS_DIR, "processed_data")
    ensure_directory_exists(metrics_dir)
    ensure_directory_exists(processed_data_dir)

    # Load data
    df = load_data_old()

    # Preprocess the data specific for OvR
    if target == "AntisocialTrajectory":
        datasets, features = preprocess_ast_ovr(df, FEATURES_FOR_AST)
    else:
        datasets, features = preprocess_sut_ovr(df, FEATURES_FOR_SUT)

    for key, (X, y) in datasets.items():
        results = []

        logging.info(f"Starting model for {key} ...\n")

        logging.info("Implementing Statistical Control...\n")
        X = add_squared_terms(X)

        param_grid = None  # Not performing grid search

        # Defining the model
        model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, class_weight='balanced')

        if interaction:

            # Creating pairs for interaction terms
            temp = features.copy()
            temp.remove("PolygenicScoreEXT_x_Is_Male")
            temp.remove("PolygenicScoreEXT_x_Age")
            temp.remove("Age")
            temp.remove("Is_Male")

            if target == "AntisocialTrajectory":
                temp.remove("SubstanceUseTrajectory")
            else:
                temp.remove("AntisocialTrajectory")

            fixed_element = "PolygenicScoreEXT"
            feature_pairs = [(fixed_element, x) for x in temp if x != fixed_element]

            for feature_pair in feature_pairs:
                X_train, y_train, X_val, y_val, X_test, y_test = ap_with_it(X, y, feature_pair, features)

                # if key == "2_vs_3":
                # Before training the model, save the processed datasets.
                # save_processed_data(X_train, y_train, target, key, interaction, processed_data_dir, feature_pair)

                # Training the model
                best_model = train_model(X_train, y_train, model, param_grid)

                # Validate the model
                y_val_pred = best_model.predict(X_val)
                val_metrics = calculate_metrics(y_val, y_val_pred, MODEL_NAME, target, "validation")

                # Test the model
                y_test_pred = best_model.predict(X_test)
                test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, target, "test")

                results.append({
                    "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics
                })
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = ap_without_it(
                X, y, features)

            # Before training the model, save the processed datasets.
            # save_processed_data(X_train, y_train, target, key, interaction, processed_data_dir)

            # Training the model
            best_model = train_model(X_train, y_train, model, param_grid)

            # Validate the model
            y_val_pred = best_model.predict(X_val)
            val_metrics = calculate_metrics(y_val, y_val_pred, MODEL_NAME, target, "validation")

            # Test the model
            y_test_pred = best_model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, target, "test")

            results.append({
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics
            })

        logger.info(f"Completed {key} classification.\n")

        save_results(target, f"{key}", results, metrics_dir, interaction, MODEL_NAME)

    logger.info(f"One-vs-all {MODEL_NAME} completed.")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(interaction=True, target=target_2)
