import logging
import os
import warnings

from sklearn.linear_model import LogisticRegression

from Phase_1.config import *
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import apply_preprocessing_with_interaction_terms, \
    apply_preprocessing_without_interaction_terms, preprocess_ast_ovr, preprocess_sut_ovr
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import add_squared_terms, calculate_metrics, \
    ensure_directory_exists, save_results, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "logistic_regression"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "binary"


def main(interaction, target, statistical_control):
    logger.info("Starting one-vs-all logistic regression...")

    # Subdirectories for a model and metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    ensure_directory_exists(metrics_dir)

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
        if statistical_control:
            logging.info("Implementing Statistical Control...\n")
            X = add_squared_terms(X)
            temp = features.copy()
            temp.remove("Age")
            temp.remove("Is_Male")
            temp.remove("PolygenicScoreEXT_x_Is_Male")
            temp.remove("PolygenicScoreEXT_x_Age")

        if interaction:

            fixed_element = "PolygenicScoreEXT"

            feature_pairs = [(fixed_element, x) for x in temp if x != fixed_element]

            for feature_pair in feature_pairs:
                X_train_resampled, y_train_resampled, X_test_final, y_test = apply_preprocessing_with_interaction_terms(
                    X, y, feature_pair, features)

                # Training the model
                model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                           l1_ratio=0.5)

                param_grid = None  # Not performing grid search

                best_model = train_model(X_train_resampled, y_train_resampled, model, param_grid)

                # Predictions
                y_train_pred = best_model.predict(X_train_resampled)
                y_test_pred = best_model.predict(X_test_final)

                # Calculate metrics
                train_metrics = calculate_metrics(y_train_resampled, y_train_pred, MODEL_NAME, target, "train")
                test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, target, "test")

                # Append the results
                results.append({
                    "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics
                })
        else:
            X_train_resampled, y_train_resampled, X_test_final, y_test = apply_preprocessing_without_interaction_terms(
                X, y, features)

            # Training the model
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5)

            param_grid = None  # Not performing grid search

            best_model = train_model(X_train_resampled, y_train_resampled, model, param_grid)

            # Predictions
            y_train_pred = best_model.predict(X_train_resampled)
            y_test_pred = best_model.predict(X_test_final)

            # Calculate metrics
            train_metrics = calculate_metrics(y_train_resampled, y_train_pred, MODEL_NAME, target, "train")
            test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, target, "test")

            # Append the results
            results.append({
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            })

        logger.info(f"Completed {key} classification.\n")

        save_results(target, f"{key}", results, metrics_dir, interaction, statistical_control)

    logger.info("One-vs-all logistic regression completed.")


if __name__ == '__main__':
    main(interaction=True, target="AntisocialTrajectory", statistical_control="True")
