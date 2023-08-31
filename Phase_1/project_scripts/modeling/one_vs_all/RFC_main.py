import logging
import os
import warnings

from sklearn.ensemble import RandomForestClassifier

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

MODEL_NAME = "random_forest"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
TYPE_OF_CLASSIFICATION = "binary"


def main(interaction, target):
    logger.info(f"Starting one-vs-all {MODEL_NAME}...")

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

        logging.info("Implementing Statistical Control...\n")
        X = add_squared_terms(X)

        param_grid = None  # Not performing grid search

        # Training the model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )

        if interaction:

            temp = features.copy()
            temp.remove("PolygenicScoreEXT_x_Is_Male")
            temp.remove("PolygenicScoreEXT_x_Age")
            temp.remove("Age")
            temp.remove("Is_Male")

            fixed_element = "PolygenicScoreEXT"
            feature_pairs = [(fixed_element, x) for x in temp if x != fixed_element]

            for feature_pair in feature_pairs:
                X_train, y_train, X_val, y_val, X_test, y_test = apply_preprocessing_with_interaction_terms(
                    X, y, feature_pair, features)

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
            X_train, y_train, X_val, y_val, X_test, y_test = apply_preprocessing_without_interaction_terms(
                X, y, features)

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

        save_results(target, f"{key}", results, metrics_dir, interaction)

    logger.info(f"One-vs-all {MODEL_NAME} completed.")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(interaction=True, target=target_1)
