import logging
import logging
import os
import warnings

from sklearn.linear_model import LogisticRegression

from Phase_1.config import TARGET_1
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import apply_preprocessing, preprocess_ovr
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, \
    ensure_directory_exists, save_results, train_model

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
    metrics_dir = os.path.join(RESULTS_DIR, "metrics\\without Race")

    ensure_directory_exists(model_dir)
    ensure_directory_exists(metrics_dir)
    # Load data
    df = load_data_old()

    # Preprocess the data specific for OvR
    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    # List of features to consider for interactions
    # feature_pairs = list(itertools.combinations(features, 2))

    features = ["Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth", "Is_Male"]
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    for key, (X, y) in datasets.items():
        results = []

        logging.info(f"Starting model for {key} ...\n")

        for feature_pair in feature_pairs:
            X_train_resampled, y_train_resampled, X_test_final, y_test = apply_preprocessing(X, y, feature_pair, key)

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

            # Append the results
            results.append({
                "interaction": f"{feature_pair[0]}_x_{feature_pair[1]}",
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            })

        logging.info("Saving results ...\n")

        save_results(TARGET_1, f"{key}", results, metrics_dir)

        logger.info(f"Completed {key} classification.")

    logger.info("One-vs-all logistic regression completed.")


if __name__ == '__main__':
    main()
