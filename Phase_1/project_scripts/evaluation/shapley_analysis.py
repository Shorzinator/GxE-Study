import os
import logging
import pickle

import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Phase_1.config import TARGET_1, FEATURES
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import preprocess_ovr, split_data, imputation_pipeline, \
    imputation_applier, scaling_pipeline, scaling_applier, balance_data
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import add_interaction_terms, train_model, save_results, \
    ensure_directory_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "logistic_regression_shapley"
RESULTS_DIR = get_path_from_root("results", "evaluation", "shapley_analysis")
TYPE_OF_CLASSIFICATION = "binary"


def compute_shap_values(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def save_shap_values(shap_values, key):
    save_path = get_path_from_root(RESULTS_DIR, f"shap_values_{key}.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(shap_values, file)


def main():
    logger.info("Starting Shapley value analysis for one-vs-all logistic regression...")

    ensure_directory_exists(RESULTS_DIR)

    # Load data
    df = load_data_old()
    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    features = FEATURES.copy()
    features.remove("PolygenicScoreEXT")
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    for key, (X, y) in datasets.items():
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

            # Training the model (you may want to reuse your train_model function)
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)
            best_model = train_model(X_train_resampled, y_train_resampled, model, None, MODEL_NAME)

            # Compute Shapley values
            shap_values = compute_shap_values(best_model, X_test_imputed_scaled)

            # Save Shapley values
            save_shap_values(shap_values, key)
        logger.info(f"Completed Shapley analysis for {key}.")

    logger.info("Shapley value analysis completed.")


if __name__ == '__main__':
    main()
