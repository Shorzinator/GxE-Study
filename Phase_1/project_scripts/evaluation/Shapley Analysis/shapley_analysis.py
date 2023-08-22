import logging
import os
import pickle

import matplotlib.pyplot as plt
import shap
from PIL import Image
from sklearn.linear_model import LogisticRegression

from Phase_1.config import FEATURES
from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import balance_data, imputation_applier, imputation_pipeline, \
    preprocess_ovr, scaling_applier, scaling_pipeline, split_data
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import add_interaction_terms, ensure_directory_exists, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "logistic_regression_shapley"
RESULTS_DIR = get_path_from_root("results", "evaluation", "shapley_analysis")
VISUALIZATION_DIR = get_path_from_root(RESULTS_DIR, "visualizations")
TYPE_OF_CLASSIFICATION = "binary"


def compute_shap_values(model, X):
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    return shap_values


def save_to_disk(obj, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        logger.error(f"Failed to save to {path}. Error: {e}")


def load_from_disk(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logger.error(f"Failed to load from {path}. Error: {e}")


def visualize_shap_values(shap_values, X, config_name, feature_names, interaction_term=None):
    plt.figure(figsize=(40, 15))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    title = f"SHAP Summary Plot for {config_name}"
    if interaction_term:
        term_str = " x ".join(interaction_term)
        title += f" (Interaction: {term_str})"
    plt.title(title, fontsize=10)
    plt.tight_layout()
    filename = f"shap_summary_{config_name}"
    if interaction_term:
        term_str = "_".join(interaction_term)
        filename += f"_{term_str}"
    save_path = get_path_from_root(VISUALIZATION_DIR, f"{filename}.png")
    plt.savefig(save_path)
    plt.close()


def create_comparison_image(interactions_cache, interaction_term):
    images = []
    for config_name, _, _ in interactions_cache:
        term_str = "_".join(interaction_term)
        filename = f"shap_summary_{config_name}_{term_str}.png"
        image_path = get_path_from_root(VISUALIZATION_DIR, filename)
        images.append(Image.open(image_path))

    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    max_height = sum(heights)

    new_img = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    save_path = get_path_from_root(VISUALIZATION_DIR, f"comparison_plot_{term_str}.png")
    new_img.save(save_path)
    # new_img.show()


def compare_shap_plots(interactions_cache, interaction_term):
    num_configs = len(interactions_cache)
    fig, axes = plt.subplots(num_configs, 1, figsize=(15, 5 * num_configs))

    for i, (config_name, X_test, shap_vals) in enumerate(interactions_cache):
        ax = axes[i]
        shap.summary_plot(shap_vals, X_test, show=False, plot_size=None, feature_names=X_test.columns.tolist(), ax=ax)
        ax.set_title(f"{config_name} - Interaction: {' x '.join(interaction_term)}")

        # Hide the x-axis label for all but the last subplot
        if i != num_configs - 1:
            ax.set_xlabel("")

    # Hide the y-label for all subplots as we will set a common label later
    for ax in axes:
        ax.set_ylabel("")

    # Set common y-label and x-label for the entire figure
    fig.text(0.04, 0.5, 'Feature Name', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'SHAP Value', ha='center')

    term_str = "_".join(interaction_term)
    save_path = get_path_from_root(VISUALIZATION_DIR, f"comparison_plot_{term_str}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    logger.info("Starting Shapley value analysis for one-vs-all logistic regression...")
    ensure_directory_exists(RESULTS_DIR)
    ensure_directory_exists(VISUALIZATION_DIR)
    df = load_data_old()
    datasets = preprocess_ovr(df, "AntisocialTrajectory")
    features = FEATURES.copy()
    features.remove("PolygenicScoreEXT")
    fixed_element = "PolygenicScoreEXT"
    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]
    interactions_cache = {}
    for config_name, (X, y) in datasets.items():
        for feature_pair in feature_pairs:
            X_train, X_test, y_train, y_test = split_data(X, y)
            impute = imputation_pipeline()
            X_train_imputed = imputation_applier(impute, X_train)
            X_train_final = add_interaction_terms(X_train_imputed, feature_pair)
            transformed_columns = X_train_final.columns.tolist()
            X_test_imputed = imputation_applier(impute, X_test)
            X_test_final = add_interaction_terms(X_test_imputed, feature_pair)
            scaler = scaling_pipeline(transformed_columns)
            X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(scaler, X_train_final, X_test_final)
            X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train, config_name)
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5)
            best_model = train_model(X_train_resampled, y_train_resampled, model, None, MODEL_NAME)
            shap_values = compute_shap_values(best_model, X_test_imputed_scaled)
            interactions_cache.setdefault(feature_pair, []).append((config_name, X_test_imputed_scaled, shap_values))
            visualize_shap_values(shap_values, X_test_imputed_scaled, config_name, transformed_columns,
                                  interaction_term=feature_pair)
        logger.info(f"Completed Shapley analysis for {config_name}.")

    for interaction_term, data in interactions_cache.items():
        create_comparison_image(data, interaction_term)

    logger.info("Shapley value analysis completed.")


if __name__ == '__main__':
    main()
