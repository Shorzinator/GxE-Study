import os
import pickle

import matplotlib.pyplot as plt
import shap
from PIL import Image
from sklearn.linear_model import LogisticRegression

from Phase_1.project_scripts import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import *
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "logistic_regression_shapley"

RESULTS_DIR = get_path_from_root("results", "evaluation", "shapley_analysis")
VISUALIZATION_DIR = get_path_from_root(RESULTS_DIR, "visualizations")
ensure_directory_exists(RESULTS_DIR)
ensure_directory_exists(VISUALIZATION_DIR)

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
    # plt.savefig(save_path)
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


def main(target, interaction):
    logger.info("Starting Shapley value analysis for one-vs-all logistic regression...")

    # Loading the data
    df = load_data_old()

    # Applying preprocessing
    if target == "AntisocialTrajectory":
        features = FEATURES_FOR_AST
        datasets = preprocess_ast_ovr(df, features)
    else:
        features = FEATURES_FOR_SUT
        datasets = preprocess_sut_ovr(df, features)

    interactions_cache = {}
    results = []
    for key, (X, y) in datasets.items():
        if interaction:

            temp = features.copy()
            temp.remove("PolygenicScoreEXT")
            fixed_element = "PolygenicScoreEXT"
            feature_pairs = [(fixed_element, x) for x in temp if x != fixed_element]

            for feature_pair in feature_pairs:

                # Applying additional preprocessing
                X_train, y_train, X_test, y_test = apply_preprocessing_with_interaction_terms(X, y, feature_pair, key,
                                                                                              features)

                model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                           l1_ratio=0.5)

                best_model = train_model(X_train, y_train, model, None)

                shap_values = compute_shap_values(best_model, X_train)
                interactions_cache.setdefault(feature_pair, []).append((key, X_train, shap_values))
                #visualize_shap_values(shap_values, X_train, key, X_train.columns.tolist(), interaction_term=feature_pair)

                # Save the SHAP values to a DataFrame and append to the results list
                shap_df = pd.DataFrame(shap_values, columns=X_train.columns)
                shap_df['config'] = key
                shap_df['interaction'] = ' x '.join(feature_pair)
                results.append(shap_df)

        else:
            # Applying additional preprocessing
            X_train, y_train, X_test, y_test = ap_without_it(X, y, features)

            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5)

            # Train the model
            best_model = train_model(X_train, y_train, model, None)

            # Compute and visualize SHAP values
            shap_values = compute_shap_values(best_model, X_train)
            #visualize_shap_values(shap_values, X_train, key, X_train.columns.tolist())

            # Save the SHAP values to a DataFrame and append to the results list
            shap_df = pd.DataFrame(shap_values, columns=X_train.columns)
            shap_df['config'] = key
            shap_df['interaction'] = 'None'
            results.append(shap_df)

        logger.info(f"Completed Shapley analysis for {key}.")

    #for interaction_term, data in interactions_cache.items():
    #    create_comparison_image(data, interaction_term)

    # Concatenate all the results DataFrames and save to CSV
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(get_path_from_root(RESULTS_DIR, 'shapley_results.csv'), index=False)

    logger.info("Shapley value analysis completed.")


if __name__ == '__main__':
    main(target="AntisocialTrajectory", interaction="True")
