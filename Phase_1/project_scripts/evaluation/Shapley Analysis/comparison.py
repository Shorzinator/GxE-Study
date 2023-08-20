import os
import pickle

import matplotlib.pyplot as plt
import shap

from Phase_1.project_scripts import get_path_from_root

# Paths and constants
RESULTS_DIR = get_path_from_root("results", "evaluation", "shapley_analysis")
VISUALIZATION_DIR = get_path_from_root(RESULTS_DIR, "visualizations")


def load_saved_shap_values(config_name, interaction_term):
    """
    Load saved shap values based on the configuration and interaction term.

    :param config_name: Name of the configuration, e.g., "1v4", "2v4", etc.
    :param interaction_term: Tuple of the interaction term, e.g., ("FeatureA", "FeatureB")
    :return: Loaded shap values
    """
    term_str = "_".join(interaction_term)
    filename = f"shap_values_{config_name}_{term_str}.pkl"
    with open(os.path.join(RESULTS_DIR, filename), 'rb') as file:
        shap_values = pickle.load(file)
    return shap_values


def compare_shap_plots(configs, interaction_term):
    """
    Display SHAP summary plots side by side for direct comparison.

    :param configs: A list of configuration names, e.g., ["1v4", "2v4", "3v4"]
    :param interaction_term: Tuple of the interaction term, e.g., ("FeatureA", "FeatureB")
    """
    num_configs = len(configs)
    plt.figure(figsize=(15, 5 * num_configs))

    for i, config_name in enumerate(configs):
        shap_vals = load_saved_shap_values(config_name, interaction_term)

        # Assuming you have your X_test stored similarly:
        X_test = load_test_data_for_config(config_name)

        plt.subplot(num_configs, 1, i + 1)

        shap.summary_plot(shap_vals, X_test, show=False, plot_size=None)
        plt.title(f"SHAP Summary Plot for {config_name} - Interaction: {' x '.join(interaction_term)}")

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"comparison_plot_{'_'.join(interaction_term)}.png"))
    plt.show()


if __name__ == '__main__':
    interaction_term = ("PolygenicScoreEXT", "FeatureX")  # replace with your interaction term
    configs = ["1v4", "2v4", "3v4"]
    compare_shap_plots(configs, interaction_term)
