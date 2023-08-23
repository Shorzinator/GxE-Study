import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from Phase_1.config import TARGET_1
from Phase_1.project_scripts.preprocessing.preprocessing import apply_preprocessing_with_interaction_terms, \
    apply_preprocessing_without_interaction_terms, preprocess_ovr
from Phase_1.project_scripts.utility.data_loader import *
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, train_model
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

MODEL_NAME = "logistic_regression"


def run_model_with_interaction_terms():
    # Load data
    df = load_data_old()
    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    features = ["Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth", "Is_Male",
                "SubstanceUseTrajectory"]
    fixed_element = "PolygenicScoreEXT"
    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    results = {}

    for key, (X, y) in datasets.items():
        for feature_pair in feature_pairs:

            # Apply preprocessing with the specific interaction term
            logger.info("Applying supplementary steps of preprocessing ...\n")
            X_train, y_train, X_test, y_test = apply_preprocessing_with_interaction_terms(X, y, feature_pair, key)
            logger.info("Supplementary steps completed ...\n")

            logger.info("Training the model ...\n")
            # Train the model
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                       l1_ratio=0.5)
            best_model = train_model(X_train, y_train, model)
            logger.info("Training complete ...\n")

            # Evaluate the model
            logger.info("Predicting using the trained model ..\n")
            y_test_pred = best_model.predict(X_test.values)
            logger.info("Prediction complete ..\n")

            test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "text")

            # Store the metrics for this interaction term
            if feature_pair not in results:
                results[feature_pair] = {}
            results[feature_pair][key] = test_metrics

    return results, datasets, feature_pairs


def run_model_without_interaction_term():
    # Load data
    df = load_data_old()

    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    results = {}

    for key, (X, y) in datasets.items():

        # Further preprocessing
        logger.info("Applying supplementary steps of preprocessing ...\n")
        X_train, y_train, X_test, y_test = apply_preprocessing_without_interaction_terms(X, y, key)
        logger.info("Supplementary steps completed ...\n")

        # Train the model
        logger.info("Training the model ...\n")
        model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)
        best_model = train_model(X_train, y_train, model)
        logger.info("Training complete ...\n")

        # Evaluate the model
        logger.info("Predicting using the trained model ..\n")
        y_test_pred = best_model.predict(X_test.values)
        logger.info("Prediction complete ..\n")

        test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "test")

        # Store the metrics for this interaction term
        results[key] = test_metrics

    return results


def save_results_to_csv(metrics_without_interaction, metrics_with_interaction):
    # Create separate dataframes for metrics without and with interaction
    df_without = pd.DataFrame.from_dict(metrics_without_interaction, orient='index')
    df_with = {}

    for key, value in metrics_with_interaction.items():
        df_with[key] = pd.DataFrame.from_dict(value, orient='index')

    # Save dataframe to CSV
    path = get_path_from_root("results", "evaluation", "interaction_term_evaluation")
    ensure_directory_exists(path)

    # Save the metrics without interaction
    df_without.to_csv(os.path.join(path, "metrics_without_interaction.csv"))

    # Save the metrics with interaction for each feature pair
    for feature_pair, df in df_with.items():
        interaction_name = "_".join(feature_pair)
        df.to_csv(os.path.join(path, f"metrics_with_interaction_{interaction_name}.csv"))


def extract_metrics(data, key, dataset, metric):
    try:
        return float(data[key][dataset][metric])
    except:
        print(f"Error extracting {metric} for key {key} and dataset {dataset}. Value: {data[key][dataset].get(metric, 'N/A')}")
        return 0


def aggregate_differences(metrics_without_interaction, metrics_with_interaction):
    """
    Compute the differences in metrics for each interaction term.
    Return the differences.
    """
    differences = {}

    # For each feature interaction
    for feature_pair, datasets_metrics in metrics_with_interaction.items():

        # For each dataset category
        for dataset, metrics in datasets_metrics.items():

            # For simplicity, let's just focus on accuracy
            if 'Accuracy' not in metrics or 'Accuracy' not in metrics_without_interaction[dataset]:
                print(f"Accuracy missing for dataset '{dataset}'!")
                continue

            without_value = metrics_without_interaction[dataset]['Accuracy']
            with_value = metrics['Accuracy']

            # Create a composite key for clarity in plotting
            key = f"{feature_pair[0]} x {feature_pair[1]} ({dataset})"
            differences[key] = with_value - without_value

    return differences

"""
def plot_comparison(metrics_without_interaction, metrics_with_interaction_avg):
    labels = list(metrics_without_interaction.keys())
    metrics_of_interest = ['Accuracy', 'Custom_Metric', 'Matthews Correlation Coefficient',
                           'macro avg_F1-Score', 'weighted avg_F1-Score']

    barWidth = 0.3
    r1 = np.arange(len(metrics_of_interest))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(15, 8))

    for idx, label in enumerate(labels):
        plt.subplot(1, len(labels), idx + 1)

        # Ensure the metrics are available
        bars1 = [metrics_without_interaction[label].get(metric, 0) for metric in metrics_of_interest]
        bars2 = [metrics_with_interaction_avg[label].get(metric, 0) for metric in metrics_of_interest]

        plt.bar(r1, bars1, color='b', width=barWidth, edgecolor='grey', label='No Interaction')
        plt.bar(r2, bars2, color='r', width=barWidth, edgecolor='grey', label='With Interaction')

        plt.title(f'Metrics for {label}')
        plt.xticks([r + barWidth for r in range(len(bars1))], metrics_of_interest, rotation=45)
        plt.legend()

    plt.tight_layout()
    plt.show()
"""


def plot_comparison(metrics_without_interaction, metrics_with_interaction_avg):
    labels = list(metrics_without_interaction.keys())
    metrics_of_interest = ['Accuracy', 'Custom_Metric', 'Matthews Correlation Coefficient',
                           'macro avg_F1-Score', 'weighted avg_F1-Score']

    barWidth = 0.5
    r1 = np.arange(len(metrics_of_interest))

    plt.figure(figsize=(15, 8))

    for idx, label in enumerate(labels):
        plt.subplot(1, len(labels), idx + 1)

        # Calculate the differences for the metrics of interest
        differences = [
            metrics_with_interaction_avg[label].get(metric, 0) - metrics_without_interaction[label].get(metric, 0)
            for metric in metrics_of_interest]

        bars = plt.bar(r1, differences, color='g', width=barWidth, edgecolor='grey')

        plt.title(f'Metric Differences for {label}')
        plt.xticks(r1, metrics_of_interest, rotation=45)

        # Annotate each bar with its value
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def average_metrics_across_interactions(metrics_with_interaction):
    """
    Given the metrics with interactions, average the metrics for each configuration (1v4, 2v4, 3v4)
    to get a single metric value for each configuration from all the interaction models.
    """
    # Create a dictionary to store summed metrics for each configuration
    summed_metrics = {
        '1_vs_4': defaultdict(float),
        '2_vs_4': defaultdict(float),
        '3_vs_4': defaultdict(float)
    }

    # Count the number of interaction terms for each configuration
    interaction_count = {
        '1_vs_4': 0,
        '2_vs_4': 0,
        '3_vs_4': 0
    }

    # List of valid configurations
    valid_configs = ['1_vs_4', '2_vs_4', '3_vs_4']

    # Sum up the metrics for each configuration
    for key, metrics_dict in metrics_with_interaction.items():
        print(f"Processing key: {key}")  # Diagnostic print
        for config in valid_configs:
            if config in metrics_dict:
                interaction_count[config] += 1
                for metric, metric_value in metrics_dict[config].items():
                    # Check if the metric value can be converted to a float, if not, skip it
                    try:
                        float_value = float(metric_value)
                        summed_metrics[config][metric] += float_value
                    except ValueError:
                        print(f"Skipping metric: {metric}, value: {metric_value} due to conversion error.")

    # Average out the metrics for each configuration
    avg_metrics = {}
    for config, metrics in summed_metrics.items():
        avg_metrics[config] = {k: v / interaction_count[config] for k, v in metrics.items()}

    return avg_metrics


if __name__ == "__main__":
    metrics_with_interaction_test, datasets, feature_pairs = run_model_with_interaction_terms()
    metrics_without_interaction_test = run_model_without_interaction_term()

    sample_key = list(metrics_with_interaction_test.keys())[0]
    aggregated_differences = aggregate_differences(metrics_without_interaction_test, metrics_with_interaction_test)
    metrics_with_interaction_avg = average_metrics_across_interactions(metrics_with_interaction_test)

    save_results_to_csv(metrics_without_interaction_test, metrics_with_interaction_test)
    plot_comparison(metrics_without_interaction_test, metrics_with_interaction_avg)
