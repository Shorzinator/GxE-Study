import matplotlib.pyplot as plt
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

    features = ["Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth", "Is_Male"]
    fixed_element = "PolygenicScoreEXT"
    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    results = {}

    for key, (X, y) in datasets.items():
        for feature_pair in feature_pairs:

            # Apply preprocessing with the specific interaction term
            X_train, X_test, y_train, y_test = apply_preprocessing_with_interaction_terms(X, y, feature_pair, key)

            # Train the model
            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)
            best_model = train_model(X_train, y_train, model)

            # Evaluate the model
            y_test_pred = best_model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "text")

            # Store the metrics for this interaction term
            results[feature_pair] = test_metrics

    return results


def run_model_without_interaction_term():
    # Load data
    df = load_data_old()

    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    results = {}

    for key, (X, y) in datasets.items():
        # Further preprocessing
        X_train, X_test, y_train, y_test = apply_preprocessing_without_interaction_terms(X, y, key)

        # Train the model
        model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)
        best_model = train_model(X_train, y_train, model)

        # Evaluate the model
        y_test_pred = best_model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, MODEL_NAME, TARGET_1, "test")

        # Store the metrics for this interaction term
        results[key] = test_metrics

    return results


def save_results_to_csv(metrics_without_interaction, metrics_with_interaction):
    # Convert the metrics to dataframes
    df_without = pd.DataFrame.from_dict(metrics_without_interaction, orient='index',
                                        columns=['Without Interaction Terms'])
    df_with = pd.DataFrame.from_dict(metrics_with_interaction, orient='index', columns=['With Interaction Terms'])

    # Merge dataframes
    df_combined = pd.concat([df_without, df_with], axis=1)

    # Save dataframe to CSV
    path = get_path_from_root("results", "evaluation", "interaction_term_evaluation_results.csv")
    ensure_directory_exists(path)
    df_combined.to_csv(path)


def plot_comparison(metrics_without_interaction, metrics_with_interaction):
    labels = list(metrics_without_interaction.keys())
    without_values = list(metrics_without_interaction.values())
    with_values = list(metrics_with_interaction.values())

    width = 0.35
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x, without_values, width, label='Without Interaction Terms')
    rects2 = ax.bar([i + width for i in x], with_values, width, label='With Interaction Terms')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of model performance with and without interaction terms')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(get_path_from_root("project_scripts", "evaluation", "interaction_term_comparison_plot.png"))
    plt.show()


if __name__ == "__main__":
    metrics_with_interaction_test = run_model_with_interaction_terms()
    metrics_without_interaction_test = run_model_without_interaction_term()

    save_results_to_csv(metrics_without_interaction_test, metrics_with_interaction_test)
    plot_comparison(metrics_without_interaction_test, metrics_with_interaction_test)
