import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from Phase_1.config import FEATURES
from Phase_1.project_scripts.preprocessing.preprocessing import apply_preprocessing_without_interaction_terms, \
    preprocess_ovr
from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, \
    ensure_directory_exists, train_model
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = get_path_from_root("results", "evaluation", "drop column analysis")
ensure_directory_exists(RESULTS_DIR)
MODEL_NAME = "logistic_regression"


def visualize_results(df, metric, key):
    """
    Visualizes the change in metrics and saves the plots with enhanced styling and annotations.
    """
    # Set style for the plots
    sns.set_style("whitegrid")

    # Function to annotate the bars
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=9)  # Adjust fontsize for annotations

    # Decide metric-specific details
    if metric == "Accuracy":
        y_col = "change_in_accuracy"
        ylabel = "Change in Accuracy"
        save_name = "accuracy_drop_column"
    else:  # Custom Score
        y_col = "change_in_custom_score"
        ylabel = "Change in Custom Score"
        save_name = "custom_score_drop_column"

    # Sort the dataframe by the metric for better visualization
    df = df.sort_values(by=y_col, ascending=False)

    # Decide color palette based on positive or negative change
    df["positive_change"] = df[y_col] > 0
    palette = {True: "g", False: "r"}  # Green for positive change and Red for negative

    # Bar plot for given metric
    plt.figure(figsize=(20, 8))
    ax1 = sns.barplot(data=df, x="column_dropped", y=y_col, palette=df["positive_change"].map(palette), dodge=False)
    plt.title(f"Change in {metric} by Column Dropped", fontsize=20)
    plt.xlabel("Columns Dropped", fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.xticks(rotation=45)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='g', label='Positive Change'),
                       Line2D([0], [0], color='r', label='Negative Change')]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=14)

    annotate_bars(ax1)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}_{key}.png"))
    plt.close()


def evaluate_with_drop_column(target):
    """
    Evaluates model performance by iteratively dropping columns.
    """
    logging.info("Evaluating model performance by iteratively dropping columns ...")
    ensure_directory_exists(RESULTS_DIR)

    df = load_data_old()
    datasets = preprocess_ovr(df, target)

    for key, (X, y) in datasets.items():
        # Store results
        results = []

        X_train, y_train, X_test, y_test = apply_preprocessing_without_interaction_terms(X, y, key)
        X_train = pd.DataFrame(X_train)

        model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)

        # Storing the original model's performance as the baseline
        original_model = train_model(X_train, y_train, model)
        y_pred = original_model.predict(X_test)
        baseline_metrics = calculate_metrics(y_test, y_pred, MODEL_NAME, key, "test")

        feature_names = FEATURES

        for column in feature_names:
            try:
                logger.info(f"Training model without {column} column.\n")
                X_temp = X_train.drop(column, axis=1)

                # Train and get the model
                model = train_model(X_temp, y_train, model)

                # Get predictions and calculate metrics for the test set
                y_pred_temp = model.predict(X_test.drop(column, axis=1))
                temp_metrics = calculate_metrics(y_test, y_pred_temp, MODEL_NAME, key, "test")

                change_in_accuracy = baseline_metrics["Accuracy"] - temp_metrics["Accuracy"]
                change_in_custom_score = baseline_metrics["Custom_Metric"] - temp_metrics["Custom_Metric"]

                results.append({
                    "type": key, "column_dropped": column, "change_in_accuracy": change_in_accuracy,
                    "change_in_custom_score": change_in_custom_score
                })

            except Exception as e:
                logger.error(f"Error encountered when processing column {column}. Error: {str(e)}")
                continue

        logging.info("Saving results ...\n")
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(RESULTS_DIR, f"column_drop_evaluation_{key}.csv"), index=False)

        logging.info("Visualizing results ...\n")
        # Visualize the results for both metrics
        for metric in ["Accuracy", "Custom Score"]:
            visualize_results(results_df, metric, key)


if __name__ == "__main__":
    evaluate_with_drop_column(target="AntisocialTrajectory")
