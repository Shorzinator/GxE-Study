import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, train_model
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = get_path_from_root("results", "evaluation")
ensure_directory_exists(RESULTS_DIR)
MODEL_NAME = "logistic_regression"


def visualize_results(df):
    """
    Visualizes the change in metrics and saves the plots with enhanced styling and annotations.
    """
    # Set style for the plots
    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    # Function to annotate the bars
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    # Bar plot for change in accuracy
    plt.figure(figsize=(18, 7))
    ax1 = sns.barplot(data=df, x="column_dropped", y="change_in_accuracy", hue="type")
    plt.title("Change in Accuracy by Column Dropped", fontsize=18)
    plt.xlabel("Columns Dropped", fontsize=15)
    plt.ylabel("Change in Accuracy", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title="Dataset Type", loc="upper right", fontsize=12)
    annotate_bars(ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_drop_column.png"))
    plt.close()

    # Bar plot for change in custom score
    plt.figure(figsize=(18, 7))
    ax2 = sns.barplot(data=df, x="column_dropped", y="change_in_custom_score", hue="type")
    plt.title("Change in Custom Score by Column Dropped", fontsize=18)
    plt.xlabel("Columns Dropped", fontsize=15)
    plt.ylabel("Change in Custom Score", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title="Dataset Type", loc="upper right", fontsize=12)
    annotate_bars(ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "custom_score_drop_column.png"))
    plt.close()


def evaluate_with_drop_column(interaction_term, output_column):
    """
    Evaluates model performance by iteratively dropping columns.
    """
    # Store results
    results = []
    model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)

    data_1_vs_4 = pd.read_csv(get_path_from_root("data", "processed", "resampled_data_1_vs_4.csv"))
    data_2_vs_4 = pd.read_csv(get_path_from_root("data", "processed", "resampled_data_2_vs_4.csv"))
    data_3_vs_4 = pd.read_csv(get_path_from_root("data", "processed", "resampled_data_3_vs_4.csv"))

    datasets = {
        "1v4": data_1_vs_4,
        "2v4": data_2_vs_4,
        "3v4": data_3_vs_4
    }

    for key, data in datasets.items():
        # Extracting features and target
        X = data.drop([interaction_term, output_column], axis=1)  # Dropping the interaction column
        y = data[output_column]

        # Storing the original model's performance as the baseline
        original_model = train_model(X, y, model)
        y_pred = original_model.predict(X)
        baseline_metrics = calculate_metrics(y, y_pred, MODEL_NAME, key, "train")

        for column in X.columns:  # Iterating over feature columns only
            try:
                logger.info(f"Training model without {column} column.\n")
                X_temp = X.drop(column, axis=1)

                # Train and get the model
                model = train_model(X_temp, y, model)

                # Get predictions and calculate metrics for the training set
                y_pred_temp = model.predict(X_temp)
                temp_metrics = calculate_metrics(y, y_pred_temp, MODEL_NAME, key, "train")

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
        results_df.to_csv(os.path.join(RESULTS_DIR, "column_drop_evaluation.csv"), index=False)

        logging.info("Visualizing results ...\n")
        # Visualize the results
        visualize_results(results_df)


if __name__ == "__main__":
    interaction_term_column = "PolygenicScoreEXT_x_Is_Male"
    output_column_name = "AntisocialTrajectory"
    evaluate_with_drop_column(interaction_term_column, output_column_name)
