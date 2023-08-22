import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from Phase_1.config import FEATURES
from Phase_1.project_scripts.preprocessing.preprocessing import imputation_applier, imputation_pipeline, preprocess_ovr, \
    scaling_applier, scaling_pipeline, split_data
from Phase_1.project_scripts.utility.data_loader import load_data_old
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


def evaluate_with_drop_column(output_column):
    """
    Evaluates model performance by iteratively dropping columns.
    """
    logging.info("Evaluating model performance by iteratively dropping columns ...")
    ensure_directory_exists(RESULTS_DIR)

    df = load_data_old()
    datasets = preprocess_ovr(df, output_column)

    features = FEATURES.copy()
    features.remove("PolygenicScoreEXT")
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    for config_name, (X, y) in datasets.items():
        # Store results
        results = []

        for feature_pair in feature_pairs:

            # Splitting and preprocessing the data
            X_train, X_test, y_train, y_test = split_data(X, y)

            impute = imputation_pipeline()
            X_train_imputed = imputation_applier(impute, X_train)
            X_test_imputed = imputation_applier(impute, X_test)

            scaler = scaling_pipeline(X_train_imputed.columns.tolist())
            X_train_scaled, X_test_scaled = scaling_applier(scaler, X_train_imputed, X_test_imputed)

            model = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga", l1_ratio=0.5)

            # Storing the original model's performance as the baseline
            original_model = train_model(X_train_scaled, y_train, model)
            y_pred = original_model.predict(X_test_scaled)
            baseline_metrics = calculate_metrics(y_test, y_pred, MODEL_NAME, config_name, "test")

            for column in X_train_scaled.columns:  # Iterating over feature columns only
                try:
                    logger.info(f"Training model without {column} column.\n")
                    X_temp = X_train_scaled.drop(column, axis=1)

                    # Train and get the model
                    model = train_model(X_temp, y_train, model)

                    # Get predictions and calculate metrics for the test set
                    y_pred_temp = model.predict(X_test_scaled.drop(column, axis=1))
                    temp_metrics = calculate_metrics(y_test, y_pred_temp, MODEL_NAME, config_name, "test")

                    change_in_accuracy = baseline_metrics["Accuracy"] - temp_metrics["Accuracy"]
                    change_in_custom_score = baseline_metrics["Custom_Metric"] - temp_metrics["Custom_Metric"]

                    results.append({
                        "type": config_name, "column_dropped": column, "change_in_accuracy": change_in_accuracy,
                        "change_in_custom_score": change_in_custom_score
                    })

                except Exception as e:
                    logger.error(f"Error encountered when processing column {column}. Error: {str(e)}")
                    continue

            logging.info("Saving results ...\n")
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(RESULTS_DIR, f"column_drop_evaluation_{config_name}.csv"), index=False)

            logging.info("Visualizing results ...\n")
            # Visualize the results
            visualize_results(results_df)


if __name__ == "__main__":
    output_column_name = "AntisocialTrajectory"
    evaluate_with_drop_column(output_column_name)
