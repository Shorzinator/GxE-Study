import logging
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, train_model
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join("results", "evaluation")
ensure_directory_exists(RESULTS_DIR)
MODEL_NAME = "logistic_regression"

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
                logger.info(f"Training model without {column} column.")
                X_temp = X.drop(column, axis=1)

                # Train and get the model
                model = train_model(X_temp, y, model)

                # Get predictions and calculate metrics for train set
                y_pred_temp = model.predict(X_temp)
                temp_metrics = calculate_metrics(y, y_pred_temp, model.__class__.__name__, key, "train")

                change_in_accuracy = baseline_metrics["accuracy"] - temp_metrics["accuracy"]
                change_in_custom_score = baseline_metrics["custom_score"] - temp_metrics["custom_score"]

                results.append({
                                   "type": key, "column_dropped": column, "change_in_accuracy": change_in_accuracy,
                                   "change_in_custom_score": change_in_custom_score
                               })

            except Exception as e:
                logger.error(f"Error encountered when processing column {column}. Error: {str(e)}")
                continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "column_drop_evaluation.csv"), index=False)


if __name__ == "__main__":
    interaction_term_column = "PolygenicScoreEXT_x_Is_Male"
    output_column_name = "AntisocialTrajectory"
    evaluate_with_drop_column(interaction_term_column, output_column_name)
