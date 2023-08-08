import logging
import os
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.preprocessing.preprocessing import preprocess_data, split_data, imputation_pipeline, \
    balance_data
from Phase_1.project_scripts.utility.model_utils import calculate_metrics
from Phase_1.project_scripts.utility.path_utils import get_path_from_root
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CORES = 2


def train_model(outcome_val, X_train, X_test, y_train, y_test, preprocessor, NUM_CORES, model_dir):
    """ Train the model for the given outcome value """

    logger.info(f"Training model for outcome: {outcome_val} ...")

    # Applying preprocessing pipeline
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Balancing the dataset
    X_resampled, y_resampled = balance_data(X_train, y_train)
    y_resampled = y_resampled - 1

    balanced_data_path = get_path_from_root("data", "processed", f"balanced_data_{outcome_val}.csv")
    pd.DataFrame(X_resampled).to_csv(balanced_data_path, index=False)

    # XGBoost model
    xgb_model = xgb.XGBClassifier(n_jobs=NUM_CORES, use_label_encoder=False, eval_metric='mlogloss')

    # Randomized search for hyperparameter tuning
    params = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.5],
        'n_estimators': [50, 100, 200, 500],
        'gamma': [0, 0.5, 1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'objective': ['binary:logistic']
    }

    grid_search = GridSearchCV(xgb_model, params, cv=StratifiedKFold(5), scoring='f1_weighted', n_jobs=NUM_CORES)
    grid_search.fit(X_resampled, y_resampled)

    # Saving the model
    joblib.dump(grid_search.best_estimator_, os.path.join(model_dir, f"xgboost_model_{outcome_val}.pkl"))

    y_pred_train = grid_search.predict(X_train)
    y_pred_test = grid_search.predict(X_test)

    # Correct the calls to the calculate_metrics function
    train_metrics = calculate_metrics(y_train, y_pred_train, "xgboost", "AST", f"1_vs_{outcome_val}")
    test_metrics = calculate_metrics(y_test, y_pred_test, "xgboost", "AST", f"1_vs_{outcome_val}")

    return train_metrics, test_metrics


if __name__ == "__main__":
    df = load_data()

    df, outcome = preprocess_data(df, "AST")  # preprocessing
    X_train, X_test, y_train, y_test = split_data(df, outcome)  # splitting
    preprocessor = imputation_pipeline(df)  # imputation and scaling

    # Establish the model-specific directories
    model_name = "xgboost"

    # Use the path utility to get the path for xgboost_results
    results_dir = get_path_from_root("Phase_1", "results", "one_vs_all", f"{model_name}_results")

    # Ensure the xgboost_results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Subdirectories for model and metrics
    model_dir = os.path.join(results_dir, "models")
    metrics_dir = os.path.join(results_dir, "metrics")

    # Ensure the subdirectories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    metrics_dict = {}

    for outcome_val in [1, 2, 3]:
        train_metrics, test_metrics = train_model(outcome_val, X_train, X_test, y_train, y_test, preprocessor,
                                                  NUM_CORES, model_dir)
        metrics_dict[outcome_val] = {"train": train_metrics, "test": test_metrics}

    # Saving metrics to CSV
    for split, split_data in metrics_dict.items():
        df_metrics = pd.DataFrame(split_data).T
        df_metrics.to_csv(os.path.join(metrics_dir, f"xgboost_metrics_{split}.csv"))

    logger.info("Completed.")
