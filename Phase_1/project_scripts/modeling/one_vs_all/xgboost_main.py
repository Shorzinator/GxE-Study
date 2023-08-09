import logging
import os
import warnings

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from Phase_1.project_scripts.preprocessing.preprocessing import balance_data, imputation_pipeline, preprocess_ovr, \
    scaling_pipeline, split_data
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.model_utils import calculate_metrics
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CORES = 2


def train_model(outcome_val, baseline_val, X_train, X_test, y_train, y_test, preprocessor, NUM_CORES, model_dir):
    """ Train the model for the given outcome value """

    logger.info(f"Training model for outcome: {outcome_val} ...")

    # Applying preprocessing pipeline
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Balancing the dataset
    X_resampled, y_resampled = balance_data(X_train, y_train)
    logger.info(f"Data shape after balancing: {X_resampled.shape}")
    logger.info(f"Balanced training data shape for {outcome_val}: {X_resampled.shape}")
    logger.info(f"Balanced outcome value counts for {outcome_val}: \n{pd.Series(y_resampled).value_counts()}")

    # Use LabelEncoder to transform y_resampled to have values starting from 0
    le = LabelEncoder()
    y_resampled = le.fit_transform(y_resampled)

    # XGBoost model
    xgb_model = xgb.XGBClassifier(n_jobs=NUM_CORES, use_label_encoder=False, eval_metric='mlogloss', max_depth=10,
                                  learning_rate=0.1, sampling_method="gradient_based")

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
    # Train the model with default parameters
    xgb_model.fit(X_resampled, y_resampled)

    """
    grid_search = GridSearchCV(xgb_model, params, cv=StratifiedKFold(5), scoring='f1_weighted', n_jobs=NUM_CORES)
    grid_search.fit(X_resampled, y_resampled)
    """

    # Saving the model
    joblib.dump(xgb_model, os.path.join(model_dir, f"xgboost_model_{outcome_val}.pkl"))

    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    logger.info(
        f"Predicted training outcome value counts for {outcome_val}: \n{pd.Series(y_pred_train).value_counts()}")
    logger.info(f"Predicted test outcome value counts for {outcome_val}: \n{pd.Series(y_pred_test).value_counts()}")

    # Correct the calls to the calculate_metrics function
    train_metrics = calculate_metrics(y_train, y_pred_train, "xgboost", "AST", f"{outcome_val}_vs_{baseline_val}")
    test_metrics = calculate_metrics(y_test, y_pred_test, "xgboost", "AST", f"{outcome_val}_vs_{baseline_val}")

    return train_metrics, test_metrics


if __name__ == "__main__":
    df = load_data()
    logger.info(f"Original distribution of 'AST':\n{df['AntisocialTrajectory'].value_counts()}\n")
    print()

    target = "AntisocialTrajectory"

    datasets = preprocess_ovr(df, target)  # preprocessing
    logger.info(f"Data shape after preprocessing: {df.shape}")
    logger.info(f"Original distribution of 'AST':\n{df['AntisocialTrajectory'].value_counts()}\n")
    print()

    # Establish the model-specific directories
    model_name = "xgboost"

    # Use the path utility to get the path for xgboost_results
    results_dir = get_path_from_root("results", "one_vs_all", f"{model_name}_results")

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
    baseline_val = 4

    for target_val, (data, outcome) in datasets.items():
        logger.info(f"Data shape after preprocessing for {target_val}: {data.shape}")
        logger.info(f"Outcome value counts for {target_val}: \n{outcome.value_counts()}")

        # Split data
        X_train, X_test, y_train, y_test = split_data(data, outcome)
        logger.info(f"Training data shape for {target_val}: {X_train.shape}")
        logger.info(f"Test data shape for {target_val}: {X_test.shape}")
        logger.info(f"Training outcome value counts for {target_val}: \n{y_train.value_counts()}")
        logger.info(f"Test outcome value counts for {target_val}: \n{y_test.value_counts()}")

        # Applying imputation
        impute = imputation_pipeline(X_train)
        X_train = impute.fit_transform(X_train)
        X_test = impute.transform(X_test)

        # Applying scaling
        scaler = scaling_pipeline(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        logger.info(f"Training data shape after imputation and scaling for {target_val}: {X_train.shape}")
        logger.info(f"Test data shape after imputation and scaling for {target_val}: {X_test.shape}")

        train_metrics, test_metrics = train_model(target_val,
                                                  baseline_val,
                                                  X_train,
                                                  X_test,
                                                  y_train,
                                                  y_test,
                                                  preprocessor,
                                                  NUM_CORES,
                                                  model_dir)

        metrics_dict[f"{target_val}_vs_{baseline_val}"] = {"train": train_metrics, "test": test_metrics}

    # Saving metrics to CSV
    for split, split_data in metrics_dict.items():
        flattened_data = {}
        for key, value in split_data.items():
            if value is not None:
                for metric, metric_value in value.items():
                    flattened_data[f"{key}_{metric}"] = metric_value
            else:
                logger.warning(f"No metrics found for {key} in split {split}.")
        df_metrics = pd.DataFrame([flattened_data])
        df_metrics.to_csv(os.path.join(metrics_dir, f"AST_{split}_vs_{baseline_val}.csv"))

    logger.info("Completed.")