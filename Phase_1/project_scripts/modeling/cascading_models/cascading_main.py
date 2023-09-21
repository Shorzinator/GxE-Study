from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, r2_score, roc_auc_score

from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import *
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")
ensure_directory_exists(RESULTS_DIR)

# Subdirectories for metrics
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
ensure_directory_exists(METRICS_DIR)


def save_performance_metrics_csv(metrics, target):
    """
    Save the performance metrics to a CSV file

    :param metrics: Dict, performance metrics
    :param target: str, target variable name
    """
    filename = f"{target}_performance.csv"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert dictionary to DataFrame for easier CSV saving
    metrics_df = pd.DataFrame([metrics])

    # Save to CSV
    metrics_df.to_csv(filepath, index=False)

    logger.info(f"Performance metrics saved to {filepath}...\n")


def evaluate_model(predictions, y_true):
    """
    Evaluate model performance on given predictions and true labels.

    :param predictions: array-like, model predictions
    :param y_true: array-like, true labels
    :return: dict, containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average="micro")
    auc = roc_auc_score(y_true, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "auc": auc
    }

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")

    return metrics


def evaluate_regression_model(predictions, y_true):
    """
    Evaluate regression model performance on given predictions and true labels.

    :param predictions: Array-like, model predictions
    :param y_true: Array-like, true labels
    :return: Dict, containing evaluation metrics
    """

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2
    }

    return metrics


def main(target):
    logger.info(f"Starting Cascading Approach for {target}...\n")

    # Load the data
    df = load_data_old()

    # For Model 1 and Model 2
    datasets, feature_cols = preprocess_sut_ovr(df, FEATURES_FOR_SUT) if target == "SubstanceUseTrajectory" \
        else preprocess_ast_ovr(df, FEATURES_FOR_AST)

    for task, (X, y) in datasets.items():
        """
        Model 1: G + E -> Response
        """
        logger.info("Model 1 started...\n")

        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = (
            ap_without_it(X, y, feature_cols)
        )

        model_1 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, class_weight='balanced')

        param_grid_1 = None  # No hyperparameter tuning

        best_model_1 = train_model(X_train_1, y_train_1, model_1, param_grid_1)
        predictions_1 = best_model_1.predict(X_val_1)

        metrics_1 = evaluate_model(predictions_1, y_val_1)
        save_performance_metrics_csv(metrics_1, f"{target}_{task}_Model1")

        logger.info("Model 1 complete...\n")

        """
        Model 2: E -> Response (Exclude genetic features and interactions)
        """
        logger.info("Model 2 started...\n")

        E_cols = [col for col in X.columns if "PolygenicScoreEXT" not in col]  # Exclude PolygenicScoreEXT and its
        # interactions
        X = X[E_cols]

        X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = (
            ap_without_it(X, y, E_cols)
        )

        model_2 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, class_weight='balanced')

        param_grid_2 = None  # No hyperparameter tuning

        best_model_2 = train_model(X_train_2, y_train_2, model_2, param_grid_2)
        predictions_2 = best_model_2.predict(X_val_2)

        metrics_2 = evaluate_model(predictions_2, y_val_2)
        save_performance_metrics_csv(metrics_2, f"{target}_{task}_Model2")

        logger.info("Model 2 complete...\n")

    # For Model 3: G -> E
    logger.info("Model 3 started...\n")

    E_outcomes = ['Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth', 'Is_Male']
    X = pd.DataFrame(df[["PolygenicScoreEXT"]])
    y = pd.DataFrame(df[E_outcomes])

    # Primary preprocessing
    y = y.apply(lambda col: col.fillna(col.mean()))

    # Secondary Preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test = ap_without_it_genetic(X, y, ["PolygenicScoreEXT"])

    # Initialize KNNImputer
    imputer = KNNImputer(n_neighbors=5)

    # Fit on y_train
    imputer.fit(y_train)

    # Transform (impute) the datasets
    y_train_imputed = imputer.transform(y_train)
    y_val_imputed = imputer.transform(y_val)
    y_test_imputed = imputer.transform(y_test)

    # If needed, convert back to DataFrame
    y_train = pd.DataFrame(y_train_imputed, columns=y_train.columns)
    y_val = pd.DataFrame(y_val_imputed, columns=y_val.columns)
    y_test = pd.DataFrame(y_test_imputed, columns=y_test.columns)

    # Defining the model
    model_3 = RandomForestRegressor(random_state=42)

    # Defining parameter grid for hyperparameter tuning; None implies no tuning being done.
    param_grid_3 = None

    # Training the model
    best_model_3 = train_model(X_train, y_train, model_3, param_grid_3)

    # Making predictions based on the best model received.
    predictions_3 = best_model_3.predict(X_val)

    # Calculating metrics to judge model performance.
    metrics_3 = evaluate_regression_model(predictions_3, y_val)

    # Saving model performance for further analysis.
    save_performance_metrics_csv(metrics_3, f"{target}_Model3")

    logger.info("Model 3 complete...\n")

    logger.info(f"Cascading approach for {target} completed...")


if __name__ == "__main__":
    main(TARGET_1)
