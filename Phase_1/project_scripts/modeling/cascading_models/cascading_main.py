from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from Phase_1.project_scripts.modeling.cascading_models.cascading_utils import *
from Phase_1.project_scripts.utility.data_loader import load_data_old
from utility.model_utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")

# Subdirectories for metrics
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")
ensure_directory_exists(METRICS_DIR)
ensure_directory_exists(GRAPH_DIR)


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

        X_train_1, y_train_1, X_test_1, y_test_1 = ap_without_it(X, y, feature_cols)

        model_1 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, class_weight='balanced')

        param_grid_1 = None  # No hyperparameter tuning

        y_train_1 = y_train_1.values.ravel()
        best_model_1 = train_model(X_train_1, y_train_1, model_1, param_grid_1)
        predictions_1 = best_model_1.predict(X_test_1)

        metrics_1 = evaluate_model(predictions_1, y_test_1)
        save_performance_metrics_csv(metrics_1, f"{target}_{task}_Model1")

        logger.info("Model 1 complete...\n")

        """
        Model 2: E -> Response (Exclude genetic features and interactions)
        """
        logger.info("Model 2 started...\n")

        E_cols = [col for col in X.columns if "PolygenicScoreEXT" not in col]  # Exclude PolygenicScoreEXT and its
        # interactions
        X = X[E_cols]

        X_train_2, y_train_2, X_test_2, y_test_2 = ap_without_it(X, y, E_cols)

        model_2 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, class_weight='balanced')

        param_grid_2 = None  # No hyperparameter tuning

        y_train_2 = y_train_2.values.ravel()
        best_model_2 = train_model(X_train_2, y_train_2, model_2, param_grid_2)
        predictions_2 = best_model_2.predict(X_test_2)

        metrics_2 = evaluate_model(predictions_2, y_test_2)
        save_performance_metrics_csv(metrics_2, f"{target}_{task}_Model2")

        logger.info("Model 2 complete...\n")

    # For Model 3: G -> E

    # Starting model 3(a)
    logger.info("Model 3(a) started...\n")

    E_outcomes = ['Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth', 'Is_Male']
    X = pd.DataFrame(df[["PolygenicScoreEXT"]])
    y = pd.DataFrame(df[E_outcomes])

    # Primary preprocessing
    y = y.apply(lambda col: col.fillna(col.mean()))

    # Secondary Preprocessing
    # X_train, y_train, X_val, y_val, X_test, y_test = ap_without_it_genetic(X, y, ["PolygenicScoreEXT"])
    X_train, y_train, X_test, y_test = ap_without_it_genetic(X, y, ["PolygenicScoreEXT"])

    # Defining the model
    model_3 = RandomForestRegressor(random_state=42)

    # Defining parameter grid for hyperparameter tuning; None implies no tuning being done.
    param_grid = None

    # Training the model
    best_model_3a = train_model(X_train, y_train, model_3, param_grid)

    # Making predictions based on the best model received.
    predictions_3 = best_model_3a.predict(X_test)

    # Calculating metrics to judge model performance.
    metrics_3a = evaluate_regression_model(predictions_3, y_test)

    # Saving model performance for further analysis.
    save_performance_metrics_csv(metrics_3a, f"{target}_Model3(a)")

    compute_and_plot_shap_values(best_model_3a, X_train, X_test, feature_name=["PolygenicScoreEXT"],
                                 outcome_names=E_outcomes)

    logger.info("Model 3(a) complete...\n")

    # Starting Model 3(b)
    logger.info("Model 3(b) started...\n")

    for outcome in E_outcomes:
        logger.info(f"Predicting for outcome: {outcome}...\n")

        X = pd.DataFrame(df[["PolygenicScoreEXT"]])
        y = pd.DataFrame(df[outcome])

        # Primary preprocessing
        y = y.fillna(y.mean())

        # Secondary Preprocessing
        X_train, y_train, X_test, y_test = ap_without_it_genetic(X, y, ["PolygenicScoreEXT"])

        # Defining the model
        model_3b = RandomForestRegressor(random_state=42)

        # Defining parameter grid for hyperparameter tuning; None implies no tuning being done.
        param_grid = None

        # Training the model
        y_train = y_train.values.ravel()
        best_model_3b = train_model(X_train, y_train, model_3b, param_grid)

        # Making predictions based on the best model received.
        predictions_3b = best_model_3b.predict(X_test)

        # Calculating metrics to judge model performance.
        metrics_3b = evaluate_regression_model(predictions_3b, y_test)

        # Saving model performance for further analysis.
        save_performance_metrics_csv(metrics_3b, f"{target}_{outcome}_Model3(b)")

    logger.info("Model 3(b) complete...\n")

    logger.info(f"Cascading approach for {target} completed...")


if __name__ == "__main__":
    main(TARGET_1)
