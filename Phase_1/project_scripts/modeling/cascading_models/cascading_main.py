from sklearn.linear_model import LogisticRegression

from Phase_1.project_scripts.utility.data_loader import load_data_old
from Phase_1.project_scripts.utility.model_utils import *
from Phase_1.project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "cascading"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")


def main(target):
    logger.info(f"Starting Cascading Approach for {target}...\n")

    # Subdirectories for metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    ensure_directory_exists(metrics_dir)

    # Load the data
    df = load_data_old()

    # For the response models (Models 1 and 2), we need to preprocess the data for binary classification tasks
    if target in ["SubstanceUseTrajectory", "AntisocialTrajectory"]:
        datasets, feature_cols = preprocess_sut_ovr(df, FEATURES_FOR_SUT) if target == "SubstanceUseTrajectory" \
            else preprocess_ast_ovr(df, FEATURES_FOR_AST)

        for task, (X, y) in datasets.items():
            """
            Model 1: G + E -> Response
            """
            X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = (
                apply_preprocessing_without_interaction_terms(X, y, feature_cols)
            )

            model_1 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                         l1_ratio=0.5, class_weight='balanced')

            param_grid = None   # No hyperparameter tuning

            best_model_1 = train_model(X_train_1, y_train_1, model_1, param_grid)
            predictions_1 = best_model_1.predict(X_val_1)
            evaluate_model(predictions_1, y_val_1)

            """
            Model 2: E -> Response (Exclude genetic features and interactions)
            """

            X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = (
                apply_preprocessing_without_interaction_terms(X.drop(["PolygenicScoreEXT",
                                                                      "PolygenicScoreEXT_x_Is_Male",
                                                                      "PolygenicScoreEXT_x_Age"], axis=1),
                                                              y, feature_cols)
            )

            model_2 = LogisticRegression(max_iter=10000, multi_class='ovr', penalty="elasticnet", solver="saga",
                                         l1_ratio=0.5, class_weight='balanced')

            param_grid = None  # No hyperparameter tuning

            best_model_2 = train_model(X_train_2, y_train_2, model_2, feature_cols)
            predictions_2 = best_model_2.predict(X_val_2)
            evaluate_model(predictions_2, y_val_2)

    # For the Model 3 (G -> E), we'll use regression or multi-label classification depending on the nature of E features
    else:
        # Since E features are continuous, we'd use regression.
        X = df["PolygenicScoreEXT"]
        y = df[X.drop([""])]
