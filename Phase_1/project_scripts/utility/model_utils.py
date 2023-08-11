import os

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, KFold

from Phase_1.config import *
from Phase_1.project_scripts.preprocessing.preprocessing import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_interaction_terms(df, feature_pairs):
    """
    Generate interaction terms for specified feature pairs iteratively.

    Args:
    :param df: (pd.DataFrame) Original dataset.
    :param feature_pairs: (list of tuples) List of tuples where each tuple contains feature columns
                          for which interaction terms are to be generated.
    :return: df (pd.DataFrame): Dataset with added interaction terms.
    """

    logger.info(f"Generating interaction term for features: {feature_pairs}\n")

    # Create the interaction term
    interaction_column_name = f"{feature_pairs[0]}_x_{feature_pairs[1]}"
    df[interaction_column_name] = df[feature_pairs[0]] * df[feature_pairs[1]]

    # logger.info(f"Updated columns are: {df.columns}\n")

    return df


def save_results(target, type_of_classification, results, directory):
    """
    Save the results in a structured directory and file.
    :param directory: model_dir or metrics_dir
    :param type_of_classification: multinomial, binary, etc.
    :param target: Target variable (either "AST" or "SUT")
    :param results: The results data (a dictionary)
    """
    logger.info("Saving results ...\n")

    # Check if the results need to be flattened
    if any(isinstance(val, dict) for val in results.values()):
        flattened_data = [
            {"type": key, **metrics}
            for key, metrics in results.items() if key in ["train_metrics", "test_metrics"]
        ]
        results_df = pd.DataFrame(flattened_data)
    else:
        results_df = pd.DataFrame(results)

    dir_path = directory

    # Check and create the directory if it doesn't exist
    ensure_directory_exists(dir_path)

    results_file = os.path.join(dir_path, f"{target}_{type_of_classification}_{COMBINED}_{IT}.csv")

    # Save to CSV
    results_df.to_csv(results_file, index=False)


def train_model(X_train, y_train, estimator, param_grid=None, save_model=False, model_name=None,
                metric_dir=None, model_dir=None):
    """
    Train the model, optionally perform grid search, and save it.

    Args:
    :param X_train: Training data.
    :param y_train: Training labels.
    :param estimator: The model/estimator to be trained.
    :param param_grid: Hyperparameters for grid search. If None, no grid search will be performed.
    :param save_model: Boolean flag to decide whether to save the model or not.
    :param model_name: Name of the model, required if save_model is True.
    :param metric_dir: Directory to save metrics, required if target is provided.
    :param model_dir: Directory to save model, required if save_model is True.
    :param target: Target column name, required if metric_dir is provided.

    Returns:
    :return: Trained model.
    """

    logger.info("Training the multinomial logistic regression model...\n")

    if param_grid:
        # If param_grid is provided, perform GridSearchCV
        cv_method = get_cv_method()
        grid_search = perform_gcv(estimator, param_grid, 'f1_weighted', cv_method)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = estimator
        best_model.fit(X_train, y_train)

    if save_model and model_name:
        # Save the model using joblib
        model_path = os.path.join(model_dir, f"{TARGET_1}_{model_name}_{COMBINED}.pkl")
        joblib.dump(best_model, model_path)

    return best_model


def ensure_directory_exists(directory):
    """Ensure a directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_metrics(y_true, y_pred, model_name, target, type):
    """
    Calculate metrics for the multinomial model predictions.
    :param type: test or train
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param model_name: Name of the model
    :param target: Target column name
    :return: A dictionary containing the calculated metrics
    """

    # Log unique classes for validation
    # logger.info(f"Unique classes in true labels for {type}: {set(y_true)}")
    # logger.info(f"Unique classes in predicted labels for {type}: {set(y_pred)}")

    # Check if predictions are all one class
    if len(set(y_pred)) == 1:
        logger.warning(f"All predictions are of class {y_pred[0]}.")

    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "Target": target,
        "Accuracy": accuracy
    }

    # Extract metrics for all classes
    for cls, cls_report in report.items():
        if isinstance(cls_report, dict):  # To ensure we're processing a class
            metrics[f"{cls}_Precision"] = cls_report['precision']
            metrics[f"{cls}_Recall"] = cls_report['recall']
            metrics[f"{cls}_F1-Score"] = cls_report['f1-score']

    return metrics


def get_cv_method(method='KFold', n_splits=5):
    """
    Return a cross-validation method based on user's choice.

    Args:
    :param method: The desired cross-validation method. Default is 'KFold'.
    :param n_splits: Number of splits. Relevant for KFold. Default is 5.

    Returns:
    :return: An instance of the chosen cross-validation method.
    """
    if method == 'KFold':
        return KFold(n_splits=n_splits)
    # Add other CV methods as needed
    else:
        raise ValueError(f"Unknown CV method: {method}")


def perform_gcv(estimator, params, scoring, cv_method):
    """
    Performs GridSearchCV with the provided estimator, parameters and scoring.

    Args:
    :param estimator: The model/estimator for which the grid search is performed.
    :param params: Hyperparameters for grid search.
    :param scoring: The scoring metric used.
    :param cv_method: The cross-validation method.

    Returns:
    :return: An instance of GridSearchCV.
    """
    return GridSearchCV(estimator=estimator, param_grid=params, scoring=scoring, cv=cv_method, n_jobs=-1)


