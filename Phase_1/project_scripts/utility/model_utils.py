import os

import joblib
import optuna
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score

from Phase_1.config import *
from Phase_1.project_scripts.preprocessing.preprocessing import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hyperparameter_tuning(X_train, y_train, estimator, tuning_method="grid_search", **kwargs):
    """
    Performs hyperparameter tuning based on the specified method.

    Parameters:
    - X_train, y_train: Training data
    - estimator: Model instance to be tuned
    - tuning_method: Method for hyperparameter tuning ("grid_search", "random_search", "smbo", etc.)
    - kwargs: Additional arguments required for the specific tuning method

    Returns:
    - A dictionary containing the best hyperparameters, the best score, and detailed evaluation log
    """

    if tuning_method == "grid_search":
        return grid_search_tuning(X_train, y_train, estimator, **kwargs)

    elif tuning_method == "random_search":
        return random_search_tuning(X_train, y_train, estimator, **kwargs)

    elif tuning_method == "smbo":
        return smbo_tuning(X_train, y_train, estimator, **kwargs)

    else:
        raise ValueError(f"Tuning method {tuning_method} not recognized.")


def add_interaction_terms(df, feature_pairs):
    """
    Generate interaction terms for specified feature pairs iteratively.

    Args:
    :param df: (pd.DataFrame) Original dataset.
    :param feature_pairs: (List of tuples) List of tuples where each tuple contains feature columns
                          for which interaction terms are to be generated.
    :return: Df (pd.DataFrame): Dataset with added interaction terms.
    """
    logger.info(f"Generating interaction term for features: {feature_pairs}\n")

    # Create the interaction term
    interaction_column_name = f"{feature_pairs[0]}_x_{feature_pairs[1]}"
    df[interaction_column_name] = df[feature_pairs[0]] * df[feature_pairs[1]]

    # logger.info(f"Updated columns are: {df.columns}\n")

    return df


def save_results(target, type_of_classification, results, directory, combine=None, it=None):
    """
    Save the results in a structured directory and file.
    :param it:
    :param combine:
    :param directory: Model_dir or metrics_dir
    :param type_of_classification: multinomial, binary, etc.
    :param target: Target variable (either "AST" or "SUT")
    :param results: The results data (a dictionary)
    """
    try:
        # Check if the results need to be flattened
        if "train_metrics" in results[0] or "test_metrics" in results[0]:
            flattened_data = []
            for res in results:
                interaction_name = res.get("interaction", "N/A")
                for key, metrics in res.items():
                    if key in ["train_metrics", "test_metrics"]:
                        flattened_data.append({"type": key, "interaction": interaction_name, **metrics})
            results_df = pd.DataFrame(flattened_data)
        else:
            results_df = pd.DataFrame(results)

        dir_path = directory

        # Check and create the directory if it doesn't exist
        ensure_directory_exists(dir_path)

        results_file = os.path.join(dir_path, f"{target}_{type_of_classification}.csv")

        # Save to CSV
        results_df.to_csv(results_file, index=False)

    except Exception as e:
        logger.error(f"Error in save_results: {str(e)}")


def train_model(X_train, y_train, estimator, param_grid=None, save_model=False, model_name=None,
                model_dir=None, IT=None):
    """
    Train the model, optionally perform grid search, and save it.

    Args:
    :param IT:
    :param X_train: Training data.
    :param y_train: Training labels.
    :param estimator: The model/estimator to be trained.
    :param param_grid: Hyperparameters for grid search. If None, no grid search will be performed.
    :param save_model: Boolean flag to decide whether to save the model or not.
    :param model_name: Name of the model, required if save_model is True.
    :param model_dir: Directory to save model, required if save_model is True.

    Returns:
    :return: Trained model.
    """

    if param_grid:
        # If param_grid is provided, perform GridSearchCV
        cv_method = get_cv_method()
        best_model = grid_search_tuning(estimator, param_grid, 'f1_weighted', cv_method)
        best_model.fit(X_train, y_train)
    else:
        best_model = estimator

        logger.info("Fitting the model...\n")
        best_model.fit(X_train, y_train)

    if save_model and model_name:
        # Save the model using joblib
        model_path = os.path.join(model_dir, f"{TARGET_1}_{model_name}_{COMBINED}_{IT}.pkl")
        joblib.dump(best_model, model_path)

    return best_model


def ensure_directory_exists(directory):
    """Ensure a directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_metrics(y_true, y_pred, model_name, target, type, weights=None):
    """
    Calculate metrics for the multinomial model predictions.
    :param type: Test or train
    :param y_true: True labels
    :param y_pred: Predicted label
    :param model_name: Name of the model
    :param target: Target column name
    :param weights: Weights for the custom metric. Should sum to 1.
    :return: A dictionary containing the calculated metrics
    """

    logger.info("Calculating Metrics...\n")

    # Set default weights if none provided
    if weights is None:
        weights = {
            "Accuracy": 0.2,
            "MCC": 0.2,
            "Precision": 0.2,
            "Recall": 0.2,
            "F1-Score": 0.2
        }

    # Check if predictions are all one class
    assert len(set(y_pred)) != 1, "y_pred must contain more than 1 label"

    # Get a classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "Target": target,
        "Accuracy": accuracy,
        "Matthews Correlation Coefficient": mcc
    }

    # Extract metrics for all classes and calculate custom metric
    custom_metric_value = accuracy * weights["Accuracy"] + mcc * weights["MCC"]
    for cls, cls_report in report.items():
        if isinstance(cls_report, dict):  # To ensure we're processing a class
            precision = cls_report['precision']
            recall = cls_report['recall']
            f1_score = cls_report['f1-score']

            custom_metric_value += (precision * weights["Precision"] +
                                    recall * weights["Recall"] +
                                    f1_score * weights["F1-Score"])

            metrics[f"{cls}_Precision"] = precision
            metrics[f"{cls}_Recall"] = recall
            metrics[f"{cls}_F1-Score"] = f1_score

    metrics["Custom_Metric"] = custom_metric_value

    return metrics


def get_cv_method(method='KFold', n_splits=5):
    """
    Return a cross-validation method based on user's choice.

    Args:
    :param method: The desired cross-validation method. Default is 'KFold'
    :param n_splits: Number of splits. Relevant for KFold. Default is 5.

    Returns:
    :return: An instance of the chosen cross-validation method.
    """
    if method == 'KFold':
        return KFold(n_splits=n_splits)
    # Add other CV methods as needed
    else:
        raise ValueError(f"Unknown CV method: {method}")


def smbo_tuning(X_train, y_train, estimator, n_trials=100, **kwargs):
    """
    Performs Sequential Model-Based Optimization (SMBO) using Optuna.

    Parameters:
    - X_train, y_train: Training data
    - estimator: Model instance to be tuned
    - n_trials: Number of trials for optimization
    - kwargs: Additional arguments

    Returns:
    - A dictionary containing the best hyperparameters, best score, and the study object for detailed analysis
    """

    # Define the objective function for Optuna
    def objective(trial):
        # Extract hyperparameters from the trial object
        # For demonstration purposes, let's assume two hyperparameters: n_estimators and max_depth
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 2, 50, log=True)

        # Initialize the estimator with the suggested hyperparameters
        clf = estimator(n_estimators=n_estimators, max_depth=max_depth)

        # Use cross-validation to evaluate the model.
        # This can be replaced with other evaluation methods.
        # For simplicity, we'll use a fixed 5-fold cross-validation here.
        scores = cross_val_score(clf, X_train, y_train, cv=5)

        # Return the negative mean score as Optuna tries to minimize the objective.
        return -1 * scores.mean()

    # Create a study object and specify the direction is 'minimize'.
    study = optuna.create_study(direction='minimize')

    # Optimize the study, the objective function is passed in as the first argument.
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_parameters": study.best_params,
        "best_score": -1 * study.best_value,  # Convert back to positive as we minimized the negative score
        "study": study  # The study object can be used for further analysis or visualization
    }


def grid_search_tuning(estimator, param_grid, scoring, cv):
    """
    Performs hyperparameter tuning using Grid Search.

    Parameters:
    - X_train, y_train: Training data
    - estimator: Model instance to be tuned
    - param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    - kwargs: Additional arguments for GridSearchCV

    Returns:
    - A dictionary containing the best hyperparameters, best score, and the complete results grid
    """

    grid_search = GridSearchCV(estimator, param_grid, scoring, cv)

    return grid_search


def random_search_tuning(estimator, param_distributions, n_iter=100, **kwargs):
    """
    Performs hyperparameter tuning using Random Search.

    Parameters:
    - X_train, y_train: Training data
    - estimator: Model instance to be tuned
    - param_distributions:
    Dictionary with parameter names (str) as keys and distributions or lists of parameters to try.
    - n_iter: Number of parameter settings that are sampled.
    - kwargs: Additional arguments for RandomizedSearchCV

    Returns:
    - A dictionary containing the best hyperparameters, best score, and a list of evaluated combinations
    """

    random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, **kwargs)

    return random_search
