import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, accuracy_score, \
    precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, learning_curve, train_test_split
from xgboost import XGBClassifier


# The split_data function to include a validation set
def split_data(df, target):
    # Split the data into training+validation and testing sets with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target])

    # Further split the training+validation set into separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # This will split the training+validation into 60% training and 20% validation
        random_state=42,
        stratify=y_train_val)

    return pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test), \
        pd.DataFrame(y_train), pd.DataFrame(y_val), pd.DataFrame(y_test)


def prep_data_for_race_model(
        X_train_new_enhanced, y_train_new_mapped, X_val_new_enhanced, y_val_new_mapped,
        X_test_new_enhanced, y_test_new_mapped, race, race_column
):
    X_train_race = X_train_new_enhanced[X_train_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_train_race = y_train_new_mapped[X_train_new_enhanced[race_column] == race].ravel()

    X_val_race = X_val_new_enhanced[X_val_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_val_race = y_val_new_mapped[X_val_new_enhanced[race_column] == race].ravel()

    X_test_race = X_test_new_enhanced[X_test_new_enhanced[race_column] == race].drop(columns=[race_column])
    y_test_race = y_test_new_mapped[X_test_new_enhanced[race_column] == race].ravel()

    return X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race


# Declaring variable to prevent overfitting
early_stopping_rounds = 150


def train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test, final_model_name, tune=False, race=None,
        model_type="base", cv=5, resampling="with", outcome="AntisocialTrajectory"
        ):
    """
    Train and evaluate a model with optional hyperparameter tuning and cross-validation.

    Parameters:
    - model: The model to be trained.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - params: Parameter grid for hyperparameter tuning.
    - tune: Whether to perform hyperparameter tuning.
    - check_overfitting: Whether to check if the model is overfitting or not
    - model_type: Type of model ("base" or "final").
    - race: The race identifier for race-specific final models, if applicable.
    - n_splits: Number of splits for cross-validation.
    """

    model_name = f"{model_type} model" + (f" (race {race})" if race else "")

    # tag = "AST" if outcome == "AntisocialTrajectory" else "SUT"
    # model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models", "classification", tag)
    # param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "param", "classification", tag)

    # If 'tuning' is toggled to True
    if tune:
        all_params = search_spaces()
        params = all_params[final_model_name]
        model, best_params = random_search_tuning(model, params, X_train, y_train, cv=cv)

        print(f"Best Parameters for {model_name} {resampling} resampling: \n{best_params} \n")

    # Fitting the trained model
    model.fit(
        X_train,
        y_train,
        # eval_set=[(X_val, y_val)],
        # early_stopping_rounds=early_stopping_rounds,
        # verbose=False
    )
    # Get predictions using the trained model
    y_pred_train, y_pred_val, y_pred_test = get_pred_values(model, X_train, X_val, X_test)
    print(np.unique(y_pred_train), np.unique(y_pred_val), np.unique(y_pred_test))

    # Get probability values using the trained model
    y_prob_train, y_prob_val, y_prob_test = get_prob_values(model, X_train, X_val, X_test)

    # Calculate and print the accuracies
    calc_accuracy(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)

    # Calculate and print the AUC
    calc_roc_auc(y_train, y_prob_train, y_val, y_prob_val, y_test, y_prob_test)

    # Calculate and print the f1 score's
    # calc_f1_score(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)

    # Check if the model being evaluated is overfitting on the outcome currently under consideration or not.
    evaluate_overfitting(
        train_accuracy=accuracy_score(y_train, y_pred_train),
        val_accuracy=accuracy_score(y_val, y_pred_val),
        y_train_true=y_train,
        y_train_pred=model.predict(X_train),
        y_val_true=y_val,
        y_val_pred=model.predict(X_val),
        model_name=model_name,
        resampling=resampling
    )

    return model


def train_and_evaluate_with_race_feature(
        model, X_train, y_train, X_val, y_val, X_test, y_test, final_model_name, tune=False, model_type="final", cv=5,
        resampling="with", script_name=None, outcome="AntisocialTrajectory", race_column='Race'
):
    """
    Train and evaluate a model, analyzing performance separately for each race within the dataset.

    Parameters:
    - model: The model instance to be trained and evaluated.
    - X_train, y_train, X_val, y_val, X_test, y_test: Data splits for training, validation, and testing.
    - final_model_name: Name of the final model, e.g. LogisticRegression, etc.
    - tune: Whether to tune the model.
    - model_type: Type of model (e.g., "base" or "final").
    - cv: Number of cross-validation folds.
    - resampling: Resampling strategy used.
    - script_name: Name of the script, if applicable.
    - outcome: Name of the outcome variable.
    - race_column: Name of the column containing race information.
    """
    model_name = f"{model_type} model"

    # tag = "AST" if outcome == "AntisocialTrajectory" else "SUT"
    # model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models", "classification", tag)
    # param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "param", "classification", tag)

    unique_races = X_train[race_column].unique()

    if tune:
        all_params = search_spaces()
        params = all_params[final_model_name]
        model, best_params = random_search_tuning(model, params, X_train, y_train, cv=cv)

        print(f"Best Parameters for {model_name} {resampling} resampling: \n{best_params} \n")

    for race in unique_races:
        print(f"Evaluating model for race: {race}")
        # Filter the dataset for the current race
        indices_train = X_train[race_column] == race
        indices_val = X_val[race_column] == race
        indices_test = X_test[race_column] == race

        X_train_race = X_train.loc[indices_train].drop(columns=[race_column])
        y_train_race = y_train[indices_train]

        X_val_race = X_val.loc[indices_val].drop(columns=[race_column])
        y_val_race = y_val[indices_val]

        X_test_race = X_test.loc[indices_test].drop(columns=[race_column])
        y_test_race = y_test[indices_test]

        # Train the model
        model.fit(
            X_train_race,
            y_train_race,
            # eval_set=[(X_val_race, y_val_race)],
            # early_stopping_rounds=early_stopping_rounds,
            # verbose=False
        )

        # Get predictions using the trained model
        y_pred_train, y_pred_val, y_pred_test = get_pred_values(model, X_train_race, X_val_race, X_test_race)
        print(np.unique(y_pred_train), np.unique(y_pred_val), np.unique(y_pred_test))

        # Get probability values using the trained model
        y_prob_train, y_prob_val, y_prob_test = get_prob_values(model, X_train_race, X_val_race, X_test_race)

        # Calculate and print the accuracies
        calc_accuracy(y_train_race, y_pred_train, y_val_race, y_pred_val, y_test_race, y_pred_test)

        # Calculate and print the AUC
        calc_roc_auc(y_train_race, y_prob_train, y_val_race, y_prob_val, y_test_race, y_prob_test)

        # Calculate and print the f1 score's
        # calc_f1_score(y_train_race, y_pred_train, y_val_race, y_pred_val, y_test_race, y_pred_test)

        # Calculate the SHAP values and plot them
        # calc_shap_values(model, X_train_race, race)

        # Plotting Calibration Curve
        # labels = np.unique(y_val_race)
        # plot_calibration_curve(y_val_race, y_prob_val, labels)

        # plot_confusion_matrix(y_train_race, y_pred_train, labels)

        # Check for overfitting
        evaluate_overfitting(
            train_accuracy=accuracy_score(y_train_race, y_pred_train),
            val_accuracy=accuracy_score(y_val_race, y_pred_val),
            y_train_true=y_train_race,
            y_train_pred=y_pred_train,
            y_val_true=y_val_race,
            y_val_pred=y_pred_val,
            model_name=model_name,
            resampling=resampling
        )

    # Return the model
    return model


def get_pred_values(model, X_train, X_val, X_test):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_val, y_pred_test


def get_prob_values(model, X_train, X_val, X_test):
    y_prob_train = model.predict_proba(X_train)
    y_prob_val = model.predict_proba(X_val)
    y_prob_test = model.predict_proba(X_test)

    return y_prob_train, y_prob_val, y_prob_test


def calc_accuracy(y_train_race, y_pred_train, y_val_race, y_pred_val, y_test_race, y_pred_test):
    train_accuracy = accuracy_score(y_train_race, y_pred_train)
    val_accuracy = accuracy_score(y_val_race, y_pred_val)
    test_accuracy = accuracy_score(y_test_race, y_pred_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # print("\nPer-class metrics:")
    # print("Training:")
    # print(classification_report(y_train_race, y_pred_train))
    # print("Validation:")
    # print(classification_report(y_val_race, y_pred_val))
    # print("Testing:")
    # print(classification_report(y_test_race, y_pred_test))


def calc_f1_score(y_train_race, y_pred_train, y_val_race, y_pred_val, y_test_race, y_pred_test):
    f1_train = f1_score(y_train_race, y_pred_train, average='macro')
    f1_val = f1_score(y_val_race, y_pred_val, average='macro')
    f1_test = f1_score(y_test_race, y_pred_test, average='macro')

    print(f"Training F-1 Score: {f1_train:.4f}")
    print(f"Validation F-1 Score: {f1_val:.4f}")
    print(f"Testing F-1 Score: {f1_test:.4f}")


def calc_roc_auc(y_train_race, y_prob_train, y_val_race, y_prob_val, y_test_race, y_prob_test):
    train_roc_auc = roc_auc_score(y_train_race, y_prob_train, multi_class='ovr')
    val_roc_auc = roc_auc_score(y_val_race, y_prob_val, multi_class='ovr')
    test_roc_auc = roc_auc_score(y_test_race, y_prob_test, multi_class='ovr')

    print(f"Training ROC AUC: {train_roc_auc:.4f}")
    print(f"Validation ROC AUC: {val_roc_auc:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")


def plot_calibration_curve(y_true, y_prob, labels):
    """
    Plot the calibration curve for the model.

    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Predicted probabilities for each class.
        labels (list): List of class labels.
    """
    plt.figure(figsize=(8, 6))

    for i in range(len(labels)):
        y_true_class = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_class, y_prob_class, n_bins=10)

        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'Class {labels[i]}')

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot the confusion matrix as a heatmap and table.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.show()

    print("Confusion Matrix Table:")
    print(cm)


def get_mapped_data(y_train_new, y_val_new, y_test_new, y_train_old=None, y_val_old=None, y_test_old=None):
    # Map labels to start from 0
    # For new data
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}

    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)
    y_test_new_mapped = np.vectorize(label_mapping_new.get)(y_test_new)

    y_train_new_mapped = y_train_new_mapped.ravel()
    y_val_new_mapped = y_val_new_mapped.ravel()
    y_test_new_mapped = y_test_new_mapped.ravel()

    # For old data
    if y_train_old is not None:
        label_mapping_old = {label: i for i, label in enumerate(np.unique(y_train_old))}

        y_train_old_mapped = np.vectorize(label_mapping_old.get)(y_train_old)
        y_val_old_mapped = np.vectorize(label_mapping_old.get)(y_val_old)
        y_test_old_mapped = np.vectorize(label_mapping_old.get)(y_test_old)

        # Converting the arrays to be 1-D
        y_train_old_mapped = y_train_old_mapped.ravel()
        y_val_old_mapped = y_val_old_mapped.ravel()
        y_test_old_mapped = y_test_old_mapped.ravel()

        return (y_train_new_mapped, y_val_new_mapped, y_test_new_mapped, y_train_old_mapped, y_val_old_mapped,
                y_test_old_mapped)

    return y_train_new_mapped, y_val_new_mapped, y_test_new_mapped


def prep_data_for_TL(base_model, X_train_new, X_val_new, X_test_new, race_column):
    # Enhance new data with predicted probabilities from the base model and reintroduce 'Race' for race-specific
    # modeling for both training and validation enhanced sets
    base_model_probs_train = base_model.predict_proba(X_train_new.drop(columns=[race_column]))
    X_train_new_enhanced = np.hstack([X_train_new.drop(columns=[race_column]), base_model_probs_train])
    X_train_new_enhanced = pd.DataFrame(X_train_new_enhanced)
    X_train_new_enhanced[race_column] = X_train_new[race_column].values

    base_model_probs_val = base_model.predict_proba(X_val_new.drop(columns=[race_column]))
    X_val_new_enhanced = np.hstack([X_val_new.drop(columns=[race_column]), base_model_probs_val])
    X_val_new_enhanced = pd.DataFrame(X_val_new_enhanced)
    X_val_new_enhanced[race_column] = X_val_new[race_column].values

    base_model_probs_test = base_model.predict_proba(X_test_new.drop(columns=[race_column]))
    X_test_new_enhanced = np.hstack([X_test_new.drop(columns=[race_column]), base_model_probs_test])
    X_test_new_enhanced = pd.DataFrame(X_test_new_enhanced)
    X_test_new_enhanced[race_column] = X_test_new[race_column].values

    return X_train_new_enhanced, X_val_new_enhanced, X_test_new_enhanced


def random_search_tuning(model, params, race_X_train, race_y_train, cv=5):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=cv,
        verbose=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    random_search.fit(race_X_train, race_y_train.ravel())

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        best_model,
        race_X_train,
        race_y_train.ravel(),
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # Calculate mean and standard deviation of scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot the learning curves
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curves")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid()
    plt.show()

    return best_model, best_params


def evaluate_overfitting(
        train_accuracy, val_accuracy, y_train_true, y_train_pred, y_val_true, y_val_pred, model_name,
        resampling
):
    """
    Evaluate the model for overfitting using training and validation metrics.

    :param resampling: Using data with or without a resampling
    :param model_name:
    :param train_accuracy: Accuracy of the model on the training data.
    :param val_accuracy: Accuracy of the model on the validation data.
    :param y_train_true: True labels for the training data.
    :param y_train_pred: Predicted labels for the training data.
    :param y_val_true: True labels for the validation data.
    :param y_val_pred: Predicted labels for the validation data.
    :return: Dictionary containing overfitting evaluation results.
    """

    # Calculate F1 scores for training and validation sets
    f1_train = f1_score(y_train_true, y_train_pred, average='macro')
    f1_val = f1_score(y_val_true, y_val_pred, average='macro')

    # Calculate precision scores for training and validation sets
    precision_train = precision_score(y_train_true, y_train_pred, average='macro', zero_division=1)
    precision_val = precision_score(y_val_true, y_val_pred, average='macro', zero_division=1)

    # Calculate recall scores for training and validation sets
    recall_train = recall_score(y_train_true, y_train_pred, average='macro', zero_division=1)
    recall_val = recall_score(y_val_true, y_val_pred, average='macro', zero_division=1)

    # Calculate the differences in metrics between training and validation sets
    f1_diff = f1_train - f1_val
    precision_diff = precision_train - precision_val
    recall_diff = recall_train - recall_val
    acc_diff = train_accuracy - val_accuracy

    # Define thresholds for differences that would indicate overfitting
    # These are heuristic values and could be adjusted based on domain knowledge and empirical evidence
    f1_threshold = 0.2
    precision_threshold = 0.2
    recall_threshold = 0.2
    acc_threshold = 0.2

    # Check for overfitting based on multiple criteria
    is_overfitting = (
            f1_diff > f1_threshold or
            precision_diff > precision_threshold or
            recall_diff > recall_threshold or
            acc_diff > acc_threshold
    )

    print(f"Overfitting Evaluation Results for {model_name} {resampling} resampling: {is_overfitting}\n")


def get_model_instance(model_name):
    if model_name == "LogisticRegression":
        return LogisticRegression(multi_class="multinomial", random_state=42)
    elif model_name == "RandomForest":
        return RandomForestClassifier(random_state=42)
    elif model_name == "XGB":
        return XGBClassifier(random_state=42)
    # Add more models as needed
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def interpret_model(model, model_type, X_train, model_name="", race=None):
    """
    Provides an enhanced interpretation of Logistic Regression and RandomForestClassifier models.

    Parameters:
    - model: The trained model instance.
    - X_train: Training dataset used to extract feature names.
    - model_name: A string indicating the name or type of the model for print statements.
    """
    feature_names = X_train.columns.tolist()
    race_info = f" for Race {race}" if race else ""

    if model_name == "LogisticRegression":
        print(f"Interpreting {model_type} Model{race_info}: {model_name}")
        if hasattr(model, 'coef_'):
            # Displaying coefficients and odds ratios
            coefficients = model.coef_[0]
            odds_ratios = np.exp(coefficients)
            print("\nFeature Coefficients and Odds Ratios:")
            for feature, coef, odds_ratio in zip(feature_names, coefficients, odds_ratios):
                print(f"{feature}: Coef={coef:.4f}, Odds Ratio={odds_ratio:.4f}")

        print()

    elif model_name == "RandomForestClassifier":
        print(f"Interpreting {model_type} Model: {model_name}")
        if hasattr(model, 'feature_importances_'):
            # Displaying feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("\nFeature Importances:")
            for f in range(X_train.shape[1]):
                print(f"{f + 1}. feature {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

            # Plotting feature importances for visualization
            plt.figure(figsize=(10, 8))
            plt.title("Feature Importances")
            plt.barh(range(X_train.shape[1]), importances[indices], color='r', align='center')
            plt.yticks(range(X_train.shape[1]), [feature_names[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.ylabel("Feature")
            plt.show()

    else:
        print(f"Model interpretation for {model_name} is not supported.")


def equation(model, feature_names):
    """
        Prints the equation of a trained Logistic Regression model.

        Parameters:
        - model: A trained Logistic Regression model.
        - feature_names: A list of feature names used by the model.
        """

    intercept = model.intercept_[0]
    coefficients = model.coef_[0]

    # Starting the equation with the intercept
    equation = f"log(odds) = {intercept:.4f}"

    # Adding each feature and its coefficient to the equation
    for feature_name, coef in zip(feature_names, coefficients):
        equation += f" + ({coef:.4f} * {feature_name})"

    print(equation, "\n")


def calc_shap_values(model, X, race):
    """
    Computes and visualizes SHAP values for a given model and dataset.

    Parameters:
    - model: The trained model (e.g., a logistic regression model).
    - X: The dataset to compute SHAP values for (e.g., X_train or X_test).
    - race: The race identifier.
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X)

    # Compute SHAP values
    shap_values = explainer(X)

    # Compute mean absolute SHAP values
    mean_abs_shap_values = shap_values.abs.mean(0)

    # Convert mean absolute SHAP values to a list
    # mean_abs_shap_values_list = mean_abs_shap_values.tolist()

    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate bar plot for feature importance
    shap.plots.bar(mean_abs_shap_values, max_display=len(X.columns), show=False)

    # Customize the plot
    plot_title = f"SHAP Feature Importance - Race: {race}"
    ax.set_title(plot_title, fontsize=16, fontweight="bold", y=1.05)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=14, labelpad=10)
    ax.set_ylabel("Feature", fontsize=14)
    ax.tick_params(labelsize=12)

    # Adjust the bottom margin to prevent overlapping of x-axis label and title
    plt.subplots_adjust(bottom=0.15)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Close the figure to prevent additional plots from being created
    plt.close(fig)


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to load the data splits
def load_data_splits(target_variable, pgs_old="without", pgs_new="without", resampling="with"):
    suffix = "AST" if target_variable == "AntisocialTrajectory" else "SUT"
    X_train_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_train_old_{suffix}.csv")
    X_test_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_test_old_{suffix}.csv")
    X_val_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/X_val_old_{suffix}.csv")
    y_train_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_train_old_{suffix}.csv")
    y_test_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_test_old_{suffix}.csv")
    y_val_old = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_new}_PGS/{suffix}_old/y_val_old_{suffix}.csv")

    X_train_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_train_new_{suffix}.csv")
    X_test_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_test_new_{suffix}.csv")
    X_val_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/X_val_new_{suffix}.csv")
    y_train_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_train_new_{suffix}.csv")
    y_test_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_test_new_{suffix}.csv")
    y_val_new = load_data(
        f"../../../preprocessed_data/{resampling}_resampling/{pgs_old}_PGS/{suffix}_new/y_val_new_{suffix}.csv")

    return (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
            y_train_old, y_val_old, y_test_old)


def search_spaces():
    # Define search spaces for each model
    search_parameters = {
        'LogisticRegression': {
            # 'penalty': ['l2', 'elasticnet', None], # Including all types of penalties
            'penalty': ['l2'],  # Including all types of penalties
            'C': np.logspace(-5, 5, 50),  # A wider range and more values for regularization strength
            'solver': ['newton-cg', 'lbfgs'],  # Including all solvers
            'max_iter': list(range(100, 30001, 50)),  # More iterations range with finer steps
            'multi_class': ['multinomial', 'ovr'],  # All strategies for handling multiple classes
            # 'l1_ratio': np.linspace(0, 1, 20),  # Relevant for 'elasticnet' penalty, more granular range
            'fit_intercept': [True, False],  # Whether to include an intercept term or not
            'class_weight': [None, 'balanced'],  # Whether to use balanced class weights or not
            # for other cases
            'warm_start': [True, False],  # Whether to reuse the solution of the previous call as initialization
            'tol': np.logspace(-6, -1, 20),  # Tolerance for stopping criteria
        },
        'RandomForest': {
            'n_estimators': np.arange(300, 1001, 50),
            'max_depth': [None] + list(np.arange(10, 101, 10)),
            'min_samples_split': np.arange(2, 21, 2),
            'min_samples_leaf': np.arange(1, 21, 2),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'GBC': {
            'n_estimators': [10, 50, 100, 200, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
            'max_depth': [3, 5, 10, 20, 50],
            'min_samples_split': range(2, 11, 2),
            'min_samples_leaf': range(1, 11, 2),
            'subsample': [0.5, 0.75, 1.0],
            'max_features': ['sqrt', 'log2', None],
        },
        'XGB': {
            'n_estimators': (800, 1500),
            'learning_rate': (0.01, 0.1),  # Narrowed the range to focus on smaller learning rates
            'max_depth': (3, 6),  # Reduced the upper limit to prevent overly complex trees
            'min_child_weight': (3, 8),  # Adjusted the range based on previous good results
            'subsample': (0.6, 0.9),  # Narrowed the range to focus on higher values
            'colsample_bytree': (0.6, 0.9),  # Narrowed the range to focus on higher values
            'gamma': (0, 0.5),  # Reduced the upper limit to focus on smaller values
            'reg_alpha': (0.1, 1.0),  # Adjusted the range based on previous good results
            'reg_lambda': (10, 30),  # Adjusted the range based on previous good results
            'max_delta_step': (0, 3),  # Reduced the upper limit to focus on smaller values
            'colsample_bylevel': (0.6, 0.9),  # Narrowed the range to focus on higher values
        },
        'CatBoost': {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128, 254],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        },
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],  # Criterion for measuring the quality of a split
            'splitter': ['best', 'random'],  # Strategy used to choose the split at each node
            'max_depth': [None] + list(range(1, 31)),  # Maximum depth of the tree (None means unlimited)
            'min_samples_split': [2, 3, 4, 5] + list(np.linspace(0.01, 0.2, 20)),  # Minimum number of samples
            # required to split an internal node
            'min_samples_leaf': [1, 2, 3, 4, 5] + list(np.linspace(0.01, 0.1, 10)),  # Minimum number of samples
            # required to be at a leaf node
            'max_features': ['sqrt', 'log2', None] + list(np.linspace(0.1, 1.0, 10)),  # Number of features
            # to consider when looking for the best split
            'max_leaf_nodes': [None] + list(range(10, 101, 10)),  # Maximum number of leaf nodes
            'min_impurity_decrease': np.linspace(0, 0.2, 10),  # Threshold for early stopping in tree growth
            'class_weight': [None, 'balanced'],  # Weights associated with classes
        },
        'LightGBM': {
            'num_leaves': np.linspace(20, 400, 10).astype(int),  # More granular range, still need to round to integers
            'learning_rate': np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=10),  # Log-uniform distribution
            'min_child_samples': np.linspace(5, 200, 10).astype(int),  # Broader range, round to integers
            'subsample': np.linspace(0.5, 1.0, 20),  # More granular control
            'colsample_bytree': np.linspace(0.5, 1.0, 10),
            'max_depth': np.linspace(-1, 15, 5).astype(int),  # Including max_depth, round to integers
            'min_split_gain': np.linspace(0.0, 1.0, 10),
            'reg_alpha': np.linspace(0.0, 1.0, 10),
            'reg_lambda': np.linspace(0.0, 1.0, 10),
            # 'max_bin': np.linspace(200, 300, 10)  # Round to integers
        },
    }

    return search_parameters
