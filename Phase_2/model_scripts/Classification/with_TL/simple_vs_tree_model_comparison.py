import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from Phase_2.model_scripts.model_utils import load_data_splits
import seaborn as sns  # For better color palettes and themes


# Visualization
def plot_results(results, metric, target_variable):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 7))

    valid_metrics = [m for m in results[metric] if m is not None]
    valid_models = [results["Model"][i] for i, m in enumerate(results[metric]) if m is not None]

    colors = sns.color_palette('bright', n_colors=len(valid_models))
    bars = ax.bar(valid_models, valid_metrics, color=colors)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    title = f'Comparison for {target_variable} by {metric}'
    ax.set_title(title, fontsize=16, color='black')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(range(len(valid_models)))
    ax.set_xticklabels(valid_models, rotation=45, ha='right')

    # Adjust the subplot params to give some more room for the title
    plt.subplots_adjust(top=0.85)

    for bar, color in zip(bars, colors):
        height = bar.get_height()
        label_position = height + (0.01 * height)  # Slightly higher than the bar top
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, label_position),
                    xytext=(0, 3),  # 3-point vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Plot the baseline on the graph
    baseline_metric = results[metric][-1]  # Assuming baseline is the last one added
    ax.axhline(y=baseline_metric, color='r', linestyle='--', label='Baseline')
    ax.legend()
    # Indicate prediction target on the graph, below the title
    ax.text(0.5, 0.95, f"", transform=ax.transAxes, ha='center', fontsize=12, color='black')

    plt.tight_layout()
    # plt.savefig(f"../../../results/modeling/comparing_models_{target_variable.lower()}_{metric.lower()}.png")
    plt.show()


# New function to plot precision, recall, and F1 score
def plot_classification_report(results, target_variable):
    sns.set_theme(style="whitegrid")
    metrics = ["Precision", "Recall", "F1 Score"]
    fig, ax = plt.subplots(1, len(metrics), figsize=(18, 5))

    for i, metric in enumerate(metrics):
        valid_metrics = [m for m in results[metric] if m is not None]
        valid_models = [results["Model"][i] for i, m in enumerate(results[metric]) if m is not None]

        colors = sns.color_palette('viridis', n_colors=len(valid_models))
        bars = ax[i].bar(valid_models, valid_metrics, color=colors)

        ax[i].set_xlabel('Model', fontsize=14)
        ax[i].set_ylabel(metric, fontsize=14)
        ax[i].set_title(f'{metric} for {target_variable}', fontsize=16, color='black')
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        ax[i].set_xticks(range(len(valid_models)))
        ax[i].set_xticklabels(valid_models, rotation=45, ha='right')

        for bar, color in zip(bars, colors):
            height = bar.get_height()
            label_position = height + (0.005 * height)  # Slightly higher than the bar top
            ax[i].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, label_position),
                           xytext=(0, 3),  # 3-point vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    # Plot the baseline on each subplot
    for i, metric in enumerate(metrics):
        baseline_metric = results[metric][-1]  # Assuming baseline is the last one added
        ax[i].axhline(y=baseline_metric, color='r', linestyle='--', label='Baseline')
        ax[i].legend()

    plt.tight_layout()
    # plt.savefig(f"../../../results/modeling/classification_report_{target_variable.lower()}.png")
    plt.show()


# Function to compute baseline metrics
def compute_baseline_metrics(X_train, y_train, y_test):
    # Most Frequent baseline (always predicts the most common class)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    baseline_preds = dummy.predict(y_test)

    # Ensure the predictions are integers (class labels) if they're not
    baseline_preds = baseline_preds.astype(int)

    baseline_metrics = {
        'Accuracy': accuracy_score(y_test, baseline_preds),
        'Log Loss': log_loss(y_test, dummy.predict_proba(y_test)),  # This assumes y_test is not continuous
        'Precision': precision_score(y_test, baseline_preds, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, baseline_preds, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, baseline_preds, average='weighted', zero_division=0)
    }
    return baseline_metrics


def evaluate_models(target_variable):
    # Load data for the specified target variable
    _, X_train, _, X_test, _, y_train, _, y_test = load_data_splits(target_variable=target_variable, pgs_old="without",
                                                                    pgs_new="without")

    # Add a constant term for intercept for statsmodels
    X_train_glm = sm.add_constant(X_train)
    X_test_glm = sm.add_constant(X_test)

    # Remap target variables to start from 0
    y_train = y_train - 1
    y_test = y_test - 1

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Binarize the output labels for multiclass ROC AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

    # Define the models to be evaluated
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "XGBClassifier": XGBClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, n_estimators=100, min_child_weight=0,
                                       max_depth=10, max_delta_step=5, learning_rate=0.2, gamma=0, colsample_bytree=0.5,
                                       colsample_bylevel=0.5, eval_metric='mlogloss'),
        "GLM Logistic": sm.MNLogit(y_train, X_train_glm),
        "Ridge Classifier": RidgeClassifier(),
    }

    # Initialize dictionary to hold evaluation results
    results = {
        "Model": [],
        "Accuracy": [],
        "Log Loss": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
    }

    # Evaluate each model
    for name, model in models.items():
        if name != "GLM Logistic":
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            preds = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average='weighted')
            recall = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')

            results["Model"].append(name)
            results["Accuracy"].append(accuracy)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["F1 Score"].append(f1)

            # Check if the model has 'predict_proba' method for log loss calculation
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_test)
                logloss = log_loss(y_test, probas)
                results["Log Loss"].append(logloss)

            else:
                # Append None or a placeholder if log loss can't be computed
                results["Log Loss"].append(None)
        else:
            # For multinomial logistic regression, use MNLogit
            glm_result = model.fit()

            # Predictions for multinomial logistic regression
            preds = glm_result.predict(X_test_glm).idxmax(axis=1)

            # Probabilities for calculating log loss
            probas = glm_result.predict(X_test_glm).values

            # Calculate metrics
            accuracy = accuracy_score(y_test, preds)
            logloss = log_loss(y_test, probas)
            precision = precision_score(y_test, preds, average='weighted')
            recall = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')

            results["Model"].append(name)
            results["Accuracy"].append(accuracy)
            results["Log Loss"].append(logloss)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["F1 Score"].append(f1)

    # Compute and add baseline metrics after evaluating other models
    baseline_metrics = compute_baseline_metrics(X_train, y_train, y_test)

    results["Model"].append('Baseline')
    for metric in ['Accuracy', 'Log Loss', 'Precision', 'Recall', 'F1 Score']:
        metric_value = baseline_metrics.get(metric, np.nan)  # Use np.nan as default if metric is not found
        results[metric].append(metric_value)

    # Visualization
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('Model')
    # ax1.set_ylabel('Accuracy', color=color)
    # ax1.bar(results["Model"], results["Accuracy"], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()
    # color = 'tab:blue'
    # ax2.set_ylabel('Log Loss', color=color)
    # ax2.plot(results["Model"], results["Log Loss"], color=color, marker='o')
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()
    # plt.xticks(rotation=45)
    # plt.savefig("comparing_simpler_models_with_xgb_AST.png")
    # plt.show()

    # Call the new plotting functions
    plot_results(results, "Accuracy", target_variable)
    plot_results(results, "Log Loss", target_variable)
    plot_classification_report(results, target_variable)


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Evaluate models for the first target
    evaluate_models(target_2)
