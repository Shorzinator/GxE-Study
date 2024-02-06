import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
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

    # Indicate prediction target on the graph, below the title
    ax.text(0.5, 0.95, f"", transform=ax.transAxes, ha='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.savefig(f"../../../results/modeling/comparing_models_{target_variable.lower()}_{metric.lower()}.png")
    plt.show()


def evaluate_models(target_variable):
    # Load data for the specified target variable
    _, X_train, _, X_test, _, y_train, _, y_test = load_data_splits(target_variable=target_variable)

    # Add a constant term for intercept for statsmodels
    X_train_glm = sm.add_constant(X_train)
    X_test_glm = sm.add_constant(X_test)

    # Remap target variables to start from 0
    y_train = y_train - 1
    y_test = y_test - 1

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Define the models to be evaluated
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "Ridge Classifier": RidgeClassifier(),
        "XGBClassifier": XGBClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, n_estimators=100, min_child_weight=0,
                                       max_depth=10, max_delta_step=5, learning_rate=0.2, gamma=0, colsample_bytree=0.5,
                                       colsample_bylevel=0.5, eval_metric='mlogloss'),
        "GLM Logistic": sm.MNLogit(y_train, X_train_glm)
    }

    # Initialize dictionary to hold evaluation results
    results = {
        "Model": [],
        "Accuracy": [],
        "Log Loss": []
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
            results["Model"].append(name)
            results["Accuracy"].append(accuracy)

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

            # Store results
            results["Model"].append(name)
            results["Accuracy"].append(accuracy)
            results["Log Loss"].append(logloss)

    # Visualization
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.bar(results["Model"], results["Accuracy"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Log Loss', color=color)
    ax2.plot(results["Model"], results["Log Loss"], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig("comparing_simpler_models_with_xgb_AST.png")
    plt.show()

    # Call the new plotting functions
    plot_results(results, "Accuracy", target_variable)
    plot_results(results, "Log Loss", target_variable)


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Evaluate models for the first target
    evaluate_models(target_1)
