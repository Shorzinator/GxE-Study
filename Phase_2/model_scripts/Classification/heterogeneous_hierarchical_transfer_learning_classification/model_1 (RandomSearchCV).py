import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Phase_2.model_scripts.model_utils import evaluate_model, get_transfer_strategy, load_data_splits, \
    random_search_tuning, search_spaces

params = search_spaces()


# Updated function to train and evaluate race-specific models with real-time hyperparameter tuning
def train_and_evaluate_race_specific_models(
        X_train_new, y_train_new, X_test_new, y_test_new, race_column,
        intermediate_model, base_model_type, feature_based_transfer=True
):
    race_models = {}
    race_best_params = {}
    performance_metrics = {}

    for race in X_train_new[race_column].unique():

        if race == 5:  # Skip the 'Other' category
            continue

        # Preparing race-specific data
        race_X_train = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        race_y_train = y_train_new[X_train_new[race_column] == race]
        race_X_test = X_test_new[X_test_new[race_column] == race].drop(columns=[race_column])
        race_y_test = y_test_new[X_test_new[race_column] == race]

        # Map labels to start from 0 if necessary
        unique_labels = np.unique(race_y_train)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        inv_label_mapping = {i: label for label, i in label_mapping.items()}  # For inverse mapping after prediction

        race_y_train_mapped = np.vectorize(label_mapping.get)(race_y_train)

        # Define models to be used
        models = {
            # 'RandomForest': RandomForestClassifier(),
            # 'GBC': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(eval_metric='logloss'),
            # 'CatBoost': CatBoostClassifier(verbose=False, one_hot_max_size=4, loss_function="MultiClass"),
        }

        # Iterate through models and search spaces
        for model_name, model in models.items():
            print(f"Training {model_name} for race {race}\n")

            # Initialize transformed data with original data
            race_X_train_transformed = race_X_train.copy()
            race_X_test_transformed = race_X_test.copy()

            # Dynamically adjust to compatible transfer learning strategy
            transfer_strategy = get_transfer_strategy(base_model_type, model_name)

            if transfer_strategy == "direct":  # Direct parameter transfer
                if model_name == 'XGBoost':
                    model.load_model(intermediate_model.get_booster())  # Assuming intermediate_model is XGBoost
                # Add other models if applicable

            elif feature_based_transfer:  # Conditional feature-based transfer

                intermediate_predictions = intermediate_model.predict_proba(race_X_train)
                race_X_train_transformed = np.hstack([race_X_train, intermediate_predictions])
                intermediate_predictions_test = intermediate_model.predict_proba(race_X_test)
                race_X_test_transformed = np.hstack([race_X_test, intermediate_predictions_test])

                # Option to toggle tuning on or off
            tuning_required = True  # Set to False to use predefined parameters without tuning

            if tuning_required:
                # Perform hyperparameter tuning
                best_model, best_params = random_search_tuning(model, model_name, params,
                                                               race_X_train_transformed, race_y_train_mapped)
                print(f'Best Parameters for {model_name}: {best_params}')
            else:
                # Directly use the model with predefined or transferred parameters
                best_model = model  # Use the model with parameters loaded or set beforehand

                # Fit the best model or the model with predefined parameters
                best_model.fit(race_X_train_transformed, race_y_train_mapped)

            # Predict using the fitted model
            predictions = best_model.predict(race_X_test_transformed)

            # Map predictions back to original labels
            predictions = np.vectorize(inv_label_mapping.get)(predictions)
            performance = accuracy_score(race_y_test, predictions)

            # Evaluate the best model
            print(f'Best Model for race {race}: {model_name}')
            print(f'Performance: {performance}')

            # Store the best model, parameters, and performance
            race_models[(race, model_name)] = best_model
            if tuning_required:
                race_best_params[(race, model_name)] = best_params
            performance_metrics[(race, model_name)] = performance

    return race_models, race_best_params, performance_metrics


def main(target_variable, race_column="Race", tuning=True):
    if target_variable == "AntisocialTrajectory":
        tag = "AST"
    else:
        tag = "SUT"

    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable))

    # Ensure directories exist
    os.makedirs("../../../results/models/classification/HetHieTL", exist_ok=True)
    os.makedirs("../../../results/metrics/classification/HomHieTL", exist_ok=True)

    # Map labels to start from 0 for both old and new datasets
    unique_labels_old = np.unique(y_train_old)
    unique_labels_new = np.unique(y_train_new)
    label_mapping_old = {label: i for i, label in enumerate(unique_labels_old)}
    label_mapping_new = {label: i for i, label in enumerate(unique_labels_new)}

    y_train_old_mapped = np.vectorize(label_mapping_old.get)(y_train_old)
    y_test_old_mapped = np.vectorize(label_mapping_old.get)(y_test_old)
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_test_new_mapped = np.vectorize(label_mapping_new.get)(y_test_new)

    # Flatten y_train and y_test
    # y_train_new, y_test_new, y_train_old, y_test_old = [
    #     y.values.ravel() for y in (y_train_new, y_test_new, y_train_old, y_test_old_mapped)
    # ]

    # Step 1: Train base model on old data
    # base_model = RandomForestClassifier(n_estimators=650, max_depth=60, bootstrap=False, max_features='sqrt',
    #                                     min_samples_split=8, random_state=42, min_samples_leaf=1)
    base_model = XGBClassifier(eval_metric="mlogloss")
    base_model_type = "XGBoost"

    if tuning:
        base_model, best_params = random_search_tuning(base_model, base_model_type, params, X_train_old,
                                                       y_train_old_mapped)
        print(f"Best Parameters for base model: {best_params}")

    else:
        base_model.fit(X_train_old, y_train_old_mapped)

    prediction = base_model.predict(X_test_old)
    base_model_accuracy = accuracy_score(y_test_old_mapped, prediction)
    print(f"Accuracy for base model: {base_model_accuracy}")

    # Save base model and its metrics
    pickle.dump(base_model, open(f"../../../results/models/classification/HetHieTL/{tag}/BaseModel - XGB/base_model.pkl"
                                 , "wb"))
    pd.DataFrame(
        {
            'Model': ['Base Model'],
            'Accuracy': [base_model_accuracy],
            'Best Params': [str(best_params)]
        }
    ).to_csv(f"../../../results/metrics/classification/HetHieTL/{tag}/BaseModel - XGB/base_model_metrics.csv",
             index=False)

    # Step 2: Retrain base model on new data (excluding race) as an intermediate model
    intermediate_model = random_search_tuning(deepcopy(base_model), base_model_type, params, X_train_new,
                                              y_train_new_mapped)
    intermediate_model.fit(X_train_new.drop(columns=[race_column]), y_train_new_mapped)
    prediction = intermediate_model.predict(X_test_new.drop(columns=[race_column]))
    intermediate_accuracy = accuracy_score(y_test_new_mapped, prediction)

    print(f"Accuracy for intermediate model (excluding race): {intermediate_accuracy}")

    # Save intermediate model and its metrics
    pickle.dump(intermediate_model, open(f"../../../results/models/classification/HetHieTL/{tag}/intermediate_model.pkl"
                                         , "wb"))
    pd.DataFrame(
        {
            'Model': ['Intermediate Model'],
            'Accuracy': [intermediate_accuracy]
        }
    ).to_csv(f"../../../results/metrics/classification/HetHieTL/{tag}/BaseModel - XGB/"
             f"intermediate_model_metrics.csv", index=False)

    # Step 3: Train and evaluate race-specific models
    final_models, race_best_params, performance_metrics = train_and_evaluate_race_specific_models(
        X_train_new, y_train_new_mapped, X_test_new, y_test_new_mapped, race_column, intermediate_model,
        base_model_type,
        feature_based_transfer=True)

    # Save race-specific models, their best parameters, and performance metrics
    for (race, model_name), model in final_models.items():
        pickle.dump(model, open(
            f"../../../results/models/classification/HetHieTL/{tag}/{model_name}/{model_name}_race_{race}.pkl", "wb"))
        pd.DataFrame({
            'Race': [race],
            'Model': [model_name],
            'Accuracy': [performance_metrics[(race, model_name)]],
            'Best Params': [str(race_best_params[(race, model_name)])]
        }).to_csv(f"{model_name}_race_{race}_metrics.csv", index=False)

    print(f"Performance metrics for final models: {performance_metrics}")
    print(f"Best Parameters for final models: {race_best_params}")


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    # Run the main function
    main(target_1)
