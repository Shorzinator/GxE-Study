import os
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from Phase_2.model_scripts.model_utils import load_data_splits, \
    random_search_tuning, random_search_tuning_intermediate, search_spaces

params = search_spaces()


def main(target_variable, race_column, tune_base, tune_interim):

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

    # Step 1: Train base model on old data
    # for AST, with and without pgs
    base_model = XGBClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, n_estimators=1000, min_child_weight=0,
                               max_depth=10, max_delta_step=5, learning_rate=0.2, gamma=0, colsample_bytree=0.5,
                               colsample_bylevel=0.5, eval_metric='mlogloss')

    # For SUT, without pgs
    # base_model = XGBClassifier(subsample=0.5, reg_lambda=0, reg_alpha=0, n_estimators=1000, min_child_weight=0,
    #                            max_depth=10, max_delta_step=5, learning_rate=0.01, gamma=0, colsample_bytree=0.5,
    #                            colsample_bylevel=1.0, eval_metric='mlogloss')

    base_model_type = "XGBoost"

    if tune_base:
        base_model, best_params = random_search_tuning(base_model, base_model_type, params, X_train_old,
                                                       y_train_old_mapped)
        print(f"Best Parameters for base model: {best_params}")

    else:
        base_model.fit(X_train_old, y_train_old_mapped)

    prediction = base_model.predict(X_test_old)
    base_model_accuracy = accuracy_score(y_test_old_mapped, prediction)
    print(f"Accuracy for base model: {base_model_accuracy}")

    if tune_interim:
        # best_model_2, best_params = random_search_tuning_intermediate(deepcopy(base_model), base_model_type, params,
        #                                                               X_train_new.drop(columns=[race_column]),
        #                                                               y_train_new_mapped)
        best_model_2, best_params = random_search_tuning_intermediate(deepcopy(base_model), base_model_type, params,
                                                                      X_train_new,
                                                                      y_train_new_mapped)
        print(f"Best Parameters for intermediate model: {best_params}")
    else:
        best_model_1 = XGBClassifier(eval_metric="mlogloss")

        # For SUT, without pgs, without Race
        # best_model_2 = XGBClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, n_estimators=1000, min_child_weight=0,
        #                              max_depth=10, max_delta_step=5, learning_rate=0.01, gamma=0, colsample_bytree=0.5,
        #                              colsample_bylevel=0.5, eval_metric='mlogloss')

        # For AST, with pgs, with Race
        best_model_2 = XGBClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, n_estimators=1000, min_child_weight=0,
                                     max_depth=10, max_delta_step=5, learning_rate=0.01, gamma=0, colsample_bytree=0.5,
                                     colsample_bylevel=0.5, eval_metric='mlogloss')
        best_model_3 = deepcopy(base_model)

        best_model_1.fit(X_train_new.drop(columns=[race_column]), y_train_new_mapped)
        best_model_2.fit(X_train_new.drop(columns=[race_column]), y_train_new_mapped)
        best_model_3.fit(X_train_new.drop(columns=[race_column]), y_train_new_mapped)

    # prediction_1 = best_model_1.predict(X_test_new.drop(columns=[race_column]))
    # intermediate_accuracy_1 = accuracy_score(y_test_new_mapped, prediction_1)
    # print(f"Accuracy for intermediate model - default XGB - (excluding race): {intermediate_accuracy_1}")

    prediction_2 = best_model_2.predict(X_test_new)
    intermediate_accuracy_2 = accuracy_score(y_test_new_mapped, prediction_2)
    print(f"Accuracy for intermediate model - independently tuned XGB - (excluding race): {intermediate_accuracy_2}")

    # prediction_3 = best_model_3.predict(X_test_new.drop(columns=[race_column]))
    # intermediate_accuracy_3 = accuracy_score(y_test_new_mapped, prediction_3)
    # print(f"Accuracy for intermediate model - deepcopy XGB -(excluding race): {intermediate_accuracy_3}")


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    tune_base = True
    tune_interim = True

    race_column = "Race"

    # Run the main function
    main(target_1, race_column=race_column, tune_base=tune_base, tune_interim=tune_interim)
