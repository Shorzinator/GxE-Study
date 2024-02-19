import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from Phase_2.model_scripts.model_utils import (load_data_splits, random_search_tuning,
                                               random_search_tuning_intermediate, search_spaces)


def main(
        target_variable, race_column="Race", pgs_old="with", pgs_new="with",
        tune_base=False, tune_final=False, use_cv=True, n_splits=5
        ):
    params = search_spaces()

    # Load data splits
    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    label_mapping_old = {label: i for i, label in enumerate(np.unique(y_train_old))}
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_old_mapped = np.vectorize(label_mapping_old.get)(y_train_old)
    y_test_old_mapped = np.vectorize(label_mapping_old.get)(y_test_old)
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_test_new_mapped = np.vectorize(label_mapping_new.get)(y_test_new)

    # Train a base model on old data
    if use_cv:
        # Perform cross-validation on training data
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kf.split(X_train_old, y_train_old_mapped)):
            X_train_fold, X_val_fold = X_train_old[train_index], X_train_old[val_index]
            y_train_fold, y_val_fold = y_train_old_mapped[train_index], y_train_old_mapped[val_index]

            # Train the base model with cross-validation fold
            base_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            if tune_base:
                # Perform hyperparameter tuning for the base model
                base_model, best_params = random_search_tuning(base_model, params['LogisticRegression'],
                                                               X_train_fold, y_train_fold.ravel())
                print(f"Fold {fold}: Best Parameters for base model: {best_params}")
            else:
                base_model.fit(X_train_fold, y_train_fold.ravel())

            # Evaluate on the validation fold
            val_accuracy = accuracy_score(y_val_fold.ravel(), base_model.predict(X_val_fold))
            print(f"Fold {fold}: Validation Accuracy for base model: {val_accuracy}")
    else:
        # Train the base model on full training data without cross-validation
        base_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
        if tune_base:
            base_model, best_params = random_search_tuning(base_model, params['RandomForest'],
                                                           X_train_old, y_train_old_mapped.ravel())
            print(f"Best Parameters for base model: {best_params}")
        else:
            base_model.fit(X_train_old, y_train_old_mapped.ravel())
        base_model_accuracy = accuracy_score(y_test_old_mapped.ravel(), base_model.predict(X_test_old))
        print(f"Accuracy for base model: {base_model_accuracy}")

    # Enhance new data with predicted probabilities from the base model
    base_model_probs_new = base_model.predict_proba(X_train_new.drop(columns=[race_column]))
    X_train_new_enhanced = np.hstack([X_train_new.drop(columns=[race_column]), base_model_probs_new])
    base_model_probs_test = base_model.predict_proba(X_test_new.drop(columns=[race_column]))
    X_test_new_enhanced = np.hstack([X_test_new.drop(columns=[race_column]), base_model_probs_test])

    # Reintroduce 'Race' for race-specific modeling
    X_train_new_enhanced = pd.DataFrame(X_train_new_enhanced)
    X_train_new_enhanced[race_column] = X_train_new[race_column].values
    X_test_new_enhanced = pd.DataFrame(X_test_new_enhanced)
    X_test_new_enhanced[race_column] = X_test_new[race_column].values

    # Train and evaluate race-specific interim models
    for race in sorted(X_train_new[race_column].unique()):
        final_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        X_train_race = X_train_new_enhanced[X_train_new_enhanced[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new_enhanced[race_column] == race].ravel()
        X_test_race = X_test_new_enhanced[X_test_new_enhanced[race_column] == race].drop(columns=[race_column])
        y_test_race = y_test_new_mapped[X_test_new_enhanced[race_column] == race].ravel()

        if tune_final:
            final_model, best_params = random_search_tuning_intermediate(final_model, 'LogisticRegression', params,
                                                                         X_train_race, y_train_race)
            print(f"Best Parameters for interim model (race {race}): {best_params}")
        else:
            final_model.fit(X_train_race, y_train_race)

        interim_accuracy = accuracy_score(y_test_race, final_model.predict(X_test_race))
        print(f"Accuracy for interim model (race {race}): {interim_accuracy}")


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # or "SubstanceUseTrajectory"
    main(target_variable, "Race", "with", "with")
