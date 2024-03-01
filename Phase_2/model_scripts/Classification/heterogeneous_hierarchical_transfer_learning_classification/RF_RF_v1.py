import statistics

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from Phase_2.model_scripts.model_utils import (evaluate_overfitting, load_data_splits, random_search_tuning,
                                               search_spaces)


def get_model_params(target_variable, model_type, race=None):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :param race: Optional. Race identifier for race-specific final models.
    :return: Dictionary of model parameters.
    """
    # Example parameters; update these based on tuning results
    params = \
        {
            "AntisocialTrajectory":
                {
                    "base": {
                        'n_estimators': 750, 'max_depth': 80, 'random_state': 42, 'min_samples_split': 20,
                        'min_samples_leaf': 15, 'max_features': 'sqrt', 'bootstrap': True
                    },
                    "final":
                        {
                            1.0: {'n_estimators': 700, 'min_samples_split': 2, 'min_samples_leaf': 3,
                                  'max_features': None, 'max_depth': 50, 'bootstrap': True},
                            2.0: {'n_estimators': 350, 'min_samples_split': 20, 'min_samples_leaf': 11,
                                  'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False},
                            3.0: {'n_estimators': 350, 'min_samples_split': 6, 'min_samples_leaf': 9,
                                  'max_features': None, 'max_depth': 90, 'bootstrap': True},
                            4.0: {'n_estimators': 800, 'min_samples_split': 16, 'min_samples_leaf': 3,
                                  'max_features': 'log2', 'max_depth': 10, 'bootstrap': True},
                        }
                },
            "SubstanceUseTrajectory":
                {
                    "base": {
                        'n_estimators': 550, 'max_depth': 90, 'min_samples_split': 20, 'min_samples_leaf': 7,
                        'max_features': 'sqrt', 'bootstrap': True
                    },
                    "final":
                    {
                        1.0: {'n_estimators': 550, 'min_samples_split': 20, 'min_samples_leaf': 11,
                              'max_features': None, 'max_depth': None, 'bootstrap': True},
                        2.0: {'n_estimators': 800, 'min_samples_split': 16, 'min_samples_leaf': 3,
                              'max_features': 'log2', 'max_depth': 10, 'bootstrap': True},
                        3.0: {'n_estimators': 350, 'min_samples_split': 4, 'min_samples_leaf': 1,
                              'max_features': None, 'max_depth': 90, 'bootstrap': False},
                        4.0: {'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                              'max_features': 'log2', 'max_depth': 60, 'bootstrap': True},
                    }
                }
        }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_base=False, tune_final=False,
         use_cv=True, n_splits=10, resampling="without"):

    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling)

    # Map labels to start from 0
    label_mapping_old = {label: i for i, label in enumerate(np.unique(y_train_old))}
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_old_mapped = np.vectorize(label_mapping_old.get)(y_train_old)
    y_val_old_mapped = np.vectorize(label_mapping_old.get)(y_test_old)
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)

    # Train a base model on old data with dynamic parameters
    base_model_params = get_model_params(target_variable, 'base')
    base_model = RandomForestClassifier(**base_model_params)

    if use_cv:
        # Perform cross-validation on training data
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_val_acc = []
        fold_train_acc = []
        for fold, (train_index, val_index) in enumerate(kf.split(X_train_old, y_train_old_mapped)):
            X_train_fold, X_val_fold = X_train_old.iloc[train_index], X_train_old.iloc[val_index]
            y_train_fold, y_val_fold = y_train_old_mapped[train_index], y_train_old_mapped[val_index]

            # If hyperparameter tuning is enabled
            if tune_base:
                base_model, best_params = random_search_tuning(base_model, params['RandomForest'],
                                                               X_train_fold, y_train_fold.ravel())
                print(f"Fold {fold}: Best Parameters for base model: {best_params}")
            else:
                base_model.fit(X_train_fold, y_train_fold.ravel())

            # Evaluate on the validation fold
            val_accuracy = accuracy_score(y_val_fold.ravel(), base_model.predict(X_val_fold))
            fold_val_acc.append(val_accuracy)

            train_accuracy = accuracy_score(y_train_fold.ravel(), base_model.predict(X_train_fold))
            fold_train_acc.append(train_accuracy)

        # Evaluate for overfitting
        base_overfitting_results = evaluate_overfitting(
            train_accuracy=statistics.mean(fold_train_acc),
            val_accuracy=statistics.mean(fold_val_acc),
            y_train_true=y_train_old_mapped,
            y_train_pred=base_model.predict(X_train_old),
            y_val_true=y_val_old_mapped,
            y_val_pred=base_model.predict(X_val_old)
        )

        print(f"Base Model Overfitting Evaluation Results: {base_overfitting_results}")

        print("Mean Val Accuracy for base model:", statistics.mean(fold_val_acc))
        print("Mean Train Accuracy for base model:", statistics.mean(fold_train_acc), "\n")

    else:
        # Train the base model on full training data without cross-validation
        if tune_base:
            base_model, best_params = random_search_tuning(base_model, params['RandomForest'],
                                                           X_train_old, y_train_old_mapped.ravel())
            print(f"Best Parameters for base model: {best_params}")
        else:
            base_model.fit(X_train_old, y_train_old_mapped.ravel())

            # Evaluate for overfitting
            base_overfitting_results = evaluate_overfitting(
                train_accuracy=accuracy_score(y_train_old_mapped.ravel(), base_model.predict(X_train_old)),
                val_accuracy=accuracy_score(y_val_old_mapped.ravel(), base_model.predict(X_val_old)),
                y_train_true=y_train_old_mapped,
                y_train_pred=base_model.predict(X_train_old),
                y_val_true=y_val_old_mapped,
                y_val_pred=base_model.predict(X_val_old)
            )

            print(f"Base Model Overfitting Evaluation Results: {base_overfitting_results}")

        base_model_val_accuracy = accuracy_score(y_val_old_mapped.ravel(), base_model.predict(X_val_old))
        print(f"Val Accuracy for base model: {base_model_val_accuracy}")

        base_model_train_accuracy = accuracy_score(y_train_old_mapped.ravel(), base_model.predict(X_train_old))
        print(f"Train Accuracy for base model: {base_model_train_accuracy}", "\n")

    # Enhancing new data with predicted probabilities from the base model for both training and validation sets
    base_model_probs_new_train = base_model.predict_proba(X_train_new.drop(columns=[race_column]))
    X_train_new_enhanced = np.hstack([X_train_new.drop(columns=[race_column]), base_model_probs_new_train])
    base_model_probs_new_val = base_model.predict_proba(X_val_new.drop(columns=[race_column]))
    X_val_new_enhanced = np.hstack([X_val_new.drop(columns=[race_column]), base_model_probs_new_val])

    # Reintroduce 'Race' for race-specific modeling for both training and validation enhanced sets
    X_train_new_enhanced = pd.DataFrame(X_train_new_enhanced)
    X_train_new_enhanced[race_column] = X_train_new[race_column].values
    X_val_new_enhanced = pd.DataFrame(X_val_new_enhanced)  # Enhanced validation set
    X_val_new_enhanced[race_column] = X_val_new[race_column].values  # Enhanced validation set

    # Train and evaluate race-specific interim models on enhanced training and validation sets
    for race in sorted(X_train_new[race_column].unique()):
        final_model_params = get_model_params(target_variable, 'final', race)
        final_model = RandomForestClassifier(**final_model_params)

        X_train_race = X_train_new_enhanced[X_train_new_enhanced[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new_enhanced[race_column] == race].ravel()
        X_val_race = X_val_new_enhanced[X_val_new_enhanced[race_column] == race].drop(columns=[race_column])
        y_val_race = y_val_new_mapped[X_val_new_enhanced[race_column] == race].ravel()

        # Implement cross-validation for the final models
        if use_cv:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_val_accuracy = []
            cv_train_accuracy = []

            for train_index, val_index in kf.split(X_train_race, y_train_race):
                X_train_fold, X_val_fold = X_train_race.iloc[train_index], X_train_race.iloc[val_index]
                y_train_fold, y_val_fold = y_train_race[train_index], y_train_race[val_index]

                if tune_final:
                    final_model, best_params = random_search_tuning(final_model, params['LogisticRegression'],
                                                                    X_train_race,
                                                                    y_train_race)
                    print(f"Best Parameters for final model (race {race}): {best_params}")
                else:
                    final_model.fit(X_train_race, y_train_race)

                val_accuracy = accuracy_score(y_val_fold, final_model.predict(X_val_fold))
                cv_val_accuracy.append(val_accuracy)

                train_accuracy = accuracy_score(y_train_fold.ravel(), final_model.predict(X_train_fold))
                cv_train_accuracy.append(train_accuracy)

            print(f"Mean Val Accuracy for final model (race {race}): {statistics.mean(cv_val_accuracy)}")
            print(f"Mean Train Accuracy for final model (race {race}): {statistics.mean(cv_train_accuracy)}")

        else:
            if tune_final:
                final_model, best_params = random_search_tuning(final_model, params['LogisticRegression'], X_train_race,
                                                                y_train_race)
                print(f"Best Parameters for final model (race {race}): {best_params}")
            else:
                final_model.fit(X_train_race, y_train_race)
                final_accuracy = accuracy_score(y_val_race, final_model.predict(X_val_race))
                print(f"Accuracy for final model (race {race}) on validation set: {final_accuracy}")

        # After training the final model for each race
        y_train_race_pred = final_model.predict(X_train_race)
        y_val_race_pred = final_model.predict(X_val_race)
        final_overfitting_results = evaluate_overfitting(
            train_accuracy=accuracy_score(y_train_race, y_train_race_pred),
            val_accuracy=accuracy_score(y_val_race, y_val_race_pred),
            y_train_true=y_train_race,
            y_train_pred=y_train_race_pred,
            y_val_true=y_val_race,
            y_val_pred=y_val_race_pred
        )
        print(f"Final Model Overfitting Evaluation Results for Race {race}: {final_overfitting_results}\n")


if __name__ == "__main__":
    target_variable = "SubstanceUseTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "Race",
         "with",
         "with",
         tune_base=False,
         tune_final=False,
         use_cv=True,
         resampling="without",
         n_splits=5)
