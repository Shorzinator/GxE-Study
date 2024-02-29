import statistics

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from Phase_2.model_scripts.model_utils import (load_data_splits, random_search_tuning, search_spaces)


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
                    "final":
                    {
                        1.0: {'n_estimators': 750, 'min_samples_split': 20, 'min_samples_leaf': 11,
                              'max_features': 'log2', 'max_depth': 20, 'bootstrap': True},
                        2.0: {'n_estimators': 950, 'min_samples_split': 10, 'min_samples_leaf': 17,
                              'max_features': None, 'max_depth': 70, 'bootstrap': True},
                        3.0: {'n_estimators': 550, 'min_samples_split': 4, 'min_samples_leaf': 5,
                              'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                        4.0: {'n_estimators': 300, 'min_samples_split': 8, 'min_samples_leaf': 11,
                              'max_features': 'log2', 'max_depth': 30, 'bootstrap': True},
                    }
                },
            "SubstanceUseTrajectory":
                {
                    "final":
                    {
                        1.0: {'n_estimators': 850, 'min_samples_split': 14, 'min_samples_leaf': 13,
                              'max_features': 'log2', 'max_depth': 90, 'bootstrap': True},
                        2.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                              'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                        3.0: {'n_estimators': 650, 'min_samples_split': 18, 'min_samples_leaf': 1,
                              'max_features': 'log2', 'max_depth': 20, 'bootstrap': True},
                        4.0: {'n_estimators': 800, 'min_samples_split': 4, 'min_samples_leaf': 17,
                              'max_features': 'log2', 'max_depth': 30, 'bootstrap': False},
                    }
                }
        }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_final=False, use_cv=False, n_splits=5):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new)

    # Map labels to start from 0
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)

    # Train and evaluate race-specific final models without transfer learning
    for race in sorted(X_train_new[race_column].unique()):
        final_model_params = get_model_params(target_variable, 'final', race)
        final_model = RandomForestClassifier(**final_model_params)

        # Filter data for the current race
        X_train_race = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new[race_column] == race].ravel()
        X_val_race = X_val_new[X_val_new[race_column] == race].drop(columns=[race_column])
        y_val_race = y_val_new_mapped[X_val_new[race_column] == race].ravel()

        if use_cv:
            # Implement cross-validation for the final models
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_val_accuracy = []

            for fold, (train_index, val_index) in enumerate(kf.split(X_train_race, y_train_race)):
                X_train_fold, X_val_fold = X_train_race.iloc[train_index], X_train_race.iloc[val_index]
                y_train_fold, y_val_fold = y_train_race[train_index], y_train_race[val_index]

                if tune_final:
                    final_model, best_params = random_search_tuning(final_model, params['RandomForest'],
                                                                    X_train_fold, y_train_fold.ravel())
                    print(f"Fold {fold}: Best Parameters for base model: {best_params}")
                else:
                    final_model.fit(X_train_fold, y_train_fold.ravel())

                val_accuracy = accuracy_score(y_val_fold, final_model.predict(X_val_fold))
                cv_val_accuracy.append(val_accuracy)

            print(f"Mean CV Accuracy for final model (race {race}): {statistics.mean(cv_val_accuracy)}")

        else:
            final_model.fit(X_train_race, y_train_race)
            final_accuracy = accuracy_score(y_val_race, final_model.predict(X_val_race))
            print(f"Accuracy for final model (race {race}) on validation set: {final_accuracy}")


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(
        target_variable,
        "Race",
        "with",
        "with",
        False,
        True,
        5
    )
