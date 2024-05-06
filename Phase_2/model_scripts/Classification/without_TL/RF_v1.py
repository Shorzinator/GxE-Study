from sklearn.ensemble import RandomForestClassifier

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance, load_data_splits,
                                               prep_data_for_race_model, train_and_evaluate)


def get_model_params(target_variable, model_type, race=None, resampling="with"):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param resampling: Training used data with or without resampling
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :param race: Optional. Race identifier for race-specific final models.
    :return: Dictionary of model parameters.
    """
    # Example parameters; update these based on tuning results
    if resampling == "with":
        params = \
            {
                "AntisocialTrajectory":
                    {
                        "final":
                        {
                            1.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                  'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                            2.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                  'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                            3.0: {'n_estimators': 550, 'min_samples_split': 4, 'min_samples_leaf': 5,
                                  'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                            4.0: {'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                  'max_features': 'log2', 'max_depth': 60, 'bootstrap': True},
                        }
                    },
            }
    else:
        params = \
            {
                "AntisocialTrajectory":
                    {
                        "final":
                            {
                                1.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                      'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                                2.0: {
                                    'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                3.0: {
                                    'n_estimators': 550, 'min_samples_split': 4, 'min_samples_leaf': 5,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                4.0: {
                                    'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 60, 'bootstrap': True
                                },
                            }
                    },
            }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(target_variable, race_column="Race", tune_final=False, pgs_old="without", pgs_new="without", cv=10, resampling="with",
         final_model_name="RandomForest", final_model_type="final"):

    print(f"Running {final_model_name}_v2 model predicting {target_variable} {resampling} resampling and {pgs_new} PGS:\n")

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling=resampling)

    # Map labels to start from 0
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = get_mapped_data(y_train_new, y_val_new, y_test_new)

    # Train and evaluate race-specific models
    for race in sorted(X_train_new[race_column].unique()):
        # Defining final_model based on the current race in iteration and its respective parameters
        if not tune_final:
            final_model = get_model_instance(final_model_name)
            # final_model = RandomForestClassifier(**get_model_params(target_variable, "final", race, resampling))
        else:
            final_model = get_model_instance(final_model_name)

        X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race = prep_data_for_race_model(
            X_train_new,
            y_train_new_mapped,
            X_val_new,
            y_val_new_mapped,
            X_test_new,
            y_test_new_mapped,
            race, race_column)

        train_and_evaluate(final_model, X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race,
                           final_model_name, tune_final, race, model_type=final_model_type, cv=cv,
                           resampling=resampling, outcome=target_variable)


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(
        target_variable,
        "Race",
        False,
        "without",
        "without",
        5,
        "without",
        "RandomForest"
    )
