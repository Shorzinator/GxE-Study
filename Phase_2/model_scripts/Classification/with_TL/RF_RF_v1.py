import os

from sklearn.ensemble import RandomForestClassifier

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance, load_data_splits, prep_data_for_TL,
                                               search_spaces,
                                               prep_data_for_race_model, train_and_evaluate_model)


# Get the base name of the current script and strip the .py extension to use in the filename
script_name = os.path.basename(__file__).replace('.py', '')


def get_model_params(target_variable, model_type, race=None, resampling="without"):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param resampling: Modeling with resampled data or without.
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :param race: Optional. Race identifier for race-specific final models.
    :return: Dictionary of model parameters.
    """
    # Example parameters; update these based on tuning results
    if resampling == "without":
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
                                1.0: {
                                    'n_estimators': 700, 'min_samples_split': 2, 'min_samples_leaf': 3,
                                    'max_features': None, 'max_depth': 50, 'bootstrap': True
                                },
                                2.0: {
                                    'n_estimators': 350, 'min_samples_split': 20, 'min_samples_leaf': 11,
                                    'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False
                                },
                                3.0: {
                                    'n_estimators': 350, 'min_samples_split': 6, 'min_samples_leaf': 9,
                                    'max_features': None, 'max_depth': 90, 'bootstrap': True
                                },
                                4.0: {
                                    'n_estimators': 800, 'min_samples_split': 16, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 10, 'bootstrap': True
                                },
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
                                1.0: {
                                    'n_estimators': 550, 'min_samples_split': 20, 'min_samples_leaf': 11,
                                    'max_features': None, 'max_depth': None, 'bootstrap': True
                                },
                                2.0: {
                                    'n_estimators': 800, 'min_samples_split': 16, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 10, 'bootstrap': True
                                },
                                3.0: {
                                    'n_estimators': 350, 'min_samples_split': 4, 'min_samples_leaf': 1,
                                    'max_features': None, 'max_depth': 90, 'bootstrap': False
                                },
                                4.0: {
                                    'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 60, 'bootstrap': True
                                },
                            }
                    }
            }
    else:
        params = \
            {
                "AntisocialTrajectory":
                    {
                        "base": {
                            'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                            'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                        },
                        "final":
                            {
                                1.0: {
                                    'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                2.0: {
                                    'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                3.0: {
                                    'n_estimators': 350, 'min_samples_split': 6, 'min_samples_leaf': 9,
                                    'max_features': None, 'max_depth': 90, 'bootstrap': True
                                },
                                4.0: {
                                    'n_estimators': 450, 'min_samples_split': 8, 'min_samples_leaf': 7,
                                    'max_features': 'log2', 'max_depth': 80, 'bootstrap': False
                                },
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "base": {
                            'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                            'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                        },
                        "final":
                            {
                                1.0: {
                                    'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                2.0: {
                                    'n_estimators': 400, 'min_samples_split': 14, 'min_samples_leaf': 1,
                                    'max_features': 'log2', 'max_depth': 40, 'bootstrap': False
                                },
                                3.0: {
                                    'n_estimators': 600, 'min_samples_split': 6, 'min_samples_leaf': 19,
                                    'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False
                                },
                                4.0: {
                                    'n_estimators': 300, 'min_samples_split': 8, 'min_samples_leaf': 11,
                                    'max_features': 'log2', 'max_depth': 30, 'bootstrap': True
                                },
                            }
                    }
            }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(
        target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_base=False, tune_final=False,
        check_overfitting=False, cv=10, resampling="without", base_model_name="RandomForest",
        final_model_name="RandomForest", base_model_type="base", final_model_type="final"
):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling)

    # Map the train and val data as 0 to 3 from 1 to 4
    y_train_old_mapped, y_val_old_mapped, y_test_old_mapped, y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = (
        get_mapped_data(y_train_old, y_val_old, y_test_old, y_train_new, y_val_new, y_test_new))

    # Step 1 - Training base model on old data
    if not tune_base:
        base_model = RandomForestClassifier(**get_model_params(target_variable, base_model_type, resampling))
    else:
        base_model = get_model_instance(base_model_name)

    # Train a base model on old data with dynamic parameters
    base_model = train_and_evaluate_model(base_model, X_train_old, y_train_old_mapped, X_val_old, y_val_old_mapped,
                                          X_test_old, y_test_old_mapped, params[base_model_name], tune_base,
                                          check_overfitting, model_type=base_model_type, cv=cv, resampling=resampling,
                                          script_name=script_name, outcome=target_variable)

    # Prepare new data by combining it with the knowledge from old data.
    X_train_new_enhanced, X_val_new_enhanced, X_test_new_enhanced = prep_data_for_TL(base_model, X_train_new, X_val_new,
                                                                                     X_test_new, race_column)

    # Step 2 - Training race specific model with transfer learning
    # Train and evaluate race-specific final models
    for race in sorted(X_train_new[race_column].unique()):

        if not tune_final:
            # Defining final_model based on the current race in iteration and its respective parameters
            final_model = RandomForestClassifier(**get_model_params(target_variable, final_model_type, race=race,
                                                                    resampling=resampling))
        else:
            final_model = get_model_instance(final_model_name)

        # Prepare the new data for each race iteratively based on the current race being modeled.
        X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race = (
            prep_data_for_race_model(X_train_new_enhanced, y_train_new_mapped, X_val_new_enhanced, y_val_new_mapped,
                                     X_test_new_enhanced, y_test_new_mapped, race, race_column))

        train_and_evaluate_model(final_model, X_train_race, y_train_race, X_val_race, y_val_race, X_test_race,
                                 y_test_race, params[final_model_name], tune_final, check_overfitting,
                                 race, model_type=final_model_type, cv=cv, resampling=resampling,
                                 script_name=script_name, outcome=target_variable)


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "Race",
         "with",
         "with",
         tune_base=False,
         tune_final=False,
         check_overfitting=True,
         resampling="without",
         cv=10,
         base_model_name="RandomForest",
         final_model_name="RandomForest")
