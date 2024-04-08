from xgboost import XGBClassifier

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
                            1.0:
                                {
                                    'subsample': 0.9, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 800,
                                    'min_child_weight': 3, 'max_depth': 6, 'max_delta_step': 0, 'learning_rate': 0.1,
                                    'gamma': 0, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.9
                                },
                            2.0:
                                {
                                    'subsample': 0.9, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 800,
                                    'min_child_weight': 3, 'max_depth': 6, 'max_delta_step': 0, 'learning_rate': 0.1,
                                    'gamma': 0, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.9
                                },
                            3.0:
                                {
                                    'subsample': 0.6, 'reg_lambda': 30, 'reg_alpha': 1.0, 'n_estimators': 1500,
                                    'min_child_weight': 8, 'max_depth': 6, 'max_delta_step': 3, 'learning_rate': 0.1,
                                    'gamma': 0, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6
                                },
                            4.0:
                                {
                                    'subsample': 0.6, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 1500,
                                    'min_child_weight': 3, 'max_depth': 6, 'max_delta_step': 3, 'learning_rate': 0.01,
                                    'gamma': 0, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.9
                                },
                        }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                        {
                            1.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                  'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                            2.0: {'n_estimators': 700, 'min_samples_split': 16, 'min_samples_leaf': 3,
                                  'max_features': 'log2', 'max_depth': 20, 'bootstrap': False},
                            3.0: {'n_estimators': 350, 'min_samples_split': 10, 'min_samples_leaf': 11,
                                  'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True},
                            4.0: {'n_estimators': 650, 'min_samples_split': 18, 'min_samples_leaf': 1,
                                  'max_features': 'log2', 'max_depth': 20, 'bootstrap': True},
                        }
                    }
            }
    else:
        params = \
            {
                "AntisocialTrajectory":
                    {
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
                                    'n_estimators': 550, 'min_samples_split': 4, 'min_samples_leaf': 5,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                4.0: {
                                    'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 60, 'bootstrap': True
                                },
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                            {
                                1.0: {
                                    'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                    'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                                },
                                2.0: {
                                    'n_estimators': 700, 'min_samples_split': 16, 'min_samples_leaf': 3,
                                    'max_features': 'log2', 'max_depth': 20, 'bootstrap': False
                                },
                                3.0: {
                                    'n_estimators': 350, 'min_samples_split': 10, 'min_samples_leaf': 11,
                                    'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True
                                },
                                4.0: {
                                    'n_estimators': 650, 'min_samples_split': 18, 'min_samples_leaf': 1,
                                    'max_features': 'log2', 'max_depth': 20, 'bootstrap': True
                                },
                            }
                    }
            }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(target_variable, race_column="Race", tune_final=False, cv=10, resampling="with",
         final_model_name="RandomForest", final_model_type="final"):

    print(f"Running model for predicting {target_variable} {resampling} resampling:\n")

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, resampling=resampling)

    # Map labels to start from 0
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = get_mapped_data(y_train_new, y_val_new, y_test_new)

    # Train and evaluate race-specific interim models
    for race in sorted(X_train_new[race_column].unique()):
        # Defining final_model based on the current race in iteration and its respective parameters
        if not tune_final:
            # final_model = get_model_instance(final_model_name)
            final_model = XGBClassifier(
                random_state=42,
                n_estimators=1200,
                reg_alpha=0.5,
                reg_lambda=25,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                learning_rate=0.08,
                eval_metric='mlogloss'
            )

            # final_model = XGBClassifier(**get_model_params(target_variable, "final", race, resampling))
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
        True,
        5,
        "with",
        "XGB"
    )
