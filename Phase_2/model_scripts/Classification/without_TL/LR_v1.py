import os
import statistics

import numpy as np
from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance, load_data_splits,
                                               prep_data_for_race_model,
                                               search_spaces,
                                               train_and_evaluate)

# Get the base name of the current script and strip the .py extension to use in the filename
script_name = os.path.basename(__file__).replace('.py', '')


def get_model_params(target_variable, model_type, race=None, resampling="without"):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param resampling: Modeling with resampled data or without.
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :param race: Race identifier for race-specific final models.
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
                                1.0: {
                                    'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False,
                                    'class_weight': None, 'C': 1e-05
                                },
                                2.0: {
                                    'warm_start': False, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 14750, 'fit_intercept': False,
                                    'class_weight': None, 'C': 62505.51925273976
                                },
                                3.0: {
                                    'warm_start': False, 'tol': 0.05455594781168514, 'solver': 'lbfgs', 'penalty': 'l2',
                                    'multi_class': 'multinomial', 'max_iter': 17950, 'fit_intercept': False,
                                    'class_weight': None, 'C': 86.85113737513521
                                },
                                4.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
                                },
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                            {
                                1.0: {
                                    'warm_start': True, 'tol': 6.951927961775606e-05, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 19550,
                                    'fit_intercept': True, 'class_weight': None, 'C': 138.9495494373139
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
                                },
                                3.0: {
                                    'warm_start': True, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 5850, 'fit_intercept': True,
                                    'class_weight': None, 'C': 568.9866029018305
                                },
                                4.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 27950,
                                    'fit_intercept': True, 'class_weight': 'balanced', 'C': 86.85113737513521
                                },
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
                                    'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False,
                                    'class_weight': None, 'C': 1e-05
                                },
                                2.0: {
                                    'warm_start': False, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 14750, 'fit_intercept': False,
                                    'class_weight': None, 'C': 62505.51925273976
                                },
                                3.0: {
                                    'warm_start': False, 'tol': 0.05455594781168514, 'solver': 'lbfgs', 'penalty': 'l2',
                                    'multi_class': 'multinomial', 'max_iter': 17950, 'fit_intercept': False,
                                    'class_weight': None, 'C': 86.85113737513521
                                },
                                4.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
                                },
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                            {
                                1.0: {
                                    'warm_start': True, 'tol': 6.951927961775606e-05, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 19550,
                                    'fit_intercept': True, 'class_weight': None, 'C': 138.9495494373139
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
                                },
                                3.0: {
                                    'warm_start': True, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 5850, 'fit_intercept': True,
                                    'class_weight': None, 'C': 568.9866029018305
                                },
                                4.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 27950,
                                    'fit_intercept': True, 'class_weight': 'balanced', 'C': 86.85113737513521
                                },
                            }
                    }
            }

    return params[target_variable][model_type][race]


def main(
        target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_final=False, check_overfitting=False,
        cv=10, resampling="without", final_model_name="LogisticRegression", final_model_type="final"
):
    print(f"Running model for predicting {target_variable} {resampling} resampling:\n")

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, _, _, _, _, _, _) = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = get_mapped_data(y_train_new, y_val_new, y_test_new)

    # Train and evaluate race-specific final models directly on the new data
    for race in sorted(X_train_new[race_column].unique()):
        # Defining final_model based on the current race in iteration and its respective parameters
        if not tune_final:
            final_model = LogisticRegression(**get_model_params(target_variable, "final", race, resampling))
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
                           resampling=resampling, script_name=script_name, outcome=target_variable)


if __name__ == "__main__":
    resampling = "without"

    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "Race",
         "with",
         "with",
         False,
         True,
         5,
         resampling,
         "LogisticRegression")
