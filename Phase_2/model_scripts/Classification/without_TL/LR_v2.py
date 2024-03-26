import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import (get_model_instance, load_data_splits, search_spaces,
                                               train_and_evaluate_with_race_feature)


# Get the base name of the current script and strip the .py extension to use in the filename
script_name = os.path.basename(__file__).replace('.py', '')


def get_model_params(target_variable, model_type, resampling="without"):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param resampling: Modeling with resampled data or without.
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :return: Dictionary of model parameters.
    """
    # Example parameters; update these based on tuning results

    if resampling == "without":
        params = \
            {
                "AntisocialTrajectory":
                    {
                        "final":
                            {
                                'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs', 'penalty': 'l2',
                                'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False, 'class_weight': None,
                                'C': 1e-05
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                            {
                                'warm_start': False, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7750, 'fit_intercept': True,
                                'class_weight': None, 'C': 1456.3484775012444
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
                                'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs', 'penalty': 'l2',
                                'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False, 'class_weight': None,
                                'C': 1e-05
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "final":
                            {
                                'warm_start': False, 'tol': 0.0026366508987303583, 'solver': 'newton-cg',
                                'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7750, 'fit_intercept': True,
                                'class_weight': None, 'C': 1456.3484775012444
                            }
                    }
            }

    return params[target_variable][model_type]


def main(
        target_variable, pgs_old="with", pgs_new="with", tune_final=False, check_overfitting=False,
        cv=5, resampling="without", final_model_name="LogisticRegression", final_model_type="final"
):
    print(f"Running model for predicting {target_variable} {resampling} resampling:\n")

    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, _, _, _, _, _, _) = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)
    y_test_new_mapped = np.vectorize(label_mapping_new.get)(y_test_new)

    # Train and evaluate race-specific final models directly on the new data

    if not tune_final:
        final_model = LogisticRegression(**get_model_params(target_variable, "final", resampling))
    else:
        final_model = get_model_instance(final_model_name)

    train_and_evaluate_with_race_feature(final_model, X_train_new, y_train_new_mapped.ravel(), X_val_new, y_val_new_mapped.ravel(),
                                         X_test_new, y_test_new_mapped.ravel(), params[final_model_name], tune_final,
                                         check_overfitting, model_type=final_model_type, cv=cv, resampling=resampling,
                                         script_name=script_name, outcome=target_variable)


if __name__ == "__main__":
    resampling = "without"

    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "with",
         "with",
         False,
         True,
         5,
         resampling,
         "LogisticRegression")
