import os

from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance, load_data_splits,
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
            }

    return params[target_variable][model_type]


def main(target_variable, tune_final=False, pgs_old="with", pgs_new="with", cv=5, resampling="without", final_model_name="LogisticRegression",
         model_type="final"):
    """
    :param cv:
    :param final_model_name:
    :param resampling:
    :param tune_final:
    :param model_type:
    :param target_variable:
    :param pgs_new:
    :param pgs_old:

    """
    print(f"Running {final_model_name}_v2 model predicting {target_variable} {resampling} resampling and {pgs_new} PGS:\n")

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, _, _, _, _, _, _) = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = get_mapped_data(y_train_new, y_val_new, y_test_new)

    # Train and evaluate race-specific final models directly on the new data
    if not tune_final:
        final_model = LogisticRegression(random_state=42, multi_class='ovr')
        # final_model = LogisticRegression(**get_model_params(target_variable, "final", resampling))
    else:
        final_model = get_model_instance(final_model_name)

    train_and_evaluate_with_race_feature(final_model, X_train_new, y_train_new_mapped, X_val_new, y_val_new_mapped,
                                         X_test_new, y_test_new_mapped, final_model_name, tune_final,
                                         model_type=model_type, cv=cv, resampling=resampling)


if __name__ == "__main__":

    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         True,
         "without",
         "without",
         5,
         "with",
         "LogisticRegression")
