import os

from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance,
                                               load_updated_data, train_and_evaluate_updated,
                                               )

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


def main(target_variable, tune_final=False, cv=5, final_model_name="LogisticRegression", model_type="final"):
    """
    :param cv:
    :param final_model_name:
    :param tune_final:
    :param model_type:
    :param target_variable:
    """
    print(f"Running {final_model_name}_v3 model predicting {target_variable}:\n")

    # Load data splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_updated_data(target_variable)

    # Train and evaluate race-specific final models directly on the new data
    if not tune_final:
        final_model = LogisticRegression(random_state=42, multi_class='ovr')
        # final_model = LogisticRegression(**get_model_params(target_variable, "final", resampling))
    else:
        final_model = get_model_instance(final_model_name)

    train_and_evaluate_updated(final_model, X_train, y_train.values.ravel(), X_val, y_val.values.ravel(), X_test,
                               y_test.values.ravel(), final_model_name, tune_final, cv)


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         True,
         5,
         "LogisticRegression")
