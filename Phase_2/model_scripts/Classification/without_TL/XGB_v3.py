import os

from sklearn.ensemble import RandomForestClassifier

from Phase_2.model_scripts.model_utils import (get_mapped_data, get_model_instance,
                                               load_updated_data, train_and_evaluate_updated,
                                               train_and_evaluate_updated_race_stratified)

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
                                'n_estimators': 600, 'min_samples_split': 6, 'min_samples_leaf': 19,
                                'max_features': 'log2', 'max_depth': 10, 'bootstrap': False
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
                                'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False
                            }

                    },
            }

    return params[target_variable][model_type]


def main(target_variable, tune_final=False, cv=5, final_model_name="XGB", model_type="final"):

    print(f"Running {final_model_name}_v3 model predicting {target_variable}:\n")

    # Load data splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_updated_data(target_variable)

    # Map labels to start from 0
    y_train, y_val, y_test = get_mapped_data(y_train, y_val, y_test)

    # Train and evaluate race-specific final models directly on the new data
    if not tune_final:
        final_model = get_model_instance(final_model_name)
        # final_model = RandomForestClassifier(**get_model_params(target_variable, "final", resampling))
    else:
        final_model = get_model_instance(final_model_name)

    # train_and_evaluate_updated(final_model, X_train, y_train.values.ravel(), X_val, y_val.values.ravel(), X_test,
    #                            y_test.values.ravel(), final_model_name, tune_final, cv)

    train_and_evaluate_updated_race_stratified(final_model, X_train, y_train, X_val, y_val, X_test, y_test,
                                               final_model_name, tune_final, cv,
                                               ['H1GI6A', 'H1GI6B', 'H1GI6C', 'H1GI6D', 'H1GI6E'])


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         False,
         5,
         "XGB")
