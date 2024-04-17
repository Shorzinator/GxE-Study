import os

from sklearn.ensemble import RandomForestClassifier

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
                            {'n_estimators': 600, 'min_samples_split': 6, 'min_samples_leaf': 19, 'max_features': 'log2', 'max_depth': 10, 'bootstrap': False}

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


def main(target_variable, tune_final=False, pgs_old="without", pgs_new="without", cv=5, resampling="without", final_model_name="RandomForest",
         model_type="final"):

    print(f"Running {final_model_name}_v2 model predicting {target_variable} {resampling} resampling and {pgs_new} PGS:\n")

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, _, _, _, _, _, _) = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped = get_mapped_data(y_train_new, y_val_new, y_test_new)

    # Train and evaluate race-specific final models directly on the new data
    if not tune_final:
        # final_model = RandomForestClassifier(random_state=42)
        final_model = RandomForestClassifier(**get_model_params(target_variable, "final", resampling))
    else:
        final_model = get_model_instance(final_model_name)

    train_and_evaluate_with_race_feature(final_model, X_train_new, y_train_new_mapped, X_val_new, y_val_new_mapped,
                                         X_test_new, y_test_new_mapped, final_model_name, tune_final,
                                         model_type=model_type, cv=cv, resampling=resampling)


if __name__ == "__main__":

    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         False,
         "without",
         "without",
         5,
         "without",
         "RandomForest")
