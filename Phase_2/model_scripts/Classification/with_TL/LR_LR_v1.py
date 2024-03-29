import os

import pandas as pd
from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import (equation, calc_shap_values, get_mapped_data, get_model_instance,
                                               interpret_model,
                                               load_data_splits,
                                               prep_data_for_TL,
                                               prep_data_for_race_model, search_spaces, train_and_evaluate)

# Get the base name of the current script and strip the .py extension to use in the filename
script_name = os.path.basename(__file__).replace('.py', '')


def get_model_params(target_variable, model_type, race=None, resampling="without"):
    """
    Returns the model parameters based on the target variable, model type, and race.
    :param resampling: To toggle between parameters tuned with or without resampled data
    :param target_variable: 'AntisocialTrajectory' or 'SubstanceUseTrajectory'
    :param model_type: 'base' or 'final'
    :param race: Race identifier for race-specific final models.
    :return: Dictionary of model parameters.
    """
    # Example parameters; update these based on tuning results
    if resampling == "without":
        params = \
            {
                "AntisocialTrajectory":
                    {
                        "base": {
                            'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs', 'penalty': 'l2',
                            'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False, 'class_weight': None,
                            'C': 1e-05
                        },
                        "final":
                            {
                                1.0: {
                                    'warm_start': False, 'tol': 0.029763514416313194, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 2400, 'fit_intercept': False,
                                    'class_weight': None, 'C': 1e-05
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
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
                        "base": {
                            'warm_start': True, 'tol': 6.951927961775606e-05, 'solver': 'newton-cg', 'penalty': 'l2',
                            'multi_class': 'multinomial', 'max_iter': 19550, 'fit_intercept': True,
                            'class_weight': None, 'C': 138.9495494373139
                        },
                        "final":
                            {
                                1.0: {
                                    'warm_start': False, 'tol': 0.00023357214690901214, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'ovr', 'max_iter': 5100, 'fit_intercept': False,
                                    'class_weight': None, 'C': 1.2648552168552958
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
                                },
                                3.0: {
                                    'warm_start': False, 'tol': 0.05455594781168514, 'solver': 'lbfgs', 'penalty': 'l2',
                                    'multi_class': 'ovr', 'max_iter': 29300, 'fit_intercept': False,
                                    'class_weight': 'balanced', 'C': 33.9322177189533
                                },
                                4.0: {
                                    'warm_start': False, 'tol': 1e-06, 'solver': 'newton-cg', 'penalty': 'l2',
                                    'multi_class': 'ovr', 'max_iter': 7650, 'fit_intercept': True,
                                    'class_weight': 'balanced', 'C': 2329.951810515372
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
                            'warm_start': True, 'tol': 1.1288378916846883e-05, 'solver': 'lbfgs', 'penalty': 'l2',
                            'multi_class': 'multinomial', 'max_iter': 2950, 'fit_intercept': False,
                            'class_weight': None, 'C': 0.7906043210907702
                        },
                        "final":
                            {
                                1.0: {
                                    'warm_start': True, 'tol': 6.951927961775606e-05, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 19550,
                                    'fit_intercept': True, 'class_weight': None, 'C': 138.9495494373139
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 0.0004281332398719391, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 29050,
                                    'fit_intercept': False, 'class_weight': None, 'C': 54.286754393238596
                                },
                                3.0: {
                                    'warm_start': True, 'tol': 3.792690190732254e-05, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 7950,
                                    'fit_intercept': False, 'class_weight': None, 'C': 0.07543120063354623
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
                        "base": {
                            'warm_start': False, 'tol': 2.06913808111479e-05, 'solver': 'lbfgs', 'penalty': 'l2',
                            'multi_class': 'ovr', 'max_iter': 3700, 'fit_intercept': True, 'class_weight': None,
                            'C': 1456.3484775012444
                        },
                        "final":
                            {
                                1.0: {
                                    'warm_start': False, 'tol': 1e-06, 'solver': 'lbfgs', 'penalty': 'l2',
                                    'multi_class': 'ovr', 'max_iter': 20900, 'fit_intercept': True,
                                    'class_weight': None, 'C': 8.286427728546842
                                },
                                2.0: {
                                    'warm_start': True, 'tol': 6.951927961775606e-05, 'solver': 'newton-cg',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 19550,
                                    'fit_intercept': True, 'class_weight': None, 'C': 138.9495494373139
                                },
                                3.0: {
                                    'warm_start': False, 'tol': 1e-06, 'solver': 'newton-cg', 'penalty': 'l2',
                                    'multi_class': 'ovr', 'max_iter': 7650, 'fit_intercept': True,
                                    'class_weight': 'balanced', 'C': 2329.951810515372
                                },
                                4.0: {
                                    'warm_start': True, 'tol': 0.0026366508987303583, 'solver': 'lbfgs',
                                    'penalty': 'l2', 'multi_class': 'multinomial', 'max_iter': 22750,
                                    'fit_intercept': True, 'class_weight': 'balanced', 'C': 0.018420699693267165
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
        check_overfitting=False, cv=10, resampling="without", base_model_name="LogisticRegression",
        final_model_name="LogisticRegression", base_model_type="base", final_model_type="final", interpret=False
):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling)

    # Map the train and val data as 0 to 3 from 1 to 4
    y_train_new_mapped, y_val_new_mapped, y_test_new_mapped, y_train_old_mapped, y_val_old_mapped, y_test_old_mapped = (
        get_mapped_data(y_train_new, y_val_new, y_test_new, y_train_old, y_val_old, y_test_old))

    if not tune_base:
        # Step 1 - Training base model on old data
        base_model = LogisticRegression(**get_model_params(target_variable, "base"))
    else:
        base_model = get_model_instance(base_model_name)

    # Train a base model on old data with dynamic parameters
    base_model = train_and_evaluate(base_model, X_train_old, y_train_old_mapped, X_val_old, y_val_old_mapped,
                                    X_test_old, y_test_old_mapped, params[base_model_name], tune_base,
                                    check_overfitting, model_type=base_model_type, cv=cv, resampling=resampling,
                                    script_name=script_name, outcome=target_variable)

    # Interpreting base-model
    if interpret:
        interpret_model(base_model, base_model_type, X_train_old, base_model_name)
        equation(base_model, X_train_old.columns.tolist())
        # explore_shap_values(base_model, X_train_old)

    # Prepare new data by combining it with the knowledge from old data.
    X_train_new_enhanced, X_val_new_enhanced, X_test_new_enhanced = prep_data_for_TL(base_model, X_train_new, X_val_new,
                                                                                     X_test_new, race_column)

    # Step 2 - Training race specific model with transfer learning
    # Train and evaluate race-specific final models
    for race in sorted(X_train_new[race_column].unique()):

        if not tune_final:
            # Defining final_model based on the current race in iteration and its respective parameters
            final_model = LogisticRegression(**get_model_params(target_variable, "final", race))
        else:
            final_model = get_model_instance(final_model_name)

        # Prepare the new data for each race iteratively based on the current race being modeled.
        X_train_race, y_train_race, X_val_race, y_val_race, X_test_race, y_test_race = (
            prep_data_for_race_model(X_train_new_enhanced, y_train_new_mapped, X_val_new_enhanced, y_val_new_mapped,
                                     X_test_new_enhanced, y_test_new_mapped, race, race_column))

        final_model = train_and_evaluate(final_model, X_train_race, y_train_race, X_val_race, y_val_race, X_test_race,
                                         y_test_race, params[final_model_name], tune_final, check_overfitting, race,
                                         model_type=final_model_type, cv=cv, resampling=resampling,
                                         script_name=script_name, outcome=target_variable)

        # Interpreting final model per race
        if interpret:
            interpret_model(final_model, final_model_type, X_train_race, final_model_name, race)
            equation(final_model, X_train_new.columns.tolist())
            # explore_shap_values(final_model, pd.DataFrame(X_train_race))


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    resampling = "without"

    main(target_variable,
         "Race",
         "with",
         "with",
         False,
         False,
         True,
         5,  # The least populated class in y has only 5 members
         resampling,
         "LogisticRegression",
         "LogisticRegression",
         interpret=True
         )


# First, run the model based on an untuned and non-cv model.
# Second, tune the model and re-evaluate the results.
# Third, with the tuned parameters, re-evaluate the model with cross-validation.
# Fourth, do this process with and without resampling to check results in both scenarios.
# Fifth, save the parameters in the parameter retrieving function for the case which works better - with or without
# resampling.
