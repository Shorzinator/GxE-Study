from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from Phase_2.model_scripts.model_utils import (get_mapped_data, load_data_splits, prep_data_for_TL, search_spaces,
                                               prep_data_for_race_model, train_and_evaluate)


def get_model_params(target_variable, model_type, resampling, race=None):
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
                        "base": {'splitter': 'best', 'min_samples_split': 0.11,
                                 'min_samples_leaf': 0.020000000000000004, 'min_impurity_decrease': 0.13333333333333333,
                                 'max_leaf_nodes': 60, 'max_features': 'log2', 'max_depth': 24, 'criterion': 'entropy',
                                 'class_weight': None},
                        "final":
                            {
                                1.0: {'n_estimators': 500, 'min_samples_split': 4, 'min_samples_leaf': 11,
                                      'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True},
                                2.0: {'n_estimators': 400, 'min_samples_split': 14, 'min_samples_leaf': 17,
                                      'max_features': None, 'max_depth': 10, 'bootstrap': True},
                                3.0: {'n_estimators': 550, 'min_samples_split': 4, 'min_samples_leaf': 5,
                                      'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                                4.0: {'n_estimators': 300, 'min_samples_split': 8, 'min_samples_leaf': 11,
                                      'max_features': 'log2', 'max_depth': 30, 'bootstrap': True},
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "base": {'splitter': 'best', 'min_samples_split': 0.04, 'min_samples_leaf': 0.04000000000000001,
                                 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 40, 'max_features': 0.4,
                                 'max_depth': 27, 'criterion': 'entropy', 'class_weight': None},
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
                        "base": {'splitter': 'random', 'min_samples_split': 5, 'min_samples_leaf': 4,
                                 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 70, 'max_features': 0.8,
                                 'max_depth': 23, 'criterion': 'gini', 'class_weight': 'balanced'},
                        "final":
                            {
                                1.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                      'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                                2.0: {'n_estimators': 650, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                      'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False},
                                3.0: {'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                      'max_features': 'log2', 'max_depth': 60, 'bootstrap': True},
                                4.0: {'n_estimators': 400, 'min_samples_split': 14, 'min_samples_leaf': 1,
                                      'max_features': 'log2', 'max_depth': 40, 'bootstrap': False},
                            }
                    },
                "SubstanceUseTrajectory":
                    {
                        "base": {'splitter': 'random', 'min_samples_split': 5, 'min_samples_leaf': 4,
                                 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 70, 'max_features': 0.8,
                                 'max_depth': 23, 'criterion': 'gini', 'class_weight': 'balanced'},
                        "final":
                            {
                                1.0: {'n_estimators': 950, 'min_samples_split': 6, 'min_samples_leaf': 3,
                                      'max_features': 'log2', 'max_depth': 60, 'bootstrap': True},
                                2.0: {'n_estimators': 600, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                      'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': True},
                                3.0: {'n_estimators': 600, 'min_samples_split': 8, 'min_samples_leaf': 1,
                                      'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': True},
                                4.0: {'n_estimators': 400, 'min_samples_split': 8, 'min_samples_leaf': 17,
                                      'max_features': 'log2', 'max_depth': None, 'bootstrap': True},
                            }
                    }
            }

    if model_type == 'base':
        return params[target_variable][model_type]
    else:  # 'final' model
        return params[target_variable][model_type][race]


def main(
        target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_base=False, tune_final=False,
        use_cv=False, n_splits=5, resampling="without", base_model_name="RandomForest",
        final_model_name="RandomForest"
):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling)

    # Map the train and val data as 0 to 3 from 1 to 4
    y_train_old_mapped, y_val_old_mapped, y_train_new_mapped, y_val_new_mapped = get_mapped_data(y_train_old, y_val_old,
                                                                                                 y_train_new, y_val_new)

    # Step 1 - Training base model on old data
    if not tune_base:
        base_model = DecisionTreeClassifier(**get_model_params(target_variable, "base", resampling))
    else:
        base_model = DecisionTreeClassifier()

    # Train a base model on old data with dynamic parameters
    base_model = train_and_evaluate(base_model, X_train_old, y_train_old_mapped, X_val_old, y_val_old_mapped,
                                    params[base_model_name], tune_base, use_cv, "base", resampling=resampling)

    # Prepare new data by combining it with the knowledge from old data.
    X_train_new_enhanced, X_val_new_enhanced = prep_data_for_TL(base_model, X_train_new, X_val_new, race_column)

    # Step 2 - Training race specific model with transfer learning
    # Train and evaluate race-specific final models
    for race in sorted(X_train_new[race_column].unique()):

        if not tune_final:
            # Defining final_model based on the current race in iteration, and its respective parameters
            final_model = RandomForestClassifier(**get_model_params(target_variable, "final", race=race,
                                                                    resampling=resampling))
        else:
            final_model = RandomForestClassifier()

        # Prepare the new data for each race iteratively based on the current race being modeled.
        X_train_race, y_train_race, X_val_race, y_val_race = prep_data_for_race_model(X_train_new_enhanced,
                                                                                      y_train_new_mapped,
                                                                                      X_val_new_enhanced,
                                                                                      y_val_new_mapped, race,
                                                                                      race_column)

        train_and_evaluate(final_model, X_train_race, y_train_race, X_val_race, y_val_race, params[final_model_name],
                           tune_final, use_cv, "final", race, n_splits, resampling)


if __name__ == "__main__":
    target_variable = "SubstanceUseTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "Race",
         "with",
         "with",
         tune_base=False,
         tune_final=False,
         use_cv=True,
         resampling="without",
         n_splits=5,
         base_model_name="DecisionTree",
         final_model_name="RandomForest")
