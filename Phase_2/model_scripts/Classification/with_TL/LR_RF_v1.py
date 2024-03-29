from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from Phase_2.model_scripts.model_utils import get_mapped_data, load_data_splits, prep_data_for_TL, \
    prep_data_for_race_model, search_spaces, train_and_evaluate


def main(
        target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_base=False, tune_final=False,
        use_cv=False, n_splits=5, resampling="without", base_model_name="LogisticRegression",
        final_model_name="LogisticRegression"
):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new, resampling)

    # Map the train and val data as 0 to 3 from 1 to 4
    y_train_old_mapped, y_val_old_mapped, y_train_new_mapped, y_val_new_mapped = get_mapped_data(y_train_new, y_val_new,
                                                                                                 y_test_new, y_train_old,
                                                                                                 y_val_old, y_test_old)

    # Step 1 - Training base model on old data
    # base_model = LogisticRegression(**get_model_params(target_variable, "base"))
    base_model = LogisticRegression()

    # Train a base model on old data with dynamic parameters
    base_model = train_and_evaluate(base_model, X_train_old, y_train_old_mapped, X_val_old, y_val_old_mapped,
                                    params[base_model_name], tune_base, use_cv, "base")

    # Prepare new data by combining it with the knowledge from old data.
    X_train_new_enhanced, X_val_new_enhanced = prep_data_for_TL(base_model, X_train_new, X_val_new, race_column)

    # Step 2 - Training race specific model with transfer learning
    # Train and evaluate race-specific final models
    for race in sorted(X_train_new[race_column].unique()):
        # Defining final_model based on the current race in iteration and its respective parameters
        # final_model = RandomForestClassifier(**get_model_params(target_variable, "final", race))
        final_model = RandomForestClassifier()

        # Prepare the new data for each race iteratively based on the current race being modeled.
        X_train_race, y_train_race, X_val_race, y_val_race = prep_data_for_race_model(X_train_new_enhanced,
                                                                                      y_train_new_mapped,
                                                                                      X_val_new_enhanced,
                                                                                      y_val_new_mapped, race,
                                                                                      race_column)

        train_and_evaluate(final_model, X_train_race, y_train_race, X_val_race, y_val_race, params[final_model_name],
                           tune_final, use_cv, "final", race, n_splits)


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable,
         "Race",
         "with",
         "with",
         False,
         False,
         True,
         5,  # The least populated class in y has only 5 members
         "without",  # Does not do well with resampling
         "LogisticRegression",
         "RandomForest"
         )
