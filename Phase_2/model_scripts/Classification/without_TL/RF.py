import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from Phase_2.model_scripts.model_utils import (load_data_splits, random_search_tuning_intermediate, search_spaces)


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_final=False):
    params = search_spaces()

    # Load data splits
    X_train_new, X_train_old, X_test_new, X_test_old, y_train_new, y_train_old, y_test_new, y_test_old = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_test_new_mapped = np.vectorize(label_mapping_new.get)(y_test_new)

    # Train and evaluate race-specific final models without transfer learning
    for race in sorted(X_train_new[race_column].unique()):
        # Filter data for the current race
        X_train_race = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new[race_column] == race].ravel()
        X_test_race = X_test_new[X_test_new[race_column] == race].drop(columns=[race_column])
        y_test_race = y_test_new_mapped[X_test_new[race_column] == race].ravel()

        # Define the final model (Random Forest)
        final_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

        if tune_final:
            # Tune the final model's hyperparameters for the current race
            final_model, best_params = random_search_tuning_intermediate(final_model, 'RandomForest', params,
                                                                         X_train_race, y_train_race)
            print(f"Best Parameters for final model (race {race}): {best_params}")
        else:
            # Train the final model on race-specific data
            final_model.fit(X_train_race, y_train_race)

        # Evaluate the final model on the race-specific test set
        final_accuracy = accuracy_score(y_test_race, final_model.predict(X_test_race))
        print(f"Accuracy for final model without TL (race {race}): {final_accuracy}")


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # or "SubstanceUseTrajectory"
    main(target_variable, "Race", "with", "with")
