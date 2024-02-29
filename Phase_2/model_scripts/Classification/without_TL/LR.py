import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from Phase_2.model_scripts.model_utils import load_data_splits, random_search_tuning, search_spaces


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_final=False):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, _, _, _, _, _, _) = (
        load_data_splits(target_variable, pgs_old, pgs_new))

    # Map labels to start from 0
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)

    # Train and evaluate race-specific final models directly on the new data
    for race in sorted(X_train_new[race_column].unique()):
        final_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')

        X_train_race = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new[race_column] == race].ravel()
        X_val_race = X_val_new[X_val_new[race_column] == race].drop(columns=[race_column])
        y_val_race = y_val_new_mapped[X_val_new[race_column] == race].ravel()

        if tune_final:
            final_model, best_params = random_search_tuning(final_model, params['RandomForest'], X_train_race,
                                                            y_train_race)
            print(f"Best Parameters for final model (race {race}): {best_params}")
        else:
            final_model.fit(X_train_race, y_train_race)

        final_accuracy = accuracy_score(y_val_race, final_model.predict(X_val_race))
        print(f"Accuracy for final model without TL (race {race}): {final_accuracy}")


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable, "Race", "with", "with")
