import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from Phase_2.model_scripts.model_utils import load_data_splits, random_search_tuning, search_spaces


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_base=False, tune_final=False):

    params = search_spaces()

    # Load data splits, including validation sets
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new)

    # Map labels to start from 0
    label_mapping_old = {label: i for i, label in enumerate(np.unique(y_train_old))}
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_old = np.vectorize(label_mapping_old.get)(y_train_old)
    y_test_old = np.vectorize(label_mapping_old.get)(y_test_old)
    y_val_old = np.vectorize(label_mapping_old.get)(y_val_old)
    y_train_new = np.vectorize(label_mapping_new.get)(y_train_new)
    y_test_new = np.vectorize(label_mapping_new.get)(y_test_new)
    y_val_new = np.vectorize(label_mapping_new.get)(y_val_new)

    # Train a base model on old data
    base_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    if tune_base:
        base_model, best_params = random_search_tuning(base_model, params['LogisticRegression'], X_train_old,
                                                       y_train_old.values.ravel())
        print(f"Best Parameters for base model: {best_params}")
    else:
        base_model.fit(X_train_old, y_train_old)
    base_model_accuracy = accuracy_score(y_val_old, base_model.predict(X_val_old))
    print(f"Validation Accuracy for base model: {base_model_accuracy}")

    # Enhance new data with predicted probabilities from the base model for final model training
    base_model_probs_new = base_model.predict_proba(X_train_new.drop(columns=[race_column]))
    X_train_new_enhanced = np.hstack([X_train_new.drop(columns=[race_column]), base_model_probs_new])

    # Convert X_train_new_enhanced back to a pandas DataFrame
    X_train_new_enhanced = pd.DataFrame(X_train_new_enhanced, columns=X_train_new.drop(columns=[race_column]).columns.
                                        tolist() + ['base_model_probs'])

    # Ensure race column is included for filtering
    X_train_new_enhanced[race_column] = X_train_new[race_column].values

    # Train and evaluate race-specific final models
    for race in sorted(X_train_new[race_column].unique()):
        final_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        X_train_race = X_train_new_enhanced[X_train_new_enhanced[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new[X_train_new_enhanced[race_column] == race]

        if tune_final:
            final_model, best_params = random_search_tuning(final_model, params, X_train_race,
                                                            y_train_race)
            print(f"Best Parameters for final model (race {race}): {best_params}")
        else:
            final_model.fit(X_train_race, y_train_race)

        # Use the validation set for evaluating final model performance
        X_val_race = X_val_new[X_val_new[race_column] == race].drop(columns=[race_column])
        y_val_race = y_val_new[X_val_new[race_column] == race]
        final_accuracy = accuracy_score(y_val_race, final_model.predict(X_val_race))
        print(f"Validation Accuracy for final model (race {race}): {final_accuracy}")


if __name__ == "__main__":
    target_variable = "AntisocialTrajectory"  # or "SubstanceUseTrajectory"
    main(target_variable, "Race", "with", "with")
