import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

from Phase_2.model_scripts.model_utils import (load_data_splits, random_search_tuning, search_spaces)


def main(target_variable, race_column="Race", pgs_old="with", pgs_new="with", tune_final=False):
    params = search_spaces()

    # Load data splits
    (X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, X_train_old, X_val_old, X_test_old,
     y_train_old, y_val_old, y_test_old) = load_data_splits(target_variable, pgs_old, pgs_new)

    # Map labels to start from 0
    label_mapping_new = {label: i for i, label in enumerate(np.unique(y_train_new))}
    y_train_new_mapped = np.vectorize(label_mapping_new.get)(y_train_new)
    y_val_new_mapped = np.vectorize(label_mapping_new.get)(y_val_new)

    # Train and evaluate race-specific final models without transfer learning
    for race in sorted(X_train_new[race_column].unique()):
        X_train_race = X_train_new[X_train_new[race_column] == race].drop(columns=[race_column])
        y_train_race = y_train_new_mapped[X_train_new[race_column] == race].ravel()
        X_val_race = X_val_new[X_val_new[race_column] == race].drop(columns=[race_column])
        y_val_race = y_val_new_mapped[X_val_new[race_column] == race].ravel()

        # Initialize LightGBM model
        final_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, subsample=0.9736842105263157,
                                         reg_lambda=0.5555555555555556, reg_alpha=0.8888888888888888, num_leaves=315,
                                         min_split_gain=0.4444444444444444, min_child_samples=91, max_depth=7,
                                         learning_rate=0.007937005259840991, colsample_bytree=0.7777777777777778)

        if tune_final:
            final_model, best_params = random_search_tuning(final_model, params['LightGBM'],
                                                            X_train_race, y_train_race)
            print(f"Best Parameters for final model (race {race}): {best_params}")
        else:
            final_model.fit(X_train_race, y_train_race)

        final_accuracy = accuracy_score(y_val_race, final_model.predict(X_val_race))
        print(f"Accuracy for final model without TL (race {race}) on validation set: {final_accuracy}")


if __name__ == "__main__":
    target_variable = "SubstanceUseTrajectory"  # "AntisocialTrajectory" or "SubstanceUseTrajectory"
    main(target_variable, "Race", "with", "with", tune_final=False)
