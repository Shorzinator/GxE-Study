import logging

from Phase_2.model_scripts.model_utils import split_data
from Phase_2.preprocessing_scripts.preprocessing_utils import apply_smote_nc, handle_family_clusters, \
    initial_cleaning, \
    load_old_data, robust_scaling_continuous_variables_old, save_preprocessed_data
from config import FEATURES_FOR_AST_old, FEATURES_FOR_SUT_old

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessing_pipeline(features, target, file_path_to_save, resampling):
    """
        Applies the entire preprocessing pipeline to a dataset and saves the preprocessed data.

        Parameters:
        - data_path (str): Path to the dataset.
        - features (list): List of feature names to be included in preprocessing.
        - target (str): The name of the target variable.
        - file_path_to_save (str): Path where the preprocessed data will be saved.

        Returns:
        None: The function saves the preprocessed data to the specified file path.
        """

    # Load data
    df = load_old_data()

    # Initial cleaning and feature engineering
    df, feature_cols = initial_cleaning(df, features, target)

    # Handle family clusters
    df = handle_family_clusters(df)

    # Splitting the datasets to prevent data/information leakage
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)

    # Apply StandardScaler
    X_train, X_val, X_test = robust_scaling_continuous_variables_old(X_train, X_val, X_test, feature_cols, target)

    # Apply encoding
    if resampling:
        print(f"\nResampling is {resampling} - \n")
        if target == 'AntisocialTrajectory':
            categorical_indices = [X_train.columns.get_loc("SubstanceUseTrajectory")]
            X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)
        elif target == 'SubstanceUseTrajectory':
            categorical_indices = [X_train.columns.get_loc("AntisocialTrajectory")]
            X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)

    # Saving the splits
    suffix = "AST" if target == "AntisocialTrajectory" else "SUT"
    if target == "AntisocialTrajectory":
        save_preprocessed_data(X_train, f"{file_path_to_save}X_train_old_{suffix}.csv", "X_train")
        save_preprocessed_data(y_train, f"{file_path_to_save}y_train_old_{suffix}.csv", "y_train")
        save_preprocessed_data(X_val, f"{file_path_to_save}X_val_old_{suffix}.csv", "X_val")
        save_preprocessed_data(y_val, f"{file_path_to_save}y_val_old_{suffix}.csv", "y_val")
        save_preprocessed_data(X_test, f"{file_path_to_save}X_test_old_{suffix}.csv", "X_test")
        save_preprocessed_data(y_test, f"{file_path_to_save}y_test_old_{suffix}.csv", "y_test")
    else:
        save_preprocessed_data(X_train, f"{file_path_to_save}X_train_old_{suffix}.csv", "X_train")
        save_preprocessed_data(y_train, f"{file_path_to_save}y_train_old_{suffix}.csv", "y_train")
        save_preprocessed_data(X_val, f"{file_path_to_save}X_val_old_{suffix}.csv", "X_val")
        save_preprocessed_data(y_val, f"{file_path_to_save}y_val_old_{suffix}.csv", "y_val")
        save_preprocessed_data(X_test, f"{file_path_to_save}X_test_old_{suffix}.csv", "X_test")
        save_preprocessed_data(y_test, f"{file_path_to_save}y_test_old_{suffix}.csv", "y_test")


def main(TARGET, pgs, resampling):
    # Assigning features based on the outcome.
    if TARGET == "AntisocialTrajectory":
        FEATURES = FEATURES_FOR_AST_old
        SAVE_PATH = f"../preprocessed_data/{resampling}_resampling/{pgs}_PGS/AST_old/"
    else:
        FEATURES = FEATURES_FOR_SUT_old
        SAVE_PATH = f"../preprocessed_data/{resampling}_resampling/{pgs}_PGS/SUT_old/"

    resampling_bool = True if resampling == "with" else False
    preprocessing_pipeline(FEATURES, TARGET, SAVE_PATH, resampling_bool)


if __name__ == '__main__':
    resampling = "without"
    pgs = "with"
    main("AntisocialTrajectory", pgs, resampling)
    main("SubstanceUseTrajectory", pgs, resampling)
