import logging

from Phase_2.model_scripts.model_utils import split_data
from Phase_2.preprocessing_scripts.preprocessing_utils import (apply_smote_nc,
                                                               handle_family_clusters,
                                                               initial_cleaning, initial_cleaning_without_genetics,
                                                               load_new_data,
                                                               robust_scaling_continuous_variables_new,
                                                               save_preprocessed_data)
from config import FEATURES_FOR_AST_new, FEATURES_FOR_SUT_new

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessing_pipeline(features, target, file_path_to_save, resampling, pgs):
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
    df = load_new_data()
    df = df.dropna(subset=["Race"])
    df = df.drop(df[df["Race"] == 5].index)

    # Initial cleaning and feature engineering
    if pgs == "without":
        df = initial_cleaning_without_genetics(df, target)
        pgs_dropped = True
    else:
        df, feature_cols = initial_cleaning(df, features, target)
        pgs_dropped = False

    # Handle family clusters
    df = handle_family_clusters(df)

    # Splitting the datasets to prevent data/information leakage
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)

    # Apply StandardScaler
    X_train, X_val, X_test = robust_scaling_continuous_variables_new(X_train, X_val, X_test, features, target, pgs_dropped)

    # Apply encoding
    if resampling:
        if target == 'AntisocialTrajectory':
            categorical_indices = [X_train.columns.get_loc(col) for col in
                                   ["Race", "SubstanceUseTrajectory"]]
            X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)

        elif target == 'SubstanceUseTrajectory':
            categorical_indices = [X_train.columns.get_loc(col) for col in
                                   ["Race", "AntisocialTrajectory"]]
            X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)

        # Saving the splits
    suffix = "AST" if target == "AntisocialTrajectory" else "SUT"
    save_preprocessed_data(X_train, f"{file_path_to_save}X_train_new_{suffix}.csv", "X_train")
    save_preprocessed_data(y_train, f"{file_path_to_save}y_train_new_{suffix}.csv", "y_train")
    save_preprocessed_data(X_val, f"{file_path_to_save}X_val_new_{suffix}.csv", "X_val")
    save_preprocessed_data(y_val, f"{file_path_to_save}y_val_new_{suffix}.csv", "y_val")
    save_preprocessed_data(X_test, f"{file_path_to_save}X_test_new_{suffix}.csv", "X_test")
    save_preprocessed_data(y_test, f"{file_path_to_save}y_test_new_{suffix}.csv", "y_test")


def main(target, pgs, resampling):
    # Assigning features based on the outcome.
    if target == "AntisocialTrajectory":
        features = FEATURES_FOR_AST_new
        SAVE_PATH = f"../preprocessed_data/{resampling}_resampling/{pgs}_PGS/AST_new/"
    else:
        features = FEATURES_FOR_SUT_new
        SAVE_PATH = f"../preprocessed_data/{resampling}_resampling/{pgs}_PGS/SUT_new/"

    resampling_bool = True if resampling == "with" else False
    preprocessing_pipeline(features, target, SAVE_PATH, resampling_bool, pgs)


if __name__ == '__main__':
    resampling = "with"
    pgs = "without"
    main("AntisocialTrajectory", pgs, resampling)
    main("SubstanceUseTrajectory", pgs, resampling)
