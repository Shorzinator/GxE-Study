import logging

from Phase_2.model_scripts.model_utils import split_data
from Phase_2.preprocessing_scripts.preprocessing_utils import (apply_smote_nc,
                                                               handle_family_clusters, initial_cleaning, load_new_data,
                                                               robust_scaling_continuous_variables_new,
                                                               save_preprocessed_data)
from config import FEATURES_FOR_AST_new, FEATURES_FOR_SUT_new

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessing_pipeline(features, target, file_path_to_save):
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
    df, feature_cols = initial_cleaning(df, features, target)

    # Handle family clusters
    df = handle_family_clusters(df)

    # Splitting the datasets to prevent data/information leakage
    X_train, X_test, y_train, y_test = split_data(df, target)

    # Apply Scaling
    X_train, X_test = robust_scaling_continuous_variables_new(X_train, X_test, feature_cols, target)

    # Encoding the Race feature
    # categorical_features = ["Race"]
    # X_train, X_test = encode_categorical_variables(X_train, X_test, categorical_features)

    # Apply encoding
    if target == 'AntisocialTrajectory':
        # X_train, X_test = encode_ast_sut_variable(X_train, X_test, target, 'SubstanceUseTrajectory', baseline=3)
        # X_train.fillna(0, inplace=True)
        # X_test.fillna(0, inplace=True)
        # categorical_indices = [X_train.columns.get_loc(col) for col in
        #                        ['SubstanceUseTrajectory_1.0', 'SubstanceUseTrajectory_2.0']]

        # categorical_indices = [X_train.columns.get_loc(col) for col in
        #                        ["Race_1.0", "Race_2.0", "Race_3.0", "Race_4.0"]]

        categorical_indices = [X_train.columns.get_loc(col) for col in
                               ["Race", "SubstanceUseTrajectory"]]
        X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)
        # X_train, y_train = apply_smote(X_train, y_train)

    elif target == 'SubstanceUseTrajectory':
        # X_train, X_test = encode_ast_sut_variable(X_train, X_test, target, 'AntisocialTrajectory', baseline=4)
        # X_train.fillna(0, inplace=True)
        # X_test.fillna(0, inplace=True)
        # categorical_indices = [X_train.columns.get_loc(col) for col in
        #                        ["AntisocialTrajectory_1.0", "AntisocialTrajectory_2.0", "AntisocialTrajectory_3.0"]]

        # categorical_indices = [X_train.columns.get_loc(col) for col in
        #                        ["Race_1.0", "Race_2.0", "Race_3.0", "Race_4.0"]]

        categorical_indices = [X_train.columns.get_loc(col) for col in
                               ["Race", "AntisocialTrajectory"]]
        X_train, y_train = apply_smote_nc(X_train, y_train, categorical_features_indices=categorical_indices)
        # X_train, y_train = apply_smote(X_train, y_train)

    # Saving the splits
    if target == "AntisocialTrajectory":
        temp = "AST"
        save_preprocessed_data(X_train, f"{file_path_to_save}X_train_new_{temp}.csv", "X_train")
        save_preprocessed_data(y_train, f"{file_path_to_save}y_train_new_{temp}.csv", "X_test")
        save_preprocessed_data(X_test, f"{file_path_to_save}X_test_new_{temp}.csv", "y_train")
        save_preprocessed_data(y_test, f"{file_path_to_save}y_test_new_{temp}.csv", "y_test")
    else:
        temp = "SUT"
        save_preprocessed_data(X_train, f"{file_path_to_save}X_train_new_{temp}.csv", "X_train")
        save_preprocessed_data(y_train, f"{file_path_to_save}y_train_new_{temp}.csv", "X_test")
        save_preprocessed_data(X_test, f"{file_path_to_save}X_test_new_{temp}.csv", "y_train")
        save_preprocessed_data(y_test, f"{file_path_to_save}y_test_new_{temp}.csv", "y_test")


def main(TARGET):
    # Assigning features based on the outcome.
    if TARGET == "AntisocialTrajectory":
        FEATURES = FEATURES_FOR_AST_new
        SAVE_PATH = "../preprocessed_data/with_PGS/AST_new/"
    else:
        FEATURES = FEATURES_FOR_SUT_new
        SAVE_PATH = "../preprocessed_data/with_PGS/SUT_new/"

    preprocessing_pipeline(FEATURES, TARGET, SAVE_PATH)


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    main(target_2)
