import logging

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import RobustScaler

from Phase_2.model_scripts.model_utils import split_data
from Phase_2.preprocessing_scripts.preprocessing_utils import robust_scaling_continuous_variables_new, \
    save_preprocessed_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessing_pipeline(df, target, file_path_to_save):

    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    rows_after_dropping = len(df)
    logger.info(f"Dropped {initial_rows - rows_after_dropping} rows due to missing target values...")

    # Assuming your DataFrame is named 'df'
    df = df.drop(columns=['ID'])

    # Splitting the datasets to prevent data/information leakage
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)
    cols = X_train.columns
    # print(cols)

    # Apply robust scaling to the data
    # scaler = RobustScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_train = pd.DataFrame(X_train, columns=cols)
    #
    # X_val = scaler.fit_transform(X_val)
    # X_val = pd.DataFrame(X_val, columns=cols)
    #
    # X_test = scaler.transform(X_test)
    # X_test = pd.DataFrame(X_test, columns=cols)

    # logger.info("Continuous variables normalized in both training and test set.")

    # Saving the splits
    suffix = "AST" if target == "AntisocialTrajectory" else "SUT"
    save_preprocessed_data(X_train, f"{file_path_to_save}X_train_updated_{suffix}.csv", "X_train")
    save_preprocessed_data(y_train, f"{file_path_to_save}y_train_updated_{suffix}.csv", "y_train")
    save_preprocessed_data(X_val, f"{file_path_to_save}X_val_updated_{suffix}.csv", "X_val")
    save_preprocessed_data(y_val, f"{file_path_to_save}y_val_updated_{suffix}.csv", "y_val")
    save_preprocessed_data(X_test, f"{file_path_to_save}X_test_updated_{suffix}.csv", "X_test")
    save_preprocessed_data(y_test, f"{file_path_to_save}y_test_updated_{suffix}.csv", "y_test")


def main(df, TARGET):
    # Assigning features based on the outcome.
    if TARGET == "AntisocialTrajectory":
        SAVE_PATH = f"../preprocessed_data/updated_data/AST/"
    else:
        SAVE_PATH = f"../preprocessed_data/updated_data/SUT/"

    preprocessing_pipeline(df, TARGET, SAVE_PATH)


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\merged_without_PGS.csv")

    main(df, "AntisocialTrajectory")
    main(df, "SubstanceUseTrajectory")
