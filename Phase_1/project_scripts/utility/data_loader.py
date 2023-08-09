import logging

import pandas as pd

from Phase_1.project_scripts.utility.path_utils import get_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = get_data_path("Data_GxE_on_EXT_trajectories_new.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Enhanced Validity Checks
    logger.info(f"Data types:\n{df.dtypes}")  # Check data types
    logger.info(f"Missing values count:\n{df.isnull().sum()}")  # Check for missing values
    logger.info(f"First 5 rows of data:\n{df.head()}\n")  # Display first few rows

    return df
