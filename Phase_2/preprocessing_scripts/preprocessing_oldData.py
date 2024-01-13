import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import yeojohnson
import logging

from config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from project_scripts import get_data_path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_PATH_old = get_data_path("Data_GxE_on_EXT_trajectories_old.csv")


def load_old_data():
    df = pd.read_csv(DATA_PATH_old)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    return df


def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df


def apply_yeojohnson_transformation(df, features):
    # Handle outliers
    # df = remove_outliers(df, features)

    transformed_df = df.copy()
    for feature in features:
        try:
            transformed, _ = yeojohnson(transformed_df[feature])
            transformed_df[feature] = transformed
        except Exception as e:
            logger.error(f"Error in Yeo-Johnson transformation for {feature}: {e}")
    return transformed_df


def min_max_scaling_continuous_features(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def encode_ast_sut_variable(df, column, baseline):
    """
    Encodes a categorical variable using one-hot encoding, excluding the baseline category.
    :param df: DataFrame
    :param column: Column to be encoded
    :param baseline: Baseline category to be excluded
    :return: DataFrame with encoded column
    """
    if column in df.columns:
        # Convert column to float for consistency
        df[column] = df[column].astype(float)

        # Check if baseline exists
        if baseline not in df[column].unique():
            logger.warning(f"Baseline category {baseline} not found in {column}. Skipping encoding.")
            return df

        # Use OneHotEncoder without explicitly defining categories
        encoder = OneHotEncoder(drop=[baseline], sparse_output=False)
        encoded_features = encoder.fit_transform(df[[column]])

        # Create DataFrame with new encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

        # Convert these representations to NaN
        missing_value_representations = ['<null>', '-', '']
        for representation in missing_value_representations:
            df.replace(representation, np.nan, inplace=True)

        # Manually set null values to 0 in the new columns
        for col in encoded_df.columns:
            encoded_df[col].fillna(0, inplace=True)

        # Drop the original column and join the new features
        return df.drop(column, axis=1).join(encoded_df)

    else:
        logger.warning(f"{column} not found in DataFrame.")
        return df


def standard_scaling_continuous_variables(df, feature_cols, target):
    f = feature_cols.copy()
    f.remove("Is_Male")
    if target == "AntisocialTrajectory":
        f.remove("SubstanceUseTrajectory")
    else:
        f.remove("AntisocialTrajectory")

    # Add a check to ensure all columns are present
    missing_cols = [col for col in f if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in dataframe for normalization: {missing_cols}")
        return df  # or handle the missing columns as appropriate

    scaler = StandardScaler()
    df[f] = scaler.fit_transform(df[f])
    return df


def impute_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def handle_family_clusters(df):
    family_counts = df['FamilyID'].value_counts()
    df['InFamilyCluster'] = df['FamilyID'].apply(lambda x: int(family_counts[x] > 1) if pd.notnull(x) else 0)
    return df


def initial_cleaning(df, features, target):
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)
    df.drop("Sex", inplace=True, axis=1)

    # Handling outliers
    features_to_handle_outliers = ['PolygenicScoreEXT', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect',
                                   'ParentalWarmth', 'Age']  # Adjust as needed
    for feature in features_to_handle_outliers:  # Define this list based on your dataset
        df = remove_outliers(df, feature)

    # Drop rows where the target variable is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target])
    rows_after_dropping = len(df)
    logger.info(f"Dropped {initial_rows - rows_after_dropping} rows due to missing target values...")

    # Feature Engineering
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']
    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    return df, feature_cols


def plot_feature_distribution(df, feature):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df[feature].hist(bins=30, alpha=0.5)
    plt.title(f'Histogram of {feature}')

    # QQ-Plot
    plt.subplot(1, 2, 2)
    scipy.stats.probplot(df[feature], dist="norm", plot=plt)
    plt.title(f"QQ Plot of {feature}")
    plt.savefig(f"results/old_data/QQ Plot of {feature} after YJ Transformation")
    # plt.show()


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def save_preprocessed_data(df, file_path, target):
    if target == "AntisocialTrajectory":
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)


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
    df = load_old_data()
    df.drop("ID", axis=1, inplace=True)

    # Initial cleaning and feature engineering
    df, feature_cols = initial_cleaning(df, features, target)
    logger.info("Initial cleaning and feature engineering completed.")

    # Handle family clusters
    df = handle_family_clusters(df)
    logger.info("Family clusters handled.")
    df.drop("FamilyID", axis=1, inplace=True)

    # Splitting the datasets to prevent data/information leakage


    # Apply transformation
    continuous_features = ["DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth"]

    df = apply_yeojohnson_transformation(df, continuous_features)
    logger.info("Yeo-Johnson transformation applied.")

    # for col in continuous_features:
    #     plot_feature_distribution(df, col)

    # df = scale_features(df, continuous_features)
    # logger.info("MinMaxScalar applied.")

    # Apply StandardScalar to continuous variables
    df = standard_scaling_continuous_variables(df, feature_cols, target)
    logger.info("Continuous variables normalized.")

    # Conditional encoding based on target
    if target == 'AntisocialTrajectory':
        logger.info("Applying encoding on AST.")
        df = encode_ast_sut_variable(df, 'SubstanceUseTrajectory', baseline=3)
    elif target == 'SubstanceUseTrajectory':
        logger.info("Applying encoding on SUT.")
        df = encode_ast_sut_variable(df, 'AntisocialTrajectory', baseline=4)
    logger.info("Categorical variables handled.")

    # Impute missing values
    # df = impute_missing_values(df)
    # logger.info("Missing values imputed.")

    df.fillna(0, inplace=True)

    # Save the preprocessed data
    save_preprocessed_data(df, file_path_to_save, target)
    logger.info(f"Preprocessed data saved to {file_path_to_save}.")


def main(TARGET):
    # Assigning features based on the outcome.
    if TARGET == "AntisocialTrajectory":
        FEATURES = FEATURES_FOR_AST
        SAVE_PATH = '../preprocessed_data/preprocessed_data_old_AST.csv'
    else:
        FEATURES = FEATURES_FOR_SUT
        SAVE_PATH = '../preprocessed_data/preprocessed_data_old_SUT.csv'

    preprocessing_pipeline(FEATURES, TARGET, SAVE_PATH)


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(target_2)
