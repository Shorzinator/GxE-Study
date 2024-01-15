import logging

import numpy as np
import pandas as pd
import scipy
from imblearn.over_sampling import ADASYN
from matplotlib import pyplot as plt
from scipy.stats import yeojohnson
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from utility.path_utils import get_data_path

DATA_PATH_new = get_data_path("Data_GxE_on_EXT_trajectories_new.csv")
DATA_PATH_old = get_data_path("Data_GxE_on_EXT_trajectories_old.csv")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_new_data():
    df = pd.read_csv(DATA_PATH_new)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    return df


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


def is_continuous_feature(df, feature, unique_threshold=10, skewness_threshold=0.3):
    """
    Determines if a feature in a dataframe is continuous based on several criteria:
    - Data type (float or int)
    - Number of unique values
    - Distribution skewness

    Parameters:
    - df (DataFrame): The dataframe containing the feature.
    - feature (str): The name of the feature to check.
    - unique_threshold (int): The minimum number of unique values to consider a feature continuous.
    - skewness_threshold (float): The minimum skewness to consider a feature continuous.

    Returns:
    - bool: True if the feature is continuous, False otherwise.
    """
    if df[feature].dtype in [np.float64, np.int64]:
        if df[feature].nunique() > unique_threshold:
            if abs(df[feature].skew()) > skewness_threshold:
                return True
    return False


def apply_yeojohnson_transformation(df):
    """
    Apply the Yeo-Johnson transformation to continuous features in the dataframe.
    The function uses multiple criteria to identify continuous features.

    Parameters:
    - df (DataFrame): The dataframe to transform.

    Returns:
    - DataFrame: The dataframe with transformed features.
    """
    transformed_df = df.copy()

    for feature in df.columns:
        if is_continuous_feature(df, feature):
            try:
                # Only apply transformation if the feature has more than one unique value
                if len(df[feature].unique()) > 1:
                    transformed, _ = yeojohnson(transformed_df[feature])
                    transformed_df[feature] = transformed
                else:
                    logger.info(f"No transformation applied for {feature} as it has a single unique value.")
            except Exception as e:
                logger.error(f"Error in Yeo-Johnson transformation for {feature}: {e}")

    logger.info("Yeo-Johnson transformation applied.")

    return transformed_df


def min_max_scaling_continuous_features(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def handle_categorical_variables(df):
    categorical_features = ['Race']  # Update this list as per your actual columns
    existing_features = [col for col in categorical_features if col in df.columns]
    if not existing_features:
        logger.warning("No categorical features found for one-hot encoding.")
        return df

    df = df.dropna(subset=['Race'])

    categorical_transformer = OneHotEncoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, existing_features)
        ],
        remainder='passthrough'
    )

    # Apply the ColumnTransformer
    transformed_data = preprocessor.fit_transform(df)

    # Create a new DataFrame with transformed data and updated column names
    transformed_df = pd.DataFrame(transformed_data, columns=preprocessor.get_feature_names_out(), index=df.index)

    # Remove prefixes from column names
    transformed_df.columns = [col.split('__')[-1] for col in transformed_df.columns]

    return transformed_df


def encode_ast_sut_variable(df, target, column, baseline):
    """
    Encodes a categorical variable using one-hot encoding, excluding the baseline category.
    :param df: DataFrame
    :param target: The target variable (AST or SUT)
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

        if target == "AntisocialTrajectory":
            logger.info("Applied encoding on SUT.")
        else:
            logger.info("Applied encoding on AST.")

        logger.info("Categorical variables handled.")

        # Drop the original column and join the new features
        # temp = df.drop(column, axis=1).join(encoded_df)

        return df.drop(column, axis=1).join(encoded_df)

    else:
        logger.warning(f"{column} not found in DataFrame.")
        return df


def standard_scaling_continuous_variables_old(df, feature_cols, target):

    f = feature_cols.copy()
    f.remove("Is_Male")
    f.remove("PolygenicScoreEXT_x_Is_Male")

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

    logger.info("Continuous variables normalized.")

    return df


def standard_scaling_continuous_variables_new(df, feature_cols, target):

    f = feature_cols.copy()

    f.remove("Is_Male")
    f.remove("Race")
    f.remove("PolygenicScoreEXT_x_Age")

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


def apply_adasyn(X_train, y_train, strategy="auto"):
    adasyn = ADASYN(sampling_strategy=strategy, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    # Rounding off binary columns to the nearest integers (0 or 1)
    binary_columns = [col for col in X_resampled.columns if X_resampled[col].nunique() == 2]
    X_resampled[binary_columns] = np.round(X_resampled[binary_columns])
    unique_values_1 = X_resampled['SubstanceUseTrajectory_1.0'].unique()
    unique_values_2 = X_resampled['SubstanceUseTrajectory_2.0'].unique()
    # print(unique_values_1, unique_values_2)
    # exit()
    return X_resampled, y_resampled


def apply_smote_nc(X_train, y_train, categorical_features_indices):
    from imblearn.over_sampling import SMOTENC

    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def handle_family_clusters(df):
    family_counts = df['FamilyID'].value_counts()
    df['InFamilyCluster'] = df['FamilyID'].apply(lambda x: int(family_counts[x] > 1) if pd.notnull(x) else 0)

    df.drop("FamilyID", axis=1, inplace=True)

    logger.info("Family clusters handled.")

    return df


def initial_cleaning(df, features, target):
    df.drop("ID", axis=1, inplace=True)

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

    # Replace -0 with 0 in the DataFrame
    df = df.replace(-0, 0)

    logger.info("Initial cleaning and feature engineering completed.")

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
    plt.savefig(f"results/new_data/QQ Plot of {feature} after StandardScalar")
    # plt.show()


def save_preprocessed_data(df, file_path, tag):
    df.to_csv(file_path, index=False)
    logger.info(f"{tag} saved to {file_path}.")
