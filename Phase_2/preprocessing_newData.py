import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from scipy import stats
import logging

from config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from project_scripts import get_data_path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_PATH_new = get_data_path("Data_GxE_on_EXT_trajectories_new.csv")
DATA_PATH_old = get_data_path("Data_GxE_on_EXT_trajectories_old.csv")


def load_new_data():
    df = pd.read_csv(DATA_PATH_new)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

    return df


def load_old_data():
    df = pd.read_csv(DATA_PATH_old)
    if df.empty:
        raise ValueError("Data is empty or not loaded properly.")
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

    return df


def apply_boxcox_transformation(df, features):
    transformed_df = df.copy()
    for feature in features:
        min_value = transformed_df[feature].min()
        offset = (-min_value + 1) if min_value <= 0 else 0
        # Ensure all values are positive
        if (transformed_df[feature] + offset).min() <= 0:
            logger.error(f"Non-positive values found in {feature} after offset.")
            continue
        try:
            transformed, _ = stats.boxcox(transformed_df[feature] + offset)
            transformed_df[feature] = transformed
        except Exception as e:
            logger.error(f"Error in Box-Cox transformation for {feature}: {e}")
    return transformed_df


def handle_categorical_variables(df):
    categorical_features = ['Race']  # Update this list as per your actual columns
    existing_features = [col for col in categorical_features if col in df.columns]
    if not existing_features:
        logger.warning("No categorical features found for one-hot encoding.")
        return df
    categorical_transformer = OneHotEncoder(drop='first')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, existing_features)
        ],
        remainder='passthrough'
    )
    return pd.DataFrame(preprocessor.fit_transform(df), columns=preprocessor.get_feature_names_out())


def normalize_continuous_variables(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def impute_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def handle_family_clusters(df):
    family_counts = df['FamilyID'].value_counts()
    df['InFamilyCluster'] = df['FamilyID'].apply(lambda x: family_counts[x] > 1 if pd.notnull(x) else False)
    return df


def initial_cleaning(df, features, target):
    logger.info("Initiating Primary preprocessing...\n")
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handling outliers for PGS using IQR
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) | (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]

    # Drop rows where the target variable is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target])
    rows_after_dropping = len(df)
    logger.info(f"Dropped {initial_rows - rows_after_dropping} rows due to missing target values...\n")

    # Feature Engineering
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']
    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    logger.info("Data Cleaning completed successfully...\n")

    return df, feature_cols


def split_data(X, y):
    logger.info("Splitting data...\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split successfully...\n")
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


def save_preprocessed_data(df, file_path, target):
    if target == "AntisocialTrajectory":
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)


def preprocessing_pipeline(data_path, features, target, file_path_to_save):
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
    df = pd.read_csv(data_path)
    logger.info("Data loaded successfully.")
    df.drop("ID", axis=1, inplace=True)

    # Initial cleaning and feature engineering
    df, feature_cols = initial_cleaning(df, features, target)
    logger.info("Initial cleaning and feature engineering completed.")

    # Handle family clusters
    df = handle_family_clusters(df)
    logger.info("Family clusters handled.")
    df.drop("FamilyID", axis=1, inplace=True)

    # Apply Box-Cox transformation
    continuous_features = ["DelinquentPeer", "SchoolConnect", "NeighborConnect", "ParentalWarmth"]
    df = apply_boxcox_transformation(df, continuous_features)  # specify the continuous features needing transformation
    logger.info("Box-Cox transformation applied.")

    # Handle categorical variables
    df = handle_categorical_variables(df)
    logger.info("Categorical variables handled.")

    # Normalize continuous variables
    df = normalize_continuous_variables(df, feature_cols)
    logger.info("Continuous variables normalized.")

    # Impute missing values
    df = impute_missing_values(df)
    logger.info("Missing values imputed.")

    # Save the preprocessed data
    save_preprocessed_data(df, file_path_to_save, target)
    logger.info(f"Preprocessed data saved to {file_path_to_save}.")


def main(TARGET):
    # Assigning features based on the outcome.
    if TARGET == "AntisocialTrajectory":
        FEATURES = FEATURES_FOR_AST
        SAVE_PATH = 'preprocessed_data_old_AST.csv'
    else:
        FEATURES = FEATURES_FOR_SUT
        SAVE_PATH = 'preprocessed_data_old_SUT.csv'

    preprocessing_pipeline(DATA_PATH_old, FEATURES, TARGET, SAVE_PATH)


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(target_2)
