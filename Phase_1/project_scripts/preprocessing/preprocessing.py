import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(df, outcome_series):
    logger.info("Splitting data ...\n")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(df, outcome_series):
        X_train = df.iloc[train_idx].reset_index(drop=True)
        X_test = df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = outcome_series.iloc[train_idx], outcome_series.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def imputation_pipeline():
    """Imputation Pipeline."""
    logger.info("Applying Imputation ...\n")
    numerical_features = ["PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect",
                          "ParentalWarmth", "Is_Male"]
    categorical_features = ["Race"]

    numeric_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=10))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def imputation_applier(impute, X):
    initial_size_train = len(X)

    input_imputed = impute.fit_transform(X)

    """
    logger.info(f"Rows before imputing X_train: {initial_size_train}. Rows after: {len(X_train_imputed)}.")
    logger.info(f"Rows before imputing X_test: {initial_size_test}. Rows after: {len(X_test_imputed)}.\n")
    """
    feature_names = (impute.named_transformers_['num'].named_steps['impute'].get_feature_names_out().tolist() +
                     impute.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out().tolist())

    X_train_imputed = pd.DataFrame(input_imputed, columns=feature_names)
    print(X_train_imputed.columns)
    # print(feature_names)
    return X_train_imputed


def scaling_pipeline(transformed_features):
    """Scaling Pipeline."""
    logger.info("Applying scaling ...\n")

    scaler = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, transformed_features),
        ],
        remainder='passthrough'  # Non-scaled features are passed through without any transformation
    )

    return preprocessor


def scaling_applier(scaler, X_train_imputed, X_test_imputed):

    initial_size_train = len(X_train_imputed)
    initial_size_test = len(X_test_imputed)

    X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)

    """
    logger.info(f"Rows before scaling X_train: {initial_size_train}. Rows after: {len(X_train_imputed_scaled)}.")
    logger.info(f"Rows before scaling X_test: {initial_size_test}. Rows after: {len(X_test_imputed_scaled)}.\n")
    """

    X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled)
    X_test_imputed_scaled = pd.DataFrame(X_test_imputed_scaled)

    return X_train_imputed_scaled, X_test_imputed_scaled


def balance_data(X_train, y_train):
    """Data Balancing Pipeline."""
    logger.info("Balancing data ...\n")

    initial_size = len(X_train)
    smote = SMOTE(random_state=0, k_neighbors=10, sampling_strategy="all", n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Rows before balancing: {initial_size}. Rows after: {len(X_resampled)}.\n")

    return X_resampled, y_resampled


def preprocess_multinomial(df, target):
    logger.info("Starting data preprocessing for multinomial logistic regression ...\n")

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    initial_size = len(df)
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) | (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]
    logger.info(f"Rows before handling outliers: {initial_size}. Rows after: {len(df)}.\n")

    # Drop rows where the target variable is missing
    initial_size = len(df)
    df = df.dropna(subset=[target])
    logger.info(f"Rows before dropping missing values in target: {initial_size}. Rows after: {len(df)}.\n")

    # Separate the target variable
    dependent = df[target]
    feature_cols = ["Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
                    "NeighborConnect", "ParentalWarmth", "Is_Male"]

    independent = df[feature_cols]
    logger.info("Data preprocessing for multinomial logistic regression completed successfully.\n")
    return independent, dependent


def preprocess_ovr(df, target):
    try:
        logger.info("Starting data preprocessing ...")

        # Convert Sex to Is_Male binary column
        df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

        # Drop rows where the target variable is missing
        df = df.dropna(subset=[target])

        # Separate the target variable
        outcome = df[target]
        feature_cols = ["Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect",
                        "ParentalWarmth", "Is_Male"]
        df = df[feature_cols]

        print(df.columns)

        # Create datasets for each binary classification task
        df_1_vs_4 = df[outcome.isin([1, 4])].copy()
        outcome_1_vs_4 = outcome[outcome.isin([1, 4])].copy()

        df_2_vs_4 = df[outcome.isin([2, 4])].copy()
        outcome_2_vs_4 = outcome[outcome.isin([2, 4])].copy()

        df_3_vs_4 = df[outcome.isin([3, 4])].copy()
        outcome_3_vs_4 = outcome[outcome.isin([3, 4])].copy()

        logger.info("Data preprocessing completed successfully.")

        return {
            "1_vs_4": (df_1_vs_4, outcome_1_vs_4),
            "2_vs_4": (df_2_vs_4, outcome_2_vs_4),
            "3_vs_4": (df_3_vs_4, outcome_3_vs_4)
        }
    except Exception as e:
        logger.error(f"Error occurred during data preprocessing: {str(e)}")
        return None
