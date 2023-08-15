import logging
import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Phase_1.project_scripts.utility.path_utils import get_path_from_root

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
    numerical_features = ["PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect",
                          "ParentalWarmth", "Is_Male"]
    # categorical_features = ["Race"]
    categorical_features = []

    numeric_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=10))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
            # ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def imputation_applier(impute, X):
    initial_size_train = len(X)
    logger.info("Applying Imputation ...\n")

    input_imputed = impute.fit_transform(X)

    """
    logger.info(f"Rows before imputing X_train: {initial_size_train}. Rows after: {len(X_train_imputed)}.")
    logger.info(f"Rows before imputing X_test: {initial_size_test}. Rows after: {len(X_test_imputed)}.\n")
    """
    feature_names = [
        "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
        "NeighborConnect", "ParentalWarmth", "Is_Male"
    ]

    X_train_imputed = pd.DataFrame(input_imputed, columns=feature_names)
    # print(X_train_imputed.columns)

    return X_train_imputed


def scaling_pipeline(transformed_features):
    """Scaling Pipeline."""

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

    logger.info("Applying scaling ...\n")

    X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)

    """
    logger.info(f"Rows before scaling X_train: {initial_size_train}. Rows after: {len(X_train_imputed_scaled)}.")
    logger.info(f"Rows before scaling X_test: {initial_size_test}. Rows after: {len(X_test_imputed_scaled)}.\n")
    """

    X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled)
    X_test_imputed_scaled = pd.DataFrame(X_test_imputed_scaled)

    return X_train_imputed_scaled, X_test_imputed_scaled


def balance_data(X_train, y_train, key):
    """Data Balancing Pipeline."""
    logger.info("Balancing data ...\n")

    initial_size = len(X_train)
    smote = SMOTE(random_state=0, k_neighbors=10, sampling_strategy="all")
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Rows before balancing: {initial_size}. Rows after: {len(X_resampled)}.\n")

    # Combine resampled data
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    # Combine original training data
    original_data = pd.concat([X_train, y_train], axis=1)

    # Assuming there is no duplication in the original dataset
    synthetic_data = resampled_data.loc[~resampled_data.index.isin(original_data.index)]

    # Save the original preprocessed data
    processed_data_path = get_path_from_root("data", "processed")
    original_data_file = os.path.join(processed_data_path, f"original_preprocessed_data_{key}.csv")
    original_data.to_csv(original_data_file, index=False)
    return X_resampled, y_resampled


def preprocess_multinomial(df, target):
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

    """
       feature_cols = [
           "Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
           "NeighborConnect", "ParentalWarmth", "Is_Male"
       ]
       """

    feature_cols_without_race = [
        "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
        "NeighborConnect", "ParentalWarmth", "Is_Male"
    ]

    independent = df[feature_cols_without_race]
    logger.info("Data preprocessing for multinomial logistic regression completed successfully.\n")
    return independent, dependent


def preprocess_ovr(df, target):
    logger.info("Starting data preprocessing for one-vs-all logistic regression...\n")

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) |
              (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]

    # Drop rows where the target variable is missing
    df = df.dropna(subset=[target])

    # Separate the target variable
    outcome = df[target]

    """
    feature_cols = [
        "Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
        "NeighborConnect", "ParentalWarmth", "Is_Male"
    ]
    """

    feature_cols_without_race = [
        "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
        "NeighborConnect", "ParentalWarmth", "Is_Male"
    ]

    df = df[feature_cols_without_race]

    # Create datasets for each binary classification task
    datasets = {
        "1_vs_4": (df[outcome.isin([1, 4])].copy(), outcome[outcome.isin([1, 4])].copy()),
        "2_vs_4": (df[outcome.isin([2, 4])].copy(), outcome[outcome.isin([2, 4])].copy()),
        "3_vs_4": (df[outcome.isin([3, 4])].copy(), outcome[outcome.isin([3, 4])].copy())
    }

    logger.info("Data preprocessing for one-vs-all logistic regression completed successfully.\n")
    return datasets, feature_cols_without_race
