import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Phase_1.config import *
from Phase_1.project_scripts.utility.model_utils import add_interaction_terms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ap_with_it(X, y, feature_pair, features):
    """
    Applies preprocessing steps including imputation, one-hot encoding, interaction terms, scaling, and balancing
    on training, validation, and testing data.

    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param feature_pair: list, list of tuples representing the feature pairs for an interaction terms
    :param features: list, list of feature names
    :return: DataFrames, preprocessed training, validation, and testing data
    """
    logger.info("Starting preprocessing with interaction terms...\n")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply imputation and one-hot encoding
    impute = imputation_pipeline(features)
    X_train_imputed = imputation_applier(impute, X_train, features, fit=True)
    # X_val_imputed = imputation_applier(impute, X_val, features)
    X_test_imputed = imputation_applier(impute, X_test, features)

    # Generate interaction terms
    X_train_final = add_interaction_terms(X_train_imputed, feature_pair)
    # X_val_final = add_interaction_terms(X_val_imputed, feature_pair)
    X_test_final = add_interaction_terms(X_test_imputed, feature_pair)

    # Apply scaling
    X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(X_train_final, X_test_final)

    # Balance data
    X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train)

    logger.info("Preprocessing with interaction terms completed successfully.\n")
    return X_train_resampled, y_train_resampled, X_test_final, y_test


def ap_without_it(X, y, features):
    """
    Applies preprocessing steps including imputation, one-hot encoding, scaling, and balancing
    on training, validation, and testing data.

    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param features: list, list of feature names
    :return: DataFrames, preprocessed training, validation, and testing data
    """
    logger.info("Starting preprocessing without interaction terms...\n")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply imputation and one-hot encoding
    impute = imputation_pipeline(features)

    X_train_final = imputation_applier(impute, X_train, features)
    # X_val_final = imputation_applier(impute, X_val, features)
    X_test_final = imputation_applier(impute, X_test, features)

    # Apply scaling
    X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(X_train_final, X_test_final)

    # Balance data
    X_train_resampled, y_train_resampled = balance_data(X_train_imputed_scaled, y_train)

    logger.info("Preprocessing without interaction terms completed successfully...\n")
    return X_train_resampled, y_train_resampled, X_test_final, y_test


def ap_without_it_genetic(X, y, features):
    """
    Applies preprocessing steps including imputation, one-hot encoding, scaling, and balancing
    on training, validation, and testing data.

    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param features: list, list of feature names
    :return: DataFrames, preprocessed training, validation, and testing data
    """
    logger.info("Starting preprocessing without interaction terms...\n")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply imputation and one-hot encoding
    impute = imputation_pipeline(features)

    X_train_final = imputation_applier(impute, X_train, features)
    X_test_final = imputation_applier(impute, X_test, features)

    # Apply scaling
    X_train_imputed_scaled, X_test_imputed_scaled = scaling_applier(X_train_final, X_test_final)

    # Balance data
    X_train_resampled, y_train_resampled = X_train_imputed_scaled, y_train

    logger.info("Preprocessing without interaction terms completed successfully...\n")
    return X_train_resampled, y_train_resampled, X_test_imputed_scaled, y_test


def split_data(X, y):
    """
    Splits the data into training, validation, and testing sets.

    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :return: DataFrames, training, validation, and testing data
    """
    logger.info("Splitting data...\n")

    # First, split into train + validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Data split successfully...\n")

    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


def imputation_pipeline(numerical_features):
    """
    Creates an imputation pipeline.

    :param numerical_features: List, list of numerical feature names
    :return: ColumnTransformer, imputation transformer
    """
    numeric_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=10))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ])

    return preprocessor


def imputation_applier(impute, df, feature_names, fit=True):
    """
    Creates an imputation pipeline and applies imputation on the data.

    :param df: Dataframe
    :param impute: ColumnTransformer, imputation transformer
    :param feature_names: list, list of feature names
    :param fit: bool, whether to fit the transformer
    :return: DataFrame, imputed data
    """

    logger.info("Applying imputation...\n")

    if fit:
        input_imputed = impute.fit_transform(df)
    else:
        input_imputed = impute.transform(df)

    X_train_imputed = pd.DataFrame(input_imputed, columns=feature_names)

    logger.info("Imputation applied successfully...\n")
    return X_train_imputed


def scaling_applier(X_train_imputed, X_test_imputed):
    """
    Applies scaling on the data.

    :param X_train_imputed: DataFrame, imputed training data
    # :param X_val_imputed: DataFrame, imputed validation data
    :param X_test_imputed: DataFrame, imputed testing data
    :return: DataFrames, scaled training, validation, and testing data
    """
    logger.info("Applying scaling...\n")

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data and transform
    X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
    # X_val_imputed_scaled = scaler.transform(X_val_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)

    # Convert back to DataFrame and retain column names
    X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled, columns=X_train_imputed.columns)
    # X_val_imputed_scaled = pd.DataFrame(X_val_imputed_scaled, columns=X_val_imputed.columns)
    X_test_imputed_scaled = pd.DataFrame(X_test_imputed_scaled, columns=X_test_imputed.columns)

    logger.info("Scaling applied successfully...\n")
    return X_train_imputed_scaled, X_test_imputed_scaled


def balance_data(X_train, y_train):
    """
    Balances the data using SMOTE.

    :param X_train: DataFrame, training data
    :param y_train: Series, training target variable
    :return: DataFrames, resampled training data and target variable
    """
    logger.info("Balancing data...\n")

    smote = SMOTE(random_state=0, k_neighbors=10, sampling_strategy="not majority")
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Print distribution after SMOTE
    # logger.info("After SMOTE:")
    # for label, count in y_resampled.value_counts().items():
    #     logger.info(f"Class {label}: {count}")

    # logger.info(f"Rows before balancing: {initial_size}. Rows after: {len(X_resampled)}.\n")

    logger.info("Data balanced successfully...\n")
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

    feature_cols_without_race = FEATURES

    independent = df[feature_cols_without_race]
    return independent, dependent


def preprocess_sut_ovr(df, features):
    logger.info("Starting data preprocessing...\n")

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    initial_size = len(df)
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) |
              (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]
    logger.info(f"Rows before handling outliers: {initial_size}. Rows after: {len(df)}.\n")

    # Separate the target variable
    outcome = df['SubstanceUseTrajectory']

    # Drop rows where the target variable is missing
    df.dropna(subset=['SubstanceUseTrajectory'])

    # Adding these terms to tackle cohort differences in Psychosocial Environments
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']

    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    df = df[feature_cols]

    # Create datasets for binary classification with Typical Use as baseline
    datasets = {
        "1_vs_3": (df[outcome.isin([1, 3])].copy(), outcome[outcome.isin([1, 3])].copy()),
        "2_vs_3": (df[outcome.isin([2, 3])].copy(), outcome[outcome.isin([2, 3])].copy())
    }

    logger.info("Data preprocessing completed successfully.\n")
    return datasets, feature_cols


def preprocess_ast_ovr(df, features):
    logger.info("Starting data preprocessing...\n")

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    initial_size = len(df)
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) |
              (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]
    logger.info(f"Rows before handling outliers: {initial_size}. Rows after: {len(df)}.\n")

    # Drop rows where the target variable is missing
    df = df.dropna(subset=["AntisocialTrajectory"])

    # Adding these terms to tackle cohort differences in Psychosocial Environments
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']

    # Separate the target variable
    outcome = df["AntisocialTrajectory"]

    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    df = df[feature_cols]

    # Create datasets for each binary classification task
    datasets = {
        "1_vs_4": (df[outcome.isin([1, 4])].copy(), outcome[outcome.isin([1, 4])].copy()),
        "2_vs_4": (df[outcome.isin([2, 4])].copy(), outcome[outcome.isin([2, 4])].copy()),
        "3_vs_4": (df[outcome.isin([3, 4])].copy(), outcome[outcome.isin([3, 4])].copy())
    }

    logger.info("Data preprocessing completed successfully.\n")
    return datasets, feature_cols


def preprocess_for_genetic_model(X, y):
    """
    Preprocesses the data for the genetic model (Model 3).

    :param X: Features
    :param y: outcome
    :return: Preprocessed features and outcomes
    """
    logger.info("Starting preprocessing for genetic model...\n")

    # Impute missing values in y with column means
    y = y.apply(lambda col: col.fillna(col.mean()))

    logger.info("Primary preprocessing for Model 3 completed successfully.\n")

    return X, y
