import logging

from Phase_1.project_scripts.preprocessing.preprocessing import balance_data, imputation_applier, imputation_pipeline, \
    split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_for_mrf(df, features, target):
    """
    Preprocess the data for Markov Random Field modeling.

    Args:
    - df (pd.DataFrame): The raw data.
    - features (list): List of features to consider.

    Returns:
    - pd.DataFrame: The preprocessed data.
    """
    # Data Cleaning
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) |
              (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]

    # Drop rows where the target variable is missing
    df = df.dropna(subset=[target])

    # Feature Engineering
    # Adding these terms to tackle cohort differences in Psychosocial Environments
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']

    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    return df, feature_cols


def apply_preprocessing_without_interaction_terms_mrf(X, y, features):
    """
    Applies preprocessing steps including imputation, one-hot encoding, interaction terms, scaling, and balancing
    on training, validation, and testing data.

    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param features: list, list of feature names
    :return: DataFrames, preprocessed training, validation, and testing data
    """
    logger.info("Starting preprocessing with interaction terms in MRF...\n")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Apply imputation and one-hot encoding
    impute = imputation_pipeline(features)
    X_train_imputed = imputation_applier(impute, X_train, features, fit=True)
    X_val_imputed = imputation_applier(impute, X_val, features)
    X_test_imputed = imputation_applier(impute, X_test, features)

    # Balance data
    X_train_resampled, y_train_resampled = balance_data(X_train_imputed, y_train)

    logger.info("Preprocessing with interaction terms completed successfully.\n")
    return X_train_resampled, y_train_resampled, X_val_imputed, y_val, X_test_imputed, y_test
