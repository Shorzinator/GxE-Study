import logging

from Phase_1.project_scripts.preprocessing.preprocessing import balance_data, imputation_applier, imputation_pipeline, \
    split_data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def primary_preprocessing_mrf(df, features, target):
    """
    Preprocess the data for Markov Random Field modeling.
    """
    logger.info("Initiating Primary preprocessing...\n")
    # Data Cleaning
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) |
              (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]

    # Check for NaN values post outlier handling
    # check_nan_values(df, "after outlier handling")

    # Drop rows where the target variable is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target])
    rows_after_dropping = len(df)
    logger.info(f"\nDropped {initial_rows - rows_after_dropping} rows due to missing target values...\n")

    # Feature Engineering
    df['PolygenicScoreEXT_x_Is_Male'] = df['PolygenicScoreEXT'] * df['Is_Male']
    df['PolygenicScoreEXT_x_Age'] = df['PolygenicScoreEXT'] * df['Age']

    feature_cols = features + ['PolygenicScoreEXT_x_Is_Male', 'PolygenicScoreEXT_x_Age']

    logger.debug(f"First few rows after primary preprocessing:\n{df.head()}")  # ADDED

    logger.info("Primary preprocessing completed successfully...\n")
    return df, feature_cols


def secondary_preprocessing(X, y, features):
    """
    Applies preprocessing steps on training, validation, and testing data.
    """
    logger.info("Initiating Secondary preprocessing...\n")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply imputation and one-hot encoding
    impute = imputation_pipeline(features)
    X_train_imputed = imputation_applier(impute, X_train, features, fit=True)
    # X_val_imputed = imputation_applier(impute, X_val, features)
    X_test_imputed = imputation_applier(impute, X_test, features)

    # Balance data
    X_train_resampled, y_train_resampled = balance_data(X_train_imputed, y_train)

    logger.info("Secondary preprocessing completed successfully...\n")
    return X_train_resampled, y_train_resampled, X_test_imputed, y_test
