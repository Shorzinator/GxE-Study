import logging

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(df, outcome_series):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(df, outcome_series):
        X_train = df.iloc[train_idx].reset_index(drop=True)
        X_test = df.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = outcome_series.iloc[train_idx], outcome_series.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def imputation_pipeline(df):
    """Imputation Pipeline."""
    categorical_features = ['Race']
    numeric_features = [col for col in df.columns if col not in categorical_features + ['AntisocialTrajectory', 'Sex']]

    numeric_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=5))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encoding
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def scaling_pipeline(df):
    """Scaling Pipeline."""
    numeric_features = [col for col in df.columns if col not in ['Race', 'AntisocialTrajectory', 'Sex', 'Is_Male']]  # Added 'Is_Male' to exclude it from scaling
    categorical_features = ['Race']

    scaler = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_features)
        ],
        remainder='passthrough'  # Non-scaled features are passed through without any transformation
    )

    return preprocessor


def balance_data(X_train, y_train):
    """Data Balancing Pipeline."""
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def preprocess_multinomial(df, target):
    logger.info("Starting data preprocessing for multinomial logistic regression ...")

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

    # Handle outliers for PolygenicScoreEXT using IQR
    Q1 = df['PolygenicScoreEXT'].quantile(0.25)
    Q3 = df['PolygenicScoreEXT'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['PolygenicScoreEXT'] < (Q1 - 1.5 * IQR)) | (df['PolygenicScoreEXT'] > (Q3 + 1.5 * IQR)))]

    # Drop rows where the target variable is missing
    df = df.dropna(subset=[target])

    # Separate the target variable
    outcome = df[target]
    feature_cols = ["Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect",
                    "NeighborConnect", "ParentalWarmth", "Is_Male"]
    df = df[feature_cols]

    logger.info("Data preprocessing for multinomial logistic regression completed successfully.")

    return df, outcome


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
