import logging

from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df, target):

    try:
        logger.info("Starting data preprocessing ...")

        # Convert Sex to Is_Male binary column
        df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

        # Drop rows where the target variable is missing
        df = df.dropna(subset=[target])

        # Separate the target variable
        outcome = df[target]
        df = df.drop(columns=[target])

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


def split_data(df, outcome):
    """Data Splitting Pipeline."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(df, df[outcome]):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = df[outcome].iloc[train_idx], df[outcome].iloc[test_idx]

    return X_train, X_test, y_train, y_test


def imputation_pipeline(df):
    """Imputation Pipeline."""
    categorical_features = ['Race']
    numeric_features = [col for col in df.columns if col not in categorical_features + ['AntisocialTrajectory', 'Sex']]

    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    logger.info(f"Dataframe shape after one-hot encoding: {df.shape}")
    print()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def balance_data(X_train, y_train):
    """Data Balancing Pipeline."""
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    return X_resampled, y_resampled
