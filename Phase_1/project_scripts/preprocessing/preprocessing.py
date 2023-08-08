from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.combine import SMOTEENN
from Phase_1.project_scripts.utility.path_utils import get_path_from_root


def preprocess_data(df, target):

    # Extract the correct outcome column based on the target
    outcome = "AntisocialTrajectory" if target == "AST" else "SubstanceUseTrajectory"

    # Convert Sex to Is_Male binary column
    df["Is_Male"] = (df["Sex"] == -0.5).astype(int)
    df = df.dropna(subset=[outcome])

    processed_data_path = get_path_from_root("data", "processed", f"{target}_preprocessed.csv")
    df.to_csv(processed_data_path, index=False)

    # Create target comparison columns
    for i in [1, 2, 3]:
        df[f"{target}_{i}_vs_4"] = (df[outcome] == i).astype(int)

    return df, outcome


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
