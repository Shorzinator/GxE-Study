import logging
import os
import warnings

import joblib
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

num_cores = 2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the data
df = pd.read_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\Data_GxE_on_EXT_trajectories.csv")

if not df.empty:
    logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
else:
    raise ValueError("Data is empty or not loaded properly.")

# Convert Sex to binary (1 for male, 0 for female)
df["Is_Male"] = (df["Sex"] == -0.5).astype(int)

# Define the feature columns and target columns
feature_cols = ["Race", "PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect",
                "ParentalWarmth", "Is_Male"]

df["AST_1_vs_4"] = (df["AntisocialTrajectory"] == 1).astype(int)
df["AST_2_vs_4"] = (df["AntisocialTrajectory"] == 2).astype(int)
df["AST_3_vs_4"] = (df["AntisocialTrajectory"] == 3).astype(int)

# Splitting columns based on type
numerical_cols = ["PolygenicScoreEXT", "Age", "DelinquentPeer", "SchoolConnect", "NeighborConnect",
                  "ParentalWarmth", "Is_Male"]
categorical_cols = ["Race"]

# Create Transformers
numerical_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=6)),
    ("standard_scalar", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Define the hyperparameter space for RandomizedSearchCV for Logistic Regression
param_space = {
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'classifier__scale_pos_weight': [1, (df["AST_1_vs_4"].sum() / (df.shape[0] - df["AST_1_vs_4"].sum()))]  # for imbalanced datasets
}


# Metrics Calculation
def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred)
    }


def run_model_for_outcome(outcome, results_df, features, df):
    logger.info(f"Processing outcome: {outcome}")
    targets = df[[outcome]]
    targets = targets.dropna(subset=[outcome])

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in stratified_split.split(features, targets):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        # save these splits for future use
        X_train.to_csv(f"X_train_{outcome}.csv", index=False)
        X_test.to_csv(f"X_test_{outcome}.csv", index=False)
        y_train.to_csv(f"y_train_{outcome}.csv", index=False)
        y_test.to_csv(f"y_test_{outcome}.csv", index=False)

        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smoteenn', SMOTEENN(random_state=0)),
            ('classifier',
             xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
                               random_state=0))
        ])

        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

        randomized_search = RandomizedSearchCV(pipeline, param_distributions=param_space, n_iter=5,
                                               scoring='roc_auc', cv=skf, n_jobs=num_cores, verbose=1,
                                               random_state=0, return_train_score=True)

        randomized_search.fit(X_train, y_train[outcome])

        # Save all results to dataframe
        cv_results_df = pd.DataFrame(randomized_search.cv_results_)
        cv_results_df.to_csv(f"results_XGBoost_{outcome}.csv")

        # Using the best estimator found
        best_pipeline = randomized_search.best_estimator_
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        train_metrics = calculate_metrics(y_train[outcome], y_train_pred)
        test_metrics = calculate_metrics(y_test[outcome], y_test_pred)

        for metric_name, metric_value in train_metrics.items():
            results_df.loc[outcome, f"Train {metric_name}"] = metric_value

        for metric_name, metric_value in test_metrics.items():
            results_df.loc[outcome, f"Test {metric_name}"] = metric_value

        # Save the best model using joblib
        joblib.dump(best_pipeline, f"best_model_XGBoost_{outcome}.joblib")

    return results_df


# Create an empty results dataframe
results_df = pd.DataFrame()

# Process each outcome
for outcome in ["AST_1_vs_4", "AST_2_vs_4", "AST_3_vs_4"]:
    features = df[feature_cols].copy()
    results_df = run_model_for_outcome(outcome, results_df, features, df)

# Save results to a CSV
output_dir = os.path.dirname(os.path.abspath(__file__))
results_df.to_csv(os.path.join(output_dir, "combined_results_RandomizedSearch_SMOTEEN_StratifiedKFold_xgb.csv"))
