# AST with KNN and SMOTE with all InteractionTerms

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def custom_score(y_true, y_score, model, X):
    y_pred = model.predict(X)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    log_loss_val = log_loss(y_true, y_score)

    weights = {'precision': 0.3, 'recall': 0.2, 'f1': 0.3, 'roc_auc': 0.2, 'accuracy': 0.5, 'log_loss_val':0.2}
    score = weights['precision'] * precision + weights['recall'] * recall + weights['f1'] * f1 + weights['roc_auc'] * roc_auc + weights['accuracy'] * accuracy + weights['log_loss_val'] * log_loss_val

    return score

custom_scorer = make_scorer(custom_score, needs_proba=True, greater_is_better=True)

# Load the dataset
df = pd.read_csv('Data_GxE_on_EXT_trajectories.csv')

# Convert Sex to binary (1 for Male, 0 for Female)
df['Is_Male'] = (df['Sex'] == -0.5).astype(int)

# Clean the dataframe
df = df.dropna(subset=['AntisocialTrajectory'])

# Define your X and y
X = df.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex'], axis=1)
y = df['AntisocialTrajectory'].astype(int)

# Standardize features
cols_to_scale = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=6)

# Perform the imputation on training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the testing data
X_test_imputed = imputer.transform(X_test)

# Convert imputed arrays back to pandas DataFrames
X_columns = X.columns
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_columns)

# Handling imbalanced data with SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)

# Convert resampled arrays back to pandas DataFrames
X_resampled = pd.DataFrame(X_resampled, columns=X_columns)

# Create DataFrame for interaction terms
interaction_df_train = pd.DataFrame(index=X_resampled.index)
interaction_df_test = pd.DataFrame(index=X_test_imputed.index)

# Create pairs of interaction terms
interaction_terms = list(itertools.combinations(X.columns, 2))

# Create all interaction terms
for term in interaction_terms:
    interaction_name = f"{term[0]}_x_{term[1]}"
    interaction_df_train[interaction_name] = X_resampled[term[0]] * X_resampled[term[1]]
    interaction_df_test[interaction_name] = X_test_imputed[term[0]] * X_test_imputed[term[1]]

# Add interaction terms to the original DataFrames
X_train_interaction = pd.concat([X_resampled, interaction_df_train], axis=1)
X_test_interaction = pd.concat([X_test_imputed, interaction_df_test], axis=1)

# List of models
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Extra Trees", ExtraTreesClassifier()),
    ("LightGBM", lgb.LGBMClassifier())
]

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'Accuracy', 'Log_Loss', 'Custom_Score'])

# Evaluate each model with all interaction terms
for model_name, model in tqdm(models, desc='Models', position=0, leave=True):  # tqdm to track progress
    interaction_df_train = pd.DataFrame(index=X_resampled.index)
    interaction_df_test = pd.DataFrame(index=X_test_imputed.index)
    
    # Create all interaction terms
    for term in interaction_terms:
        interaction_name = f"{term[0]}_x_{term[1]}"
        interaction_df_train[interaction_name] = X_resampled[term[0]] * X_resampled[term[1]]
        interaction_df_test[interaction_name] = X_test_imputed[term[0]] * X_test_imputed[term[1]]

    # Add interaction terms to the original DataFrames
    X_train_interaction = pd.concat([X_resampled, interaction_df_train], axis=1)
    X_test_interaction = pd.concat([X_test_imputed, interaction_df_test], axis=1)

    try:
        # Fit the model on the resampled training data
        model.fit(X_train_interaction, y_resampled)

        # Make predictions on the test data
        y_pred = model.predict(X_test_interaction)

        # Calculate metrics
        y_pred_proba = model.predict_proba(X_test_interaction)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        log_loss_val = log_loss(y_test, y_pred_proba)

        # Get custom score
        custom = custom_score(y_test, y_pred_proba, model, X_test_interaction)

        # Append the results to the DataFrame
        results_df = results_df.append({
            'Model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc,
            'Accuracy': accuracy,
            'Log_Loss': log_loss_val,
            'Custom_Score': custom
        }, ignore_index=True)

    except Exception as e:
        # Append error to DataFrame
        results_df = results_df.append({
            'Model': model_name,
            'Precision': 'ERROR',
            'Recall': 'ERROR',
            'F1': 'ERROR',
            'ROC_AUC': 'ERROR',
            'Accuracy': 'ERROR',
            'Log_Loss': 'ERROR',
            'Custom_Score': str(e)
        }, ignore_index=True)

    print(f"\n\nEvaluation Complete for {model_name}!")

# Save results to a CSV file
results_df.to_csv('AST_KNN_SMOTE_allInteractionTerms.csv', index=False)
