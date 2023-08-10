# AST with KNN and SMOTE

import itertools
import warnings

import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, make_scorer, precision_score, recall_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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

# Initialize the MICE imputer
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

# List of models
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Extra Trees", ExtraTreesClassifier()),
    ("LightGBM", lgb.LGBMClassifier())
]

# Create pairs of interaction terms
interaction_terms = list(itertools.combinations(X.columns, 2))

# Open file for writing results incrementally
with open('AST_KNN_SMOTE.csv', 'w') as file:
    # Write header
    file.write("Model,Interaction_Term,Precision,Recall,F1,ROC_AUC,Accuracy,Log_Loss,Custom_Score\n")

    # Evaluate each model with each interaction term
    for model_name, model in tqdm(models, desc='Models', position=0, leave=True):  # tqdm to track progress
        for term in tqdm(interaction_terms, desc=f'{model_name} Interaction Terms', position=1, leave=False):
            X_train_interaction = X_resampled.copy()
            X_test_interaction = X_test_imputed.copy()

            # Create the interaction term
            interaction_name = f"{term[0]}_x_{term[1]}"
            X_train_interaction[interaction_name] = X_train_interaction[term[0]] * X_train_interaction[term[1]]
            X_test_interaction[interaction_name] = X_test_interaction[term[0]] * X_test_interaction[term[1]]

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

                # Write the result to file
                file.write(f"{model_name},{interaction_name},{precision},{recall},{f1},{roc_auc},{accuracy},{log_loss_val},{custom}\n")

            except Exception as e:
                # Write error to file
                file.write(f"{model_name},{interaction_name},ERROR: {str(e)}\n")

print("\n\nEvaluation Complete!")
