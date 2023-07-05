from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import ADASYN
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Data_GxE_on_EXT_Trajectories (new).csv')

# Initialize the imputer
imputer = KNNImputer(n_neighbors=4)

# Perform the imputation
df_imputed = imputer.fit_transform(df)

# Convert back to pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Convert Sex to binary (1 for Male, 0 for Female)
df_imputed['Is_Male'] = (df_imputed['Sex'] == -0.5).astype(int)

# Define your X and y
X = df_imputed.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex'], axis=1)
y = df_imputed['AntisocialTrajectory'].astype(int) - 1  # Subtract 1 from the labels

# Standardize features
cols_to_scale = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Handling imbalanced data with ADASYN
adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# List of models
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Logistic Regression", LogisticRegression(max_iter=1000, multi_class='multinomial')),
    ("SVC", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("Extra Trees", ExtraTreesClassifier()),
    ("XGBoost", xgb.XGBClassifier(eval_metric='mlogloss')),
    ("LightGBM", lgb.LGBMClassifier())
]

# Create pairs of interaction terms
interaction_terms = list(itertools.combinations(X.columns, 2))

# Open file for writing results incrementally
with open('interaction_terms_evaluation.csv', 'w') as file:

    # Write header
    file.write("Model,Interaction_Term,Score\n")

    # Evaluate each model with each interaction term
    for model_name, model in tqdm(models, desc='Models', position=0, leave=True):  # tqdm to track progress
        for term in tqdm(interaction_terms, desc=f'{model_name} Interaction Terms', position=1, leave=False):
            X_train_interaction = X_resampled.copy()

            # Create the interaction term
            interaction_name = f"{term[0]}_x_{term[1]}"
            X_train_interaction[interaction_name] = X_train_interaction[term[0]] * X_train_interaction[term[1]]

            try:
                # Evaluate the model with cross-validation
                score = np.mean(cross_val_score(model, X_train_interaction, y_resampled, cv=5))

                # Write the result to file
                file.write(f"{model_name},{interaction_name},{score}\n")

            except Exception as e:
                # Write error to file
                file.write(f"{model_name},{interaction_name},ERROR: {str(e)}\n")

print("\nEvaluation Complete!")
