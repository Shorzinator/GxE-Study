from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Data_GxE_on_EXT_Trajectories (new).csv')

# List of imputers
imputers = [("KNNImputer", KNNImputer(n_neighbors=4)), ("IterativeImputer", IterativeImputer())]

# List of oversampling techniques
oversamplers = [("ADASYN", ADASYN(sampling_strategy='minority', random_state=42)), ("SMOTE", SMOTE(sampling_strategy='minority', random_state=42))]

# Temporary list to store the result dictionaries
temp_results = []

# Iterate over imputers and oversamplers
for imputer_name, imputer in imputers:
    for oversampler_name, oversampler in oversamplers:
        
        # Perform the imputation
        df_imputed = imputer.fit_transform(df)
        
        # Convert back to pandas DataFrame
        df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

        # Convert Sex to binary (1 for Male, 0 for Female)
        df_imputed['Is_Male'] = (df_imputed['Sex'] == -0.5).astype(int)

        # Define your X and y
        X = df_imputed.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex'], axis=1)
        y = df_imputed['SubstanceUseTrajectory'].astype(int) - 1

        # Standardize features
        cols_to_scale = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

        # Handling imbalanced data
        X_resampled, y_resampled = oversampler.fit_resample(X, y)

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

        # Evaluate each model with each interaction term
        for model_name, model in tqdm(models, desc='Models', position=0, leave=True):
            for term in tqdm(interaction_terms, desc=f'{model_name} Interaction Terms', position=1, leave=False):
                X_train_interaction = X_resampled.copy()

                # Create the interaction term
                interaction_name = f"{term[0]}_x_{term[1]}"
                X_train_interaction[interaction_name] = X_train_interaction[term[0]] * X_train_interaction[term[1]]

                # Evaluate the model with cross validation
                score = np.mean(cross_val_score(model, X_train_interaction, y_resampled, cv=5))
                
                # Get Feature Importances
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                else:
                    feature_importances = None

                # Add the result dictionary to the temporary list
                temp_results.append({'Imputer': imputer_name, 'Oversampler': oversampler_name, 'Model': model_name, 'Interaction_Term': interaction_name, 'Score': score, 'Feature_Importances': feature_importances})

# Convert the list of dictionaries to DataFrame
results = pd.DataFrame(temp_results)

# Save the results to a CSV file
results.to_csv('interaction_terms_evaluation_SUB.csv', index=False)

print("Evaluation Complete!")
