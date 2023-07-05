import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# Load the data
# Here, you need to provide the path to your dataset file.
df = pd.read_csv('Data_GxE_on_EXT_Trajectories (new).csv')

# Initialize the imputer
imputer = KNNImputer(n_neighbors=4)

# Perform the imputation
df_imputed = imputer.fit_transform(df)

# Convert back to pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Convert Sex to binary (1 for Male, 0 for Female)
df_imputed['Is_Male'] = (df_imputed['Sex'] == -0.5).astype(int)

# Standardize features
cols_to_scale = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']
scaler = StandardScaler()
df_imputed[cols_to_scale] = scaler.fit_transform(df_imputed[cols_to_scale])

# Define your X
X = df_imputed.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex'], axis=1)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)

# Loop through the target variables (AntisocialTrajectory and SubstanceUseTrajectory)
for target_var in ['AntisocialTrajectory', 'SubstanceUseTrajectory']:
    y = df_imputed[target_var].astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversampling
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Random Forest Model with oversampled data
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_smote, y_train_smote)

    # Get feature importances from Random Forest (based on mean decrease impurity)
    rf_importances = rf_model.feature_importances_

    # Plot Random Forest feature importances (Mean Decrease in Impurity)
    plt.figure(figsize=(10, 5))
    bars = plt.barh(range(len(rf_importances)), rf_importances, align='center')
    plt.yticks(range(len(rf_importances)), [X.columns[i] for i in range(len(rf_importances))])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Random Forest Feature Importances for {target_var} (Mean Decrease in Impurity)')
    # Add the data value on top of each bar
    for bar, value in zip(bars, rf_importances):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, '{:.4f}'.format(value), va='center', ha='left')
    plt.show()

    # Get feature importances from Random Forest (based on permutation importance)
    result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = result.importances_mean

    # Plot Random Forest feature importances based on permutation importance
    plt.figure(figsize=(10, 5))
    bars = plt.barh(range(len(perm_importances)), perm_importances, align='center')
    plt.yticks(range(len(perm_importances)), [X.columns[i] for i in range(len(perm_importances))])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Random Forest Feature Importances for {target_var} (Permutation Importance)')
    # Add the data value on top of each bar
    for bar, value in zip(bars, perm_importances):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, '{:.4f}'.format(value), va='center', ha='left')
    plt.show()
