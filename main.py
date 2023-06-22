import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load the data
df = pd.read_csv('Data_GxE_on_EXT_Trajectories (new).csv')

# initialize the imputer
imputer = KNNImputer(n_neighbors=4)

# perform the imputation
df_imputed = imputer.fit_transform(df)

# convert back to pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# To get dummy variables for categorical features
df_imputed = pd.get_dummies(df_imputed, columns=['Sex'])

# Renaming columns for better readability
df_imputed = df_imputed.rename(columns={'Sex_-0.5': 'Sex_Male', 'Sex_0.5': 'Sex_Female'})

# Define your X and y
X = df_imputed.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory'], axis=1)
y = df_imputed[['AntisocialTrajectory', 'SubstanceUseTrajectory']]

# Standardize features (only the predictors, not the target variables or IDs)
cols_to_scale = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Convert 'AntisocialTrajectory' and 'SubstanceUseTrajectory' to int
y['AntisocialTrajectory'] = y['AntisocialTrajectory'].astype(int)
y['SubstanceUseTrajectory'] = y['SubstanceUseTrajectory'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's check the shape of these datasets
print("Training set - Features: ", X_train.shape, "Target: ", y_train.shape)
print("Testing set - Features: ", X_test.shape, "Target: ",y_test.shape)

# Add an intercept to the independent variable set
X = sm.add_constant(X)

# Define the dependent variables
y_asb = y['AntisocialTrajectory']
y_sub = y['SubstanceUseTrajectory']

# Fit the models
model_asb = sm.MNLogit(y_asb, X)
model_sub = sm.MNLogit(y_sub, X)

# Get the results
result_asb = model_asb.fit()
result_sub = model_sub.fit()

# Print the model statistics
print(result_asb.summary())
print(result_sub.summary())
