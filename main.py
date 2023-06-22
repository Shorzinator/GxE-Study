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

-----------------------------------------------------------------------------------------------
# troubleshooting

# Add constant to the test data
X_test = sm.add_constant(X_test)

# Make predictions on the test set
y_pred_asb = result_asb.predict(X_test)
y_pred_sub = result_sub.predict(X_test)

# Convert probabilities into class labels
y_pred_asb = y_pred_asb.idxmax(axis=1)
y_pred_sub = y_pred_sub.idxmax(axis=1)

-----------------------------------------------------------------------------------------------

# correlation

import seaborn as sns
import matplotlib.pyplot as plt

# Generate correlation matrix
corr_matrix = df.corr()

# Plot correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

-----------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for 'AntisocialTrajectory'
accuracy_asb = accuracy_score(y_test['AntisocialTrajectory'], y_pred_asb)
precision_asb = precision_score(y_test['AntisocialTrajectory'], y_pred_asb, average='weighted')
recall_asb = recall_score(y_test['AntisocialTrajectory'], y_pred_asb, average='weighted')
f1_asb = f1_score(y_test['AntisocialTrajectory'], y_pred_asb, average='weighted')

# Calculate metrics for 'SubstanceUseTrajectory'
accuracy_sub = accuracy_score(y_test['SubstanceUseTrajectory'], y_pred_sub)
precision_sub = precision_score(y_test['SubstanceUseTrajectory'], y_pred_sub, average='weighted')
recall_sub = recall_score(y_test['SubstanceUseTrajectory'], y_pred_sub, average='weighted')
f1_sub = f1_score(y_test['SubstanceUseTrajectory'], y_pred_sub, average='weighted')

# Print the metrics
print("AntisocialTrajectory - Accuracy: ", accuracy_asb, " Precision: ", precision_asb, " Recall: ", recall_asb, " F1 Score: ", f1_asb)
print("SubstanceUseTrajectory - Accuracy: ", accuracy_sub, " Precision: ", precision_sub, " Recall: ", recall_sub, " F1 Score: ", f1_sub)

-----------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Drop the 'const' column from the test set
X_test = X_test.drop('const', axis=1)

# Define the Random Forest classifier
clf_asb = RandomForestClassifier(random_state=42)
clf_sub = RandomForestClassifier(random_state=42)

# Fit the model to the training data
clf_asb.fit(X_train, y_train['AntisocialTrajectory'])
clf_sub.fit(X_train, y_train['SubstanceUseTrajectory'])

# Predict on the test data
y_pred_asb_rf = clf_asb.predict(X_test)
y_pred_sub_rf = clf_sub.predict(X_test)

# Calculate and print the metrics
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test['AntisocialTrajectory'], y_pred_asb_rf))
print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test['SubstanceUseTrajectory'], y_pred_sub_rf))
