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

"""
Using Random Forest -

For AntisocialTrajectory results:

- Class 4 has the highest precision, recall and F1-score, and also the highest number of instances (support). This means that the model performs best when predicting this class.
- For classes 1, 2 and 3, the precision, recall and F1-score are lower. This could be due to lower representation of these classes in the dataset or the model having difficulty distinguishing these classes based on the given features.

For SubstanceUseTrajectory results:

- Classes 1 and 3 have comparable precision, recall, and F1-score, and both have much higher support than class 2.
- The model is less accurate and precise for class 2, which could again be due to lower representation of this class in the dataset or the model having difficulty distinguishing this class.
- Comparing these results to those from the multinomial logistic regression, the random forest model has improved the performance significantly in terms of all the metrics for both AntisocialTrajectory and SubstanceUseTrajectory. This could be due to the fact that random forest is a more complex model and can capture more complex relationships between features and target variables, while also being less prone to overfitting due to its ensemble nature.

In the context of the research, these results might provide more accurate and reliable predictions for both antisocial behavior and substance use trajectories. However, the model seems to struggle with classes that are under-represented in the data. If this is a concern, we could consider using techniques to handle the class imbalance, such as oversampling the minority classes, undersampling the majority class, or using a different algorithm that is more robust to class imbalances.
"""

-----------------------------------------------------------------------------------------------

# Fine tuning the random forest model

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a base model
rf = RandomForestClassifier(random_state=42)

# Initiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data for 'AntisocialTrajectory'
grid_search.fit(X_train, y_train['AntisocialTrajectory'])
grid_search.best_params_  # This line will print out the best parameters

# Predict on the test set
y_pred_asb_rf_grid = grid_search.predict(X_test)

# Calculate and print the metrics
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test['AntisocialTrajectory'], y_pred_asb_rf_grid))

# Repeat the above process for 'SubstanceUseTrajectory'
grid_search.fit(X_train, y_train['SubstanceUseTrajectory'])
grid_search.best_params_  # This line will print out the best parameters

# Predict on the test set
y_pred_sub_rf_grid = grid_search.predict(X_test)

# Calculate and print the metrics
print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test['SubstanceUseTrajectory'], y_pred_sub_rf_grid))

"""
The results of our model after performing hyperparameter tuning seem to be slightly better, but not dramatically so. Let's break down what these metrics mean:

For AntisocialTrajectory:

- The accuracy is still approximately 0.79, which is similar to the previous model.
- The weighted average precision is 0.70. High precision relates to the low false positive rate.
- The weighted average recall is 0.79. Recall (Sensitivity) is the ratio of correctly predicted positive observations to all observations in actual class. High recall relates to the low false negative rate.
- The weighted average f1-score is 0.71. The F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.

For SubstanceUseTrajectory:

- The accuracy is approximately 0.55, a slight improvement over the previous model.
- The weighted average precision is 0.55. A precision score closer to 1 indicates that the model has less false positives.
- The weighted average recall is also 0.55. A recall score closer to 1 indicates that the model has fewer false negatives.
- The weighted average f1-score is 0.51. The F1 Score takes both false positives and false negatives into account. The closer it is to 1, the better the model.

Overall, it seems that the Random Forest Classifier model has better performance than the previous model (Multinomial Logistic Regression). However, the model's performance on the SubstanceUseTrajectory target variable has much room for improvement.

It seems that the model has a difficult time predicting certain classes. This could potentially be due to class imbalance. If certain classes have far fewer samples than others, the model may have a hard time learning to predict these classes. We might have to consider techniques for handling class imbalance, such as oversampling the minority class, undersampling the majority class, or using a combination of both (SMOTE).

We can also consider using other algorithms, ensemble methods, or diving deeper into feature engineering. For instance, gradient boosting algorithms like XGBoost, LightGBM or CatBoost are known to provide better solutions in many cases.
"""

-----------------------------------------------------------------------------------------------

# SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_asb_smote, y_train_asb_smote = smote.fit_resample(X_train_scaled, y_train['AntisocialTrajectory'])
X_train_sub_smote, y_train_sub_smote = smote.fit_resample(X_train_scaled, y_train['SubstanceUseTrajectory'])

# Train the model using the resampled data
clf_asb_smote = RandomForestClassifier(random_state=42)
clf_sub_smote = RandomForestClassifier(random_state=42)

clf_asb_smote.fit(X_train_asb_smote, y_train_asb_smote)
clf_sub_smote.fit(X_train_sub_smote, y_train_sub_smote)

# Predict on the test data
y_pred_asb_rf_smote = clf_asb_smote.predict(X_test_scaled)
y_pred_sub_rf_smote = clf_sub_smote.predict(X_test_scaled)

# Calculate and print the metrics
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test['AntisocialTrajectory'], y_pred_asb_rf_smote))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test['SubstanceUseTrajectory'], y_pred_sub_rf_smote))

-----------------------------------------------------------------------------------------------

from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize and apply SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_asb_smoteenn, y_train_asb_smoteenn = smote_enn.fit_resample(X_train_scaled, y_train['AntisocialTrajectory'])
X_train_sub_smoteenn, y_train_sub_smoteenn = smote_enn.fit_resample(X_train_scaled, y_train['SubstanceUseTrajectory'])

# Initialize RandomForestClassifier
rf_asb_smoteenn = RandomForestClassifier(random_state=42)
rf_sub_smoteenn = RandomForestClassifier(random_state=42)

# Fit the model
rf_asb_smoteenn.fit(X_train_asb_smoteenn, y_train_asb_smoteenn)
rf_sub_smoteenn.fit(X_train_sub_smoteenn, y_train_sub_smoteenn)

# Make predictions
y_pred_asb_rf_smoteenn = rf_asb_smoteenn.predict(X_test_scaled)
y_pred_sub_rf_smoteenn = rf_sub_smoteenn.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test['AntisocialTrajectory'], y_pred_asb_rf_smoteenn))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test['SubstanceUseTrajectory'], y_pred_sub_rf_smoteenn))

"""
Starting with AntisocialTrajectory, the accuracy of the model has reduced slightly to 71%, as compared to the initial non-resampled model that had an accuracy of about 79%. However, the precision, recall and f1-score for the minority classes (1, 2, 3) have all improved. In other words, the model is now better at correctly identifying instances of these classes, at the cost of slightly reduced performance on the majority class (4).

In the context of the research, this can mean that the model is now more sensitive to the different trajectories of antisocial behavior, even if overall accuracy has decreased. 

Is this a good trade-off?
For instance, if it is particularly important to accurately identify participants on trajectories 1, 2, and 3, then a slightly reduced accuracy for trajectory 4 might be acceptable.

For SubstanceUseTrajectory, the accuracy of the model remains around 51%, roughly similar to the initial model without SMOTE. The metrics for the minority class (2) have improved, suggesting better detection for this class. The metrics for the other classes (1 and 3) are also comparable to the previous model.

If it's important to correctly identify the substance use trajectory for all participants, regardless of whether their trajectory is common (like 1 and 3) or less common (like 2), then the SMOTE-enhanced model might be preferable despite its lower overall accuracy.

In summary, the results indicate that the SMOTE technique has made the models more sensitive to the less common trajectories at the cost of slightly lower overall accuracy. This is a typical trade-off when dealing with imbalanced classes: aiming for high accuracy often means the model performs poorly for the minority class, while techniques to improve minority class performance can decrease overall accuracy.
"""

-----------------------------------------------------------------------------------------------

