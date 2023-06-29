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

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Define a manual label encoder function
def manual_encoder(y):
    classes = sorted(y.unique())  # Get unique classes
    class_dict = {c: i for i, c in enumerate(classes)}  # Assign each class a value from 0 to n-1
    return y.map(class_dict)  # Return the transformed labels

# Manual encoding for 'AntisocialTrajectory' and 'SubstanceUseTrajectory'
y_train_asb_smote = manual_encoder(pd.Series(y_train_asb_smote))
y_train_sub_smote = manual_encoder(pd.Series(y_train_sub_smote))
y_test_asb = manual_encoder(y_test['AntisocialTrajectory'])
y_test_sub = manual_encoder(y_test['SubstanceUseTrajectory'])

# Initialize XGBClassifier
xgb_asb = XGBClassifier(random_state=42, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss')
xgb_sub = XGBClassifier(random_state=42, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss')

# Fit the model
xgb_asb.fit(X_train_asb_smote, y_train_asb_smote)
xgb_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Make predictions
y_pred_asb_xgb = xgb_asb.predict(X_test_scaled)
y_pred_sub_xgb = xgb_sub.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test_asb, y_pred_asb_xgb))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test_sub, y_pred_sub_xgb))

"""
AntisocialTrajectory: The XGBClassifier model has an accuracy of 73%. Precision, recall, and F1-score for the class 3 are high, around 84-88%. However, for the other classes (0, 1, 2), these metrics are much lower, between 11-22%. This suggests that the model is doing well in predicting the most frequent class (class 3) but struggling with the minority classes.

SubstanceUseTrajectory: The model has an accuracy of 51%. Precision, recall, and F1-score for classes 0 and 2 are around 50-61%, but these scores are lower for class 1, around 32%. This again suggests that the model is doing fairly well for some classes but struggling with others (class 1 in this case).

This might be due to the imbalanced data. One strategy to address this could be to optimize the `scale_pos_weight` parameter or use different oversampling techniques or other methods to address class imbalance.

1. Process of arriving at the final code:
   - I started with a Random Forest Classifier to handle the multi-class classification problem.
   - As the data was imbalanced, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes.
   - I then trained two separate models for 'AntisocialTrajectory' and 'SubstanceUseTrajectory' using the balanced data.
   - However, the performance of the models was not satisfactory. Thus, I switched to the XGBoost Classifier.
   - To handle the labels in 'AntisocialTrajectory' and 'SubstanceUseTrajectory', I had to manually encode these labels from 0 to n-1, because XGBoost was throwing errors for unseen labels.
   - I used a custom encoding function that ensures that all labels are seen during training and testing.

2. Explanation of the final code:
   - The code first defines a custom function `manual_encoder()`, which manually encodes the class labels of 'AntisocialTrajectory' and 'SubstanceUseTrajectory' so that they start from 0 (which is required by XGBoost).
   - This function is applied to the target variables of the training and testing datasets.
   - Then two XGBoost Classifier models are initialized, one for each target variable.
   - The models are then trained using the SMOTE-resampled training data.
   - Predictions are made using the test data, and classification reports are generated to evaluate the model's performance.
"""

-----------------------------------------------------------------------------------------------

from lightgbm import LGBMClassifier

# Initialize LGBMClassifier
lgbm_asb = LGBMClassifier(random_state=42)
lgbm_sub = LGBMClassifier(random_state=42)

# Fit the model
lgbm_asb.fit(X_train_asb_smote, y_train_asb_smote)
lgbm_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Make predictions
y_pred_asb_lgbm = lgbm_asb.predict(X_test_scaled)
y_pred_sub_lgbm = lgbm_sub.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test_asb, y_pred_asb_lgbm))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test_sub, y_pred_sub_lgbm))

-----------------------------------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }

# Initialize LGBMClassifier
lgbm_asb = LGBMClassifier(random_state=42)

# Initialize GridSearchCV
grid = GridSearchCV(lgbm_asb, param_grid, verbose=1, cv=5, n_jobs=-1)

# Fit the model
grid.fit(X_train_asb_smote, y_train_asb_smote)

# Print the best parameters
print("Best parameters found: ", grid.best_params_)

# Make predictions using the model with the best parameters
y_pred_asb_lgbm = grid.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test_asb, y_pred_asb_lgbm))

-----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Feature importance for the XGBoost model trained on 'AntisocialTrajectory'
feature_importances_asb = pd.Series(xgb_asb.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10,6))
feature_importances_asb.sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title("Feature Importance for AntisocialTrajectory")
plt.show()

# Feature importance for the XGBoost model trained on 'SubstanceUseTrajectory'
feature_importances_sub = pd.Series(xgb_sub.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10,6))
feature_importances_sub.sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title("Feature Importance for SubstanceUseTrajectory")
plt.show()

-----------------------------------------------------------------------------------------------

# CATBoost

from catboost import CatBoostClassifier

# Initialize CatBoostClassifier
catb_asb = CatBoostClassifier(verbose=0, random_state=42)
catb_sub = CatBoostClassifier(verbose=0, random_state=42)

# Fit the model
catb_asb.fit(X_train_asb_smote, y_train_asb_smote)
catb_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Make predictions
y_pred_asb_catb = catb_asb.predict(X_test_scaled)
y_pred_sub_catb = catb_sub.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test_asb, y_pred_asb_catb))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test_sub, y_pred_sub_catb))

-----------------------------------------------------------------------------------------------

from sklearn.ensemble import VotingClassifier

# Initialize the individual models
rf_asb = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_asb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

rf_sub = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_sub = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Create the ensemble model
ensemble_asb = VotingClassifier(estimators=[('rf', rf_asb), ('xgb', xgb_asb)], voting='soft')
ensemble_sub = VotingClassifier(estimators=[('rf', rf_sub), ('xgb', xgb_sub)], voting='soft')

# Fit the ensemble model
ensemble_asb.fit(X_train_asb_smote, y_train_asb_smote)
ensemble_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Make predictions
y_pred_asb_ensemble = ensemble_asb.predict(X_test_scaled)
y_pred_sub_ensemble = ensemble_sub.predict(X_test_scaled)

# Print classification report
print("Classification report for AntisocialTrajectory:")
print(classification_report(y_test_asb, y_pred_asb_ensemble))

print("Classification report for SubstanceUseTrajectory:")
print(classification_report(y_test_sub, y_pred_sub_ensemble))

"""
For the AntisocialTrajectory, the model has performed best on the major class '3' with a Precision of 0.85 and Recall of 0.89. This is expected as this class has more instances, and models tend to perform well on the majority class. The performance on other classes (0, 1, 2) is lower, as these are the minority classes and harder for the model to learn.

The situation is similar for the SubstanceUseTrajectory. Here, the model performance is more evenly distributed across classes, but still, the F1-scores suggest that the model is not doing as well on class '1', likely because it is the minority class here.

Comparatively, this ensemble model seems to have slightly improved performance on AntisocialTrajectory in terms of overall accuracy (0.74 vs. 0.72-0.73 in earlier models) but similar or marginally lower for SubstanceUseTrajectory (0.53 vs. 0.53-0.54 in earlier models).

The improvement might seem minimal, but in a challenging multi-class, imbalanced problem like this, even small improvements can be meaningful. You should also consider looking at metrics like AUC-ROC for multi-class problems, which can provide another perspective on model performance.
"""
-----------------------------------------------------------------------------------------------

# Define a function for calculating multi-class ROC AUC Score
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# List of predictions for AntisocialTrajectory
y_preds_asb = [y_pred_asb_xgb, y_pred_asb_lgbm, y_pred_asb_catb]

# List of predictions for SubstanceUseTrajectory
y_preds_sub = [y_pred_sub_xgb, y_pred_sub_lgbm, y_pred_sub_catb]

# Calculate and print AUC-ROC for AntisocialTrajectory
print("For AntisocialTrajectory:")
for y_pred_asb in y_preds_asb:
    print("AUC-ROC (macro-average): ", multiclass_roc_auc_score(y_test_asb, y_pred_asb, average="macro"))
    print("AUC-ROC (weighted): ", multiclass_roc_auc_score(y_test_asb, y_pred_asb, average="weighted"))

# Calculate and print AUC-ROC for SubstanceUseTrajectory
print("\nFor SubstanceUseTrajectory:")
for y_pred_sub in y_preds_sub:
    print("AUC-ROC (macro-average): ", multiclass_roc_auc_score(y_test_sub, y_pred_sub, average="macro"))
    print("AUC-ROC (weighted): ", multiclass_roc_auc_score(y_test_sub, y_pred_sub, average="weighted"))

-----------------------------------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 5, 10]
}

# Initialize the LightGBM model
lgbm = LGBMClassifier(random_state=42)

# Initialize GridSearchCV
grid_search_asb = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='roc_auc_ovr_weighted', n_jobs=-1)
grid_search_sub = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='roc_auc_ovr_weighted', n_jobs=-1)

# Fit GridSearchCV for AntisocialTrajectory
grid_search_asb.fit(X_train_asb_smote, y_train_asb_smote)

# Fit GridSearchCV for SubstanceUseTrajectory
grid_search_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Get the best parameters
best_params_asb = grid_search_asb.best_params_
best_params_sub = grid_search_sub.best_params_

# Train the LightGBM model with the best parameters
lgbm_best_asb = LGBMClassifier(**best_params_asb, random_state=42)
lgbm_best_sub = LGBMClassifier(**best_params_sub, random_state=42)

lgbm_best_asb.fit(X_train_asb_smote, y_train_asb_smote)
lgbm_best_sub.fit(X_train_sub_smote, y_train_sub_smote)

# Make predictions with the tuned model
y_pred_asb_tuned = lgbm_best_asb.predict(X_test_scaled)
y_pred_sub_tuned = lgbm_best_sub.predict(X_test_scaled)

# Print classification report for the tuned model
print("Classification report for AntisocialTrajectory (Tuned LightGBM):")
print(classification_report(y_test_asb, y_pred_asb_tuned))

print("Classification report for SubstanceUseTrajectory (Tuned LightGBM):")
print(classification_report(y_test_sub, y_pred_sub_tuned))

# Print the AUC-ROC for the tuned model
print("AUC-ROC for AntisocialTrajectory (Tuned LightGBM):")
print("Macro-average: ", multiclass_roc_auc_score(y_test_asb, y_pred_asb_tuned, average="macro"))
print("Weighted: ", multiclass_roc_auc_score(y_test_asb, y_pred_asb_tuned, average="weighted"))

print("AUC-ROC for SubstanceUseTrajectory (Tuned LightGBM):")
print("Macro-average: ", multiclass_roc_auc_score(y_test_sub, y_pred_sub_tuned, average="macro"))
print("Weighted: ", multiclass_roc_auc_score(y_test_sub, y_pred_sub_tuned, average="weighted"))

-----------------------------------------------------------------------------------------------

"""
Given the challenges with the dataset, especially the class imbalance and the difficulty in modeling the minority classes effectively, let's try another approach to address class imbalance. We will use ADASYN (Adaptive Synthetic Sampling) which is an advanced form of SMOTE. ADASYN adapts the number of synthetic samples generated for each class based on how difficult it is to learn.
"""

from imblearn.over_sampling import ADASYN

# Separate the training labels
y_train_asb = y_train['AntisocialTrajectory']
y_train_sub = y_train['SubstanceUseTrajectory']

# Using ADASYN to oversample the training data
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_train_asb_adasyn, y_train_asb_adasyn = adasyn.fit_resample(X_train, y_train_asb)
X_train_sub_adasyn, y_train_sub_adasyn = adasyn.fit_resample(X_train, y_train_sub)

# Get the best parameters from previous grid search
best_params_asb = grid_search_asb.best_params_
best_params_sub = grid_search_sub.best_params_

# Initialize the model with the best parameters
lgbm_best_asb_adasyn = LGBMClassifier(**best_params_asb, random_state=42)
lgbm_best_sub_adasyn = LGBMClassifier(**best_params_sub, random_state=42)

# Train the model
lgbm_best_asb_adasyn.fit(X_train_asb_adasyn, y_train_asb_adasyn)
lgbm_best_sub_adasyn.fit(X_train_sub_adasyn, y_train_sub_adasyn)

# Make predictions
y_pred_asb_adasyn = lgbm_best_asb_adasyn.predict(X_test)
y_pred_sub_adasyn = lgbm_best_sub_adasyn.predict(X_test)

# Print classification report
print("Classification report for AntisocialTrajectory (ADASYN):")
print(classification_report(y_test['AntisocialTrajectory'], y_pred_asb_adasyn))

print("Classification report for SubstanceUseTrajectory (ADASYN):")
print(classification_report(y_test['SubstanceUseTrajectory'], y_pred_sub_adasyn))

# Print the AUC-ROC
print("AUC-ROC for AntisocialTrajectory (ADASYN):")
print("Macro-average: ", multiclass_roc_auc_score(y_test['AntisocialTrajectory'], y_pred_asb_adasyn, average="macro"))
print("Weighted: ", multiclass_roc_auc_score(y_test['AntisocialTrajectory'], y_pred_asb_adasyn, average="weighted"))

print("AUC-ROC for SubstanceUseTrajectory (ADASYN):")
print("Macro-average: ", multiclass_roc_auc_score(y_test['SubstanceUseTrajectory'], y_pred_sub_adasyn, average="macro"))
print("Weighted: ", multiclass_roc_auc_score(y_test['SubstanceUseTrajectory'], y_pred_sub_adasyn, average="weighted"))

-----------------------------------------------------------------------------------------------

# updated multinomial regression

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

# Convert Sex to binary (1 for Male, 0 for Female)
df_imputed['Is_Male'] = (df_imputed['Sex'] == -0.5).astype(int)

# Define your X and y
X = df_imputed.drop(['ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex'], axis=1)
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
