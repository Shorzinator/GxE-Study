import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the old and new datasets
file_old = "../preprocessed_data/preprocessed_data_old_AST.csv"
file_new = "../preprocessed_data/preprocessed_data_new_AST.csv"

df_old = pd.read_csv(file_old)
df_new = pd.read_csv(file_new)

# Column names of both datasets for comparison
columns_old = df_old.columns
columns_new = df_new.columns

# Prepare the common data columns for transfer learning
common_columns = [col for col in columns_old if col in columns_new]

# Prepare the old dataset
X_old = df_old[common_columns]
y_old = df_old["AntisocialTrajectory"]  # Replace with your actual target column name

# Split the old data into training and testing sets
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model on the old dataset
model_old = LogisticRegression(max_iter=1000)
model_old.fit(X_train_old, y_train_old)

# Evaluate the model on the old dataset
predictions_old = model_old.predict(X_test_old)
accuracy_old = accuracy_score(y_test_old, predictions_old)
report_old = classification_report(y_test_old, predictions_old)

# Prepare the new dataset (excluding 'Race' feature)
X_new = df_new[common_columns]
y_new = df_new["AntisocialTrajectory"]  # Replace with your actual target column name

# Split the new data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Retrain the model using the new dataset
model_new_phase1 = LogisticRegression(max_iter=1000)
model_new_phase1.fit(X_train_new, y_train_new)

# Evaluate the model on the new dataset
predictions_new_phase1 = model_new_phase1.predict(X_test_new)
accuracy_new_phase1 = accuracy_score(y_test_new, predictions_new_phase1)
report_new_phase1 = classification_report(y_test_new, predictions_new_phase1)

# Include the 'Race' feature in the new dataset
race_columns = [col for col in columns_new if "Race" in col]
X_new_with_race = df_new[common_columns + race_columns]

# Split the new data with 'Race' feature
X_train_new_with_race, X_test_new_with_race, y_train_new_with_race, y_test_new_with_race = train_test_split(
    X_new_with_race, y_new, test_size=0.2, random_state=42)

# Retrain the model with the 'Race' feature included
model_new_phase2 = LogisticRegression(max_iter=1000)
model_new_phase2.fit(X_train_new_with_race, y_train_new_with_race)

# Evaluate the model on the new dataset with 'Race' feature
predictions_new_phase2 = model_new_phase2.predict(X_test_new_with_race)
accuracy_new_phase2 = accuracy_score(y_test_new_with_race, predictions_new_phase2)
report_new_phase2 = classification_report(y_test_new_with_race, predictions_new_phase2)

# Print the results
print("Accuracy on Old Dataset:", accuracy_old)
print("Classification Report on Old Dataset:\n", report_old)
print("Accuracy on New Dataset (Phase 1):", accuracy_new_phase1)
print("Classification Report on New Dataset (Phase 1):\n", report_new_phase1)
print("Accuracy on New Dataset (Phase 2 - Including 'Race'):", accuracy_new_phase2)
print("Classification Report on New Dataset (Phase 2):\n", report_new_phase2)
