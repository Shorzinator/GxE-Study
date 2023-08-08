# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
data = pd.read_csv("/Phase 5/data/processed/Data_GxE_on_EXT_trajectories_new.csv")

# Assuming 'ID' and 'FamilyID' are non-predictive columns
X = data.drop(columns=['ID', 'FamilyID', 'AntisocialTrajectory'])  # remove these if there are other non-predictive
# columns
y = data['AntisocialTrajectory']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the One-vs-Rest classifier
clf = OneVsRestClassifier(LogisticRegression(max_iter=10000))  # using logistic regression as the base classifier
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
