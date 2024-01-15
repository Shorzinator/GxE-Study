import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

preprocessed_data = pd.read_csv("/Users/shouryamaheshwari/Desktop/GxE-Study/Phase_2/preprocessed_data/preprocessed_data_old_AST.csv")

# Separating features and target variable
X = preprocessed_data.drop(['AntisocialTrajectory'], axis=1)
y = preprocessed_data['AntisocialTrajectory']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and fitting the Ridge Regression model
model = Ridge()
model.fit(X_train, y_train)

# Predicting on the test set and evaluating the model
y_pred = model.predict(X_test)
r2_score_new = r2_score(y_test, y_pred)
print(r2_score_new)
