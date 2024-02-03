import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# Function to split the data into training and testing sets
def split_data(df, target):
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target])

    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)


# Function to evaluate model
# def evaluate_model(model, X_test, y_test, algo_type="regression"):
#     predictions = model.predict(X_test)
#     return r2_score(y_test, predictions)


def evaluate_model(model, X_test, y_test, algo_type="classification"):
    # Get the predicted probabilities for each class
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


# Function to perform randomized search on RandomForestRegressor
def tune_random_forest(model, X_train, y_train):
    # Define the parameter space to explore
    param_distributions = {
        'n_estimators': np.arange(100, 1001, 50),
        'max_depth': [None] + list(np.arange(10, 101, 10)),
        'min_samples_split': np.arange(2, 21, 2),
        'min_samples_leaf': np.arange(1, 21, 2),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Initialize the base model
    rf = model

    # Initialize RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit RandomizedSearchCV to the data
    rf_random.fit(X_train, y_train)

    # Return the best estimator
    return rf_random.best_estimator_
