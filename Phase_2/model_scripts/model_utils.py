import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)