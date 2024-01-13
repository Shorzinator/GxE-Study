import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utility.path_utils import get_path_from_root

file_path_old_ast = get_path_from_root("Phase_2", "preprocessed_data", "preprocessed_data_old_AST.csv")
file_path_new_ast = get_path_from_root("Phase_2", "preprocessed_data", "preprocessed_data_new_AST.csv")

file_path_old_sut = get_path_from_root("Phase_2", "preprocessed_data", "preprocessed_data_old_SUT.csv")
file_path_new_sut = get_path_from_root("Phase_2", "preprocessed_data", "preprocessed_data_new_SUT.csv")


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to prepare data
def prepare_data(df, target_variable, exclude_columns=None):
    if exclude_columns:
        features = [col for col in df.columns if col not in exclude_columns]
    else:
        features = [col for col in df.columns if col != target_variable]
    X = df[features]
    y = df[target_variable]
    return X, y


# Function to split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train model
def train_model(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)


# Main function to run the model training and evaluation
def main(target_variable):
    if target_variable == "AntisocialTrajectory":
        file_path_old = file_path_old_ast
        file_path_new = file_path_new_ast
    else:
        file_path_old = file_path_old_sut
        file_path_new = file_path_new_sut

    # Load the data
    df_old = load_data(file_path_old)
    df_new = load_data(file_path_new)

    # Prepare the data
    X_old, y_old = prepare_data(df_old, target_variable)
    X_new, y_new = prepare_data(df_new, target_variable, exclude_columns=['Race'])

    # Split the data
    X_train_old, X_test_old, y_train_old, y_test_old = split_data(X_old, y_old)
    X_train_new, X_test_new, y_train_new, y_test_new = split_data(X_new, y_new)

    # Train the model on the old data
    model_old = train_model(X_train_old, y_train_old)
    r2_old = evaluate_model(model_old, X_test_old, y_test_old)

    # Train the model on the new data excluding the 'Race' feature
    model_new = train_model(X_train_new, y_train_new)
    r2_new = evaluate_model(model_new, X_test_new, y_test_new)

    # Prepare the new data including the 'Race' feature
    X_new_with_race, y_new_with_race = prepare_data(df_new, target_variable)
    X_train_new_with_race, X_test_new_with_race, y_train_new_with_race, y_test_new_with_race = split_data(
        X_new_with_race, y_new_with_race)

    # Train the model on the new data including the 'Race' feature
    model_new_with_race = train_model(X_train_new_with_race, y_train_new_with_race)
    r2_new_with_race = evaluate_model(model_new_with_race, X_test_new_with_race, y_test_new_with_race)

    # Print the results
    print(f'R-squared on Old Dataset: {r2_old}')
    print(f'R-squared on New Dataset (excluding \'Race\'): {r2_new}')
    print(f'R-squared on New Dataset (including \'Race\'): {r2_new_with_race}')


if __name__ == "__main__":
    target_1 = 'AntisocialTrajectory'
    target_2 = 'SubstanceUseTrajectory'

    # Run the main function
    main(target_1)
