import pandas as pd
from sklearn.linear_model import Ridge

from Phase_2.model_scripts.model_utils import split_data, evaluate_model
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


# Function to train model
def train_model(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# Main function to run the model training and evaluation
def main(target_variable):
    if target_variable == "AntisocialTrajectory":
        X_train_old = load_data("../../../preprocessed_data/with_PGS/AST_old/X_train_old_AST.csv")
        X_test_old = load_data("../../../preprocessed_data/with_PGS/AST_old/X_test_old_AST.csv")
        y_train_old = load_data("../../../preprocessed_data/with_PGS/AST_old/y_train_old_AST.csv")
        y_test_old = load_data("../../../preprocessed_data/with_PGS/AST_old/y_test_old_AST.csv")

        X_train_new = load_data("../../../preprocessed_data/with_PGS/AST_new/X_train_new_AST.csv")
        X_test_new = load_data("../../../preprocessed_data/with_PGS/AST_new/X_test_new_AST.csv")
        y_train_new = load_data("../../../preprocessed_data/with_PGS/AST_new/y_train_new_AST.csv")
        y_test_new = load_data("../../../preprocessed_data/with_PGS/AST_new/y_test_new_AST.csv")

    else:
        X_train_old = load_data("../../../preprocessed_data/with_PGS/SUT_old/X_train_old_SUT.csv")
        X_test_old = load_data("../../../preprocessed_data/with_PGS/SUT_old/X_test_old_SUT.csv")
        y_train_old = load_data("../../../preprocessed_data/with_PGS/SUT_old/y_train_old_SUT.csv")
        y_test_old = load_data("../../../preprocessed_data/with_PGS/SUT_old/y_test_old_SUT.csv")

        X_train_new = load_data("../../../preprocessed_data/with_PGS/SUT_new/X_train_new_SUT.csv")
        X_test_new = load_data("../../../preprocessed_data/with_PGS/SUT_new/X_test_new_SUT.csv")
        y_train_new = load_data("../../../preprocessed_data/with_PGS/SUT_new/y_train_new_SUT.csv")
        y_test_new = load_data("../../../preprocessed_data/with_PGS/SUT_new/y_test_new_SUT.csv")

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
