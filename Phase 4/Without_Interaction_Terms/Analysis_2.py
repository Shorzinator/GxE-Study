# creating classification reports

from sklearn.metrics import classification_report
from joblib import load
import pandas as pd

models = ["lgbm", "lr", "rf", "xgb", "gb"]
outcomes = ["AST_1_vs_4", "AST_2_vs_4", "AST_3_vs_4"]

for model in models:
    for outcome in outcomes:
        # Load the model
        clf = load(
            f"C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_{model}\\Without IT\\best_model_{model}_{outcome}.joblib")

        # Load corresponding X_test and y_test datasets
        X_test = pd.read_csv(
            f"C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_{model}\\Without IT\\X_test_{outcome}.csv")
        y_test = pd.read_csv(
            f"C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_{model}\\Without IT\\y_test_{outcome}.csv")

        # Make predictions
        y_pred = clf.predict(X_test)
        # Generate the classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Convert the report to dataframe
        df_class_report = pd.DataFrame(class_report).transpose()

        # Save the report to csv
        df_class_report.to_csv(
            f"C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_{model}\\Without IT\\class_report_{model}_{outcome}.csv",
            index=False)

        print(f"Classification Report for Model: {model} with Outcome: {outcome} saved")
