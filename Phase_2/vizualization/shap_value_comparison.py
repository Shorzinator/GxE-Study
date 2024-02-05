import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle


# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def shap_value_comparison_AST(model, X, output_filename='shap_summary_plot_AST.png'):
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    # Get the current figure and axes
    fig, ax = plt.gcf(), plt.gca()

    # Get the current legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Define your new labels
    new_labels = ['High Decline', 'Moderate', 'Adolescence-Peak', 'Low']

    # Set the new labels
    ax.legend(handles, new_labels)

    # Set x and y labels
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Set title
    plt.title('SHAP Summary Plot for AntisocialTrajectory', fontsize=16)

    # Increase the size of the labels in the legend
    plt.setp(ax.get_legend().get_texts(), fontsize=12)

    # Save the current figure
    plt.savefig(f"../results/modeling/{output_filename}", bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    print(f"SHAP summary plot saved as {output_filename}")


def shap_value_comparison_SUT(model, X, output_filename='shap_summary_plot_SUT.png'):
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    # Get the current figure and axes
    fig, ax = plt.gcf(), plt.gca()

    # Get the current legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Define your new labels
    new_labels = ['High Use', 'Low Use', 'Typical Use']

    # Set the new labels
    ax.legend(handles, new_labels)

    # Set x and y labels
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Set title
    plt.title('SHAP Summary Plot for SubstanceUseTrajectory', fontsize=16)

    # Increase the size of the labels in the legend
    plt.setp(ax.get_legend().get_texts(), fontsize=12)

    # Save the current figure
    plt.savefig(f"../results/modeling/{output_filename}", bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    print(f"SHAP summary plot saved as {output_filename}")


# Main function to orchestrate the calls
def main(target):

    # Load your model and data
    if target == "AntisocialTrajectory":
        model = load_model('../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_1.0.pkl')
        X = pd.read_csv('../preprocessed_data/with_PGS/AST_new/X_test_new_AST.csv')

        shap_value_comparison_AST(model, X, output_filename="shap_summary_plot_AST_race_1.0.png")

    else:
        model = load_model('../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_2.0.pkl')
        X = pd.read_csv('../preprocessed_data/with_PGS/AST_new/X_test_new_AST.csv')

        shap_value_comparison_SUT(model, X, output_filename="shap_summary_plot_AST_race_2.0.png")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(target=target_1)
