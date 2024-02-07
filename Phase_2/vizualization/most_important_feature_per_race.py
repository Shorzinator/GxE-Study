import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_grouped_bar_chart(top_features, top_values, races, output_filename='most_important_per_race.png'):
    # Number of groups
    n_groups = len(races)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # The x locations for the groups
    index = np.arange(n_groups)

    # The width of the bars
    bar_width = 0.5

    # Opacity for the bars
    opacity = 0.8

    # Assigning different colors to each race for distinction
    colors = ['b', 'g', 'r', 'c']

    # Creating bars for the top feature per race
    for i, (feature, value) in enumerate(zip(top_features, top_values)):
        ax.bar(index[i], value, bar_width,
               alpha=opacity, color=colors[i],
               label=f'Race {races[i]}')

    # Adding labels for the features on the x-axis
    ax.set_xlabel('Top Features')
    ax.set_ylabel('Mean Absolute SHAP Value')
    ax.set_title('Top SHAP Value Feature by Race')
    ax.set_xticks(index)
    ax.set_xticklabels(top_features)  # Set the x labels to the names of the top features

    # It might be helpful to rotate the feature names if they are too long
    plt.xticks(rotation=10, ha='center')

    ax.legend()

    # Turn on the grid for the y-axis
    ax.yaxis.grid(True)

    # Save the plot before showing
    plt.savefig(f"../results/modeling/{output_filename}", bbox_inches='tight')
    print(f"SHAP summary plot saved as {output_filename}")

    # Show the plot
    plt.show()


# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Main function to orchestrate the calls
def main():
    model_paths = [
        '../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_1.0.pkl',
        '../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_2.0.pkl',
        '../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_3.0.pkl',
        '../results/models/classification/HetHieTL/AST/XGBClassifier/XGBClassifier_wPGS_race_4.0.pkl'
    ]

    races = ['1.0', '2.0', '3.0', '4.0']
    models = [load_model(path) for path in model_paths]
    X = pd.read_csv('../preprocessed_data/with_PGS/AST_new/X_test_new_AST.csv')

    top_features = []
    top_values = []

    # Loop over models to get the top feature and SHAP values
    for model, race in zip(models, races):
        # Ensure model is XGBoost and get feature names directly from booster
        feature_names = model.get_booster().feature_names

        # Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            mean_shap_values = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)

        # Get the index of the feature with the highest mean absolute SHAP value
        top_feature_index = np.argmax(mean_shap_values)

        # Get the name of the top feature and its value
        top_feature_name = feature_names[top_feature_index]
        top_feature_value = mean_shap_values[top_feature_index]

        # Add the feature name and value to the lists
        top_features.append(top_feature_name)
        top_values.append(top_feature_value)

    # Plot the grouped bar chart
    plot_grouped_bar_chart(top_features, top_values, races)


if __name__ == '__main__':
    main()
