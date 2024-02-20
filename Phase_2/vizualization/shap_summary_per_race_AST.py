import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np


# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def plot_grouped_bar_chart(top_features, top_values, races, output_filename='most_important_per_race.png'):
    # Number of groups
    n_groups = len(top_features)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # The x locations for the groups
    index = np.arange(n_groups)

    # The width of the bars (can be adjusted to your preference)
    bar_width = 0.2

    # Opacity for the bars (can be adjusted to your preference)
    opacity = 0.8

    # Assigning different colors to each race for distinction
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Creating bars for each top feature per race
    for i, race in enumerate(races):
        ax.bar(index + i * bar_width, top_values[i], bar_width,
               alpha=opacity, color=colors[i % len(colors)],
               label=f'Race {race}')

    # Adding labels for the features on the x-axis
    ax.set_xlabel('Top Features')
    ax.set_ylabel('Mean Absolute SHAP Value')
    ax.set_title('Top SHAP Value Features by Race for Predicting AST')
    ax.set_xticks(index + bar_width / 2 * (n_groups - 1))
    ax.set_xticklabels(top_features)
    ax.legend()

    # Turn on the grid for the y axis
    ax.yaxis.grid(True)

    # Save the plot before showing
    plt.savefig(f"../results/modeling/{output_filename}", bbox_inches='tight')
    print(f"SHAP summary plot saved as {output_filename}")

    # Show the plot
    plt.show()


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
    X = pd.read_csv('../preprocessed_data/without_resampling/with_PGS/AST_new/X_test_new_AST.csv')

    top_features = set()  # Initialize as a set and keep it as a set throughout the loop
    top_values = [[] for _ in races]

    # Loop over models to get top features and SHAP values
    for i, (model, race) in enumerate(zip(models, races)):
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

        # Add the feature name to the set of top features
        top_features.add(top_feature_name)

        # Add the feature value to the corresponding list in top_values
        top_values[i].append(top_feature_value)

    # Now convert the set of top features to a sorted list
    top_features = sorted(list(top_features))

    # Reconstruct the list of top_values for each race
    for i, model in enumerate(models):
        # Ensure model is XGBoost and get feature names directly from booster
        feature_names = model.get_booster().feature_names

        # Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            mean_shap_values = np.sum([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)

        # Reset the list for the current race
        top_values[i] = []

        # Add the feature values to the corresponding list in top_values
        for feature in top_features:
            feature_index = feature_names.index(feature)
            top_values[i].append(mean_shap_values[feature_index])

    # Plot the grouped bar chart
    plot_grouped_bar_chart(top_features, top_values, races)


if __name__ == '__main__':
    main()
