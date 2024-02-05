
import pandas as pd
import shap
import pickle
import numpy as np

import matplotlib.pyplot as plt


def plot_grouped_bar_chart(top_features, top_values, races, output_filename='shap_summary_plot.png'):
    # Number of groups
    n_groups = len(races)

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
        ax.bar(index + i * bar_width, [value[i] for value in top_values], bar_width,
               alpha=opacity, color=colors[i % len(colors)],
               label=f'Race {race}')

    # Adding labels for the features on the x-axis
    ax.set_xlabel('Top Features')
    ax.set_ylabel('Mean Absolute SHAP Value')
    ax.set_title('Top SHAP Value Features by Race')
    ax.set_xticks(index + bar_width / 2 * (n_groups - 1))
    ax.set_xticklabels(top_features)
    ax.legend()

    # Turn on the grid for the y-axis
    ax.yaxis.grid(True)

    # Save the plot before showing
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"SHAP summary plot saved as {output_filename}")

    # Show the plot
    plt.show()


# Function to load a model from a .pkl file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def shap_value_comparison(models, X, races, output_filename='most_important_per_race.png'):
    top_features = []
    top_values = []
    race_labels = []

    for model, race in zip(models, races):
        # Ensure the model is XGBoost and get feature names directly from booster
        if hasattr(model, 'get_booster'):
            feature_names = model.get_booster().feature_names
        else:
            raise ValueError("The model provided does not support get_booster(). Ensure it is an XGBoost model.")

        # Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get the mean absolute SHAP values for each feature
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)

        # Ensure mean_shap_values is one-dimensional
        if len(mean_shap_values.shape) > 1:
            mean_shap_values = np.mean(mean_shap_values, axis=0)

        # Get the index of the feature with the highest mean absolute SHAP value
        top_feature_index = np.argmax(mean_shap_values)

        # Append top feature name, its value, and race to the lists
        top_features.append(feature_names[top_feature_index])
        top_values.append(mean_shap_values[top_feature_index])
        race_labels.append(race)

    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(race_labels, top_values, alpha=0.5)

    # Annotate each point with the feature name
    for i, txt in enumerate(top_features):
        ax.annotate(txt, (race_labels[i], top_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Races', fontsize=12)
    ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12)
    ax.set_title('Top SHAP Value Feature by Race', fontsize=16)

    # Save the plot before showing
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"SHAP summary plot saved as {output_filename}")

    # Show the plot
    plt.show()


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

    # Initialize a dictionary to store the top feature and its value for each race
    top_features_dict = {}

    # Loop over models to get top features and SHAP values
    for i, (model, race) in enumerate(zip(models, races)):
        # Ensure the model is XGBoost and get feature names directly from booster
        feature_names = model.get_booster().feature_names

        # Initialize SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get the mean absolute SHAP values for each feature
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)

        # If shap_values is a list (for multi-class), we need to handle it differently
        if isinstance(shap_values, list):
            mean_shap_values = np.sum([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)

        # Get the index of the feature with the highest mean absolute SHAP value
        top_feature_index = np.argmax(mean_shap_values)

        # Get the name of the top feature and its value
        top_feature_name = feature_names[top_feature_index]
        top_feature_value = mean_shap_values[top_feature_index]

        # Store the top feature and its value in the dictionary
        top_features_dict[race] = (top_feature_name, top_feature_value)

    # Prepare the data for plotting
    top_features = [feat_val[0] for feat_val in top_features_dict.values()]
    top_values = [feat_val[1] for feat_val in top_features_dict.values()]

    # Plot the grouped bar chart
    plot_grouped_bar_chart(top_features, top_values, races, 'shap_summary_plot.png')


if __name__ == '__main__':
    main()
