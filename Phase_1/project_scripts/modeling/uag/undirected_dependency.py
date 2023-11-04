import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Phase_1.config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing_without_interaction_mrf
from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "neighborhood_regression"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")


def neighborhood_regression(target):
    logger.info(f"Starting Neighborhood Regression for {target}...\n")

    # Subdirectories for metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    ensure_directory_exists(metrics_dir)

    # Load the data
    df = load_data_old()

    # Assigning features based on the outcome.
    if target == "AntisocialTrajectory":
        features_to_use = FEATURES_FOR_AST
    else:
        features_to_use = FEATURES_FOR_SUT

    # Applying primary preprocessing
    data_preprocessed, features = primary_preprocessing_mrf(df, features_to_use, target)

    X = pd.DataFrame(data_preprocessed[features])  # feature dataset
    y = pd.DataFrame(data_preprocessed[target])    # outcome dataset

    # Applying secondary preprocessing
    X_train, y_train, X_test, y_test = secondary_preprocessing_without_interaction_mrf(X, y, features)

    # Combining data to replicate processed 'data_processed'
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined.reset_index(drop=True, inplace=True)

    y_combined = pd.concat([y_train, y_test], axis=0)
    y_combined.reset_index(drop=True, inplace=True)

    data_combined = pd.concat([X_combined, y_combined], axis=1)

    # Neighborhood Regression
    graph_edges = []
    for target_var in data_combined.columns:
        predictors = data_combined.columns.drop(target_var).tolist()
        X = sm.add_constant(data_combined[predictors])
        model = sm.OLS(data_combined[target_var], X).fit()

        # Check for significant predictors (p-value threshold: 0.05)
        significant_predictors = model.pvalues[model.pvalues < 0.05].index
        if 'const' in significant_predictors:
            significant_predictors = significant_predictors.drop('const')
        for predictor in significant_predictors:
            if (target_var, predictor) not in graph_edges and (predictor, target_var) not in graph_edges:
                graph_edges.append((target_var, predictor))

    # Define your genetic and environmental features explicitly
    genetic_feature = 'PolygenicScoreEXT'
    environmental_features = ['Age', 'Sex', 'PolygenicScoreEXT_x_Age', 'DelinquentPeers', 'SchoolConnect',
                              'NeighborConnect', 'ParentalWarmth', 'PolygenicScoreEXT_x_Is_Male',
                              'SubstanceUseTrajectory']
    outcome_variable = target

    # Graph Construction with updated nodes based on the actual feature names
    graph = nx.DiGraph()
    graph.add_edges_from(graph_edges)

    # Define the shell positions for the graph layout
    shell_pos = [[genetic_feature], environmental_features, [outcome_variable]]
    pos = nx.shell_layout(graph, shell_pos)

    # Node Customization - Now you need to base this on the actual feature names rather than the prefixes
    node_colors = []
    node_shapes = []
    for node in graph.nodes():
        if node == genetic_feature:
            node_colors.append("#1f77b4")  # Blue for Genetic Feature
            node_shapes.append('s')  # Square for Genetic Feature
        elif node in environmental_features:
            node_colors.append("#ff7f0e")  # Orange for Environmental Features
            node_shapes.append('o')  # Circle for Environmental Features
        elif node == outcome_variable:
            node_colors.append("#2ca02c")  # Green for Outcome Variable
            node_shapes.append('d')  # Diamond for Outcome Variable

    # Visualization
    plt.figure(figsize=(12, 12))

    # Drawing nodes with a larger size and based on their actual feature categories
    for shape in set(node_shapes):
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=[node for node in graph.nodes() if graph.nodes[node].get('shape') == shape],
                               node_color=[node_colors[node_shapes.index(shape)]],
                               node_size=2500, node_shape=shape)

    # Draw edges with the weights and colors
    nx.draw_networkx_edges(graph, pos, edgelist=graph_edges,
                           width=[weight for _, weight in edge_weights],
                           edge_color=[color for _, color in edge_colors],
                           arrowstyle='-|>', arrowsize=20)

    # Draw edge labels with the weights
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Adding legend for the nodes
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Genetic Feature (G)', markersize=10,
                              markerfacecolor="#1f77b4"),
                       Line2D([0], [0], marker='o', color='w', label='Environmental Features (E)', markersize=10,
                              markerfacecolor="#ff7f0e"),
                       Line2D([0], [0], marker='d', color='w', label='Outcome (O)', markersize=10,
                              markerfacecolor="#2ca02c")]
    plt.legend(handles=legend_elements, loc="best")

    plt.title("Feature Relationships Graph")
    plt.savefig("refined_neighbourhood_regression.png")
    plt.show()


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    neighborhood_regression(target=target_1)
