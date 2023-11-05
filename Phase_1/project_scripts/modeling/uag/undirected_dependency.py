import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import statsmodels.api as sm

from Phase_1.config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing_without_interaction_mrf
from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists

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
    y = pd.DataFrame(data_preprocessed[target])  # outcome dataset

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
    edge_weights = {}
    edge_labels = {}

    # Define edge colors base on significance
    edge_colors = []

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

    # Graph Construction
    graph = nx.DiGraph()
    graph.add_edges_from(graph_edges)

    # Getting regression coefficients and p-values for edges
    edge_weights = []
    edge_colors = []
    edge_labels = {}
    for edge in graph_edges:
        predictors = data_combined.columns.drop(edge[0]).tolist()
        X = sm.add_constant(data_combined[predictors])
        model = sm.OLS(data_combined[edge[0]], X).fit()
        coef = model.params[edge[1]]
        p_value = model.pvalues[edge[1]]

        edge_weights.append(abs(coef) * 5)  # Scale factor for visualization
        edge_colors.append(1 - min(p_value, 0.1))  # Makes edges with p-value > 0.1 almost transparent

        # Only annotate edges with p-value < 0.05 for clarity
        if p_value < 0.05:
            edge_labels[edge] = f"{coef:.2f}"

    # Create DataFrame for edge data
    edges_df = pd.DataFrame(graph_edges, columns=['From', 'To'])
    edges_df['Weight'] = [edge_labels.get(edge, 'n/s') for edge in graph_edges]
    edges_df['P-value'] = [model.pvalues[edge[1]] for edge in graph_edges]

    # Save to CSV
    edges_df.to_csv(os.path.join(metrics_dir, 'edges_data.csv'), index=False)

    # Adjusting the Layout
    pos = nx.kamada_kawai_layout(graph)

    # Visualization
    node_colors = ["#1f77b4" if node.startswith('G') else "#ff7f0e" if node.startswith('E') else "#2ca02c" for node in
                   graph.nodes()]
    node_sizes = [500 + 2000 * graph.degree(node) for node in graph.nodes()]  # Adjust size based on degree

    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=15,
            width=edge_weights, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1,
            font_weight="bold", alpha=0.9)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, alpha=0.7)
    plt.title("Feature Relationships Graph")
    plt.savefig("neighbourhood_regression_DiGraph.png")
    plt.show()


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    neighborhood_regression(target=target_1)
