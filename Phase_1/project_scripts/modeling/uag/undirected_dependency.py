import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import statsmodels.api as sm

from config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.preprocessing.additional_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing
from utility.model_utils import ensure_directory_exists

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
    X_train, y_train, X_test, y_test = secondary_preprocessing(X, y, features)

    # Combining data to replicate processed 'data_processed'
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined.reset_index(drop=True, inplace=True)

    y_combined = pd.concat([y_train, y_test], axis=0)
    y_combined.reset_index(drop=True, inplace=True)

    data_combined = pd.concat([X_combined, y_combined], axis=1)

    # Define positions for three groups of variables G, E, and O
    G_variables = ['PolygenicScoreEXT']  # Assuming this is your genetic variable
    E_variables = X_combined.drop('PolygenicScoreEXT', axis=1).columns
    O_variables = [target]  # The target variable you pass to the function

    # Graph Construction
    graph = nx.Graph()

    # Iterate over all columns as potential targets
    for target_var in data_combined.columns:
        predictors = data_combined.columns.drop(target_var).tolist()
        X = sm.add_constant(data_combined[predictors])
        model = sm.OLS(data_combined[target_var], X).fit()

        # Add significant predictors as edges to the graph
        for predictor in predictors:
            coef = model.params[predictor]
            p_value = model.pvalues[predictor]

            # Add edge only if significant and not a constant term
            if p_value < 0.05 and predictor != 'const':
                graph.add_edge(predictor, target_var, weight=abs(coef) * 5, p_value=p_value, label=f"{coef:.2f}")

    # Prepare for visualization
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    edge_colors = [1 - min(graph[u][v]['p_value'], 0.1) for u, v in
                   graph.edges()]  # Makes edges with p-value > 0.1 almost transparent
    edge_labels = {edge: graph.edges[edge]['label'] for edge in graph.edges() if graph.edges[edge]['p_value'] < 0.05}

    # Create DataFrame for edge data
    edges_df = pd.DataFrame(
        [{'From': u, 'To': v, 'Weight': graph[u][v]['label'], 'P-value': graph[u][v]['p_value']} for u, v in
         graph.edges()])

    # Save to CSV
    edges_df.to_csv(os.path.join(metrics_dir, f'edges_data_{target}.csv'), index=False)

    # Layout 1
    # pos = nx.kamada_kawai_layout(graph)

    # Layout 2
    # Assuming E_variables has at least one item
    min_pos, max_pos = -20, 20
    step = (max_pos - min_pos) / (len(E_variables) - 1) if len(E_variables) > 1 else 0

    spacing = 5  # This value controls the gap. Increase it to space out nodes more.
    pos = {node: (0, i + spacing) for i, node in enumerate(G_variables)}
    pos.update({node: (1, i) for i, node in enumerate(E_variables)})
    pos.update({node: (2, i + spacing) for i, node in enumerate(O_variables)})

    initial_pos = pos

    # Layout 3
    # pos = nx.spring_layout(graph, pos=initial_pos, k=2.0 / np.sqrt(graph.number_of_nodes()), iterations=50)

    # Visualization
    node_colors = ["#1f77b4" if node in G_variables else "#ff7f0e" if node in E_variables else "#2ca02c" for node in
                   graph.nodes()]
    node_sizes = [100 + 5 * graph.degree(node) for node in graph.nodes()]  # Adjust size based on degree

    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=15,
            width=edge_weights, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1,
            font_weight="bold", alpha=0.9)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, alpha=0.7)
    plt.title(f"Feature Relationships Graph for {target}")
    plt.savefig(f"{target}_neighborhood_regression_layers.png")
    plt.show()


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    neighborhood_regression(target=target_1)
