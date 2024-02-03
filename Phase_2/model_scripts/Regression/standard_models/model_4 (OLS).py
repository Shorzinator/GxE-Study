import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

from utility.model_utils import ensure_directory_exists
from utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "neighborhood_regression"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")


def load_data(file_path):
    return pd.read_csv(file_path)


def evaluate_model(y_true, y_pred):
    """
    Evaluate the model on various metrics.

    :param y_true: True target values.
    :param y_pred: Predicted target values.
    :return: Dictionary of various evaluation metrics.
    """
    return {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }


def ols(target):
    logger.info(f"Starting Neighborhood Regression for {target}...\n")

    # Subdirectories for metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    ensure_directory_exists(metrics_dir)

    X_train = load_data("../../../preprocessed_data/with_PGS/AST_old/X_train_old_AST.csv")
    X_test = load_data("../../../preprocessed_data/with_PGS/AST_old/X_test_old_AST.csv")
    y_train = load_data("../../../preprocessed_data/with_PGS/AST_old/y_train_old_AST.csv")
    y_test = load_data("../../../preprocessed_data/with_PGS/AST_old/y_test_old_AST.csv")

    # Combine the data
    X_combined = pd.concat([X_train, X_test]).fillna(0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_test]).reset_index(drop=True)

    # Combine into one dataframe
    data_combined = pd.concat([X_combined, y_combined], axis=1)

    # Define positions for three groups of variables G, E, and O
    G_variables = ['PolygenicScoreEXT'] if 'PolygenicScoreEXT' in X_combined.columns else []
    E_variables = [var for var in X_combined.columns if var not in G_variables and var != 'AntisocialTrajectory']
    O_variables = [target]

    # Graph Construction
    graph = nx.Graph()

    # Iterate over all columns as potential targets
    for target_var in O_variables + E_variables:
        predictors = data_combined.columns.drop(target_var).tolist()
        X = sm.add_constant(data_combined[predictors])
        model = sm.OLS(data_combined[target_var], X).fit()

        y_pred = model.predict(X)
        y_true = data_combined[target_var]

        # Evaluate the model
        evaluation_results = evaluate_model(y_true, y_pred)
        print(f"Evaluation results for target {target_var}:", evaluation_results)

        # Add significant predictors as edges to the graph
        for predictor in predictors:
            coef = model.params[predictor]
            p_value = model.pvalues[predictor]

            # Add edge only if significant and not a constant term
            if p_value < 0.05 and predictor != 'const':
                graph.add_edge(predictor, target_var, weight=abs(coef) * 5, p_value=p_value, label=f"{coef:.2f}")

    logger.info("Prediction Done.\n")

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

    # Assuming E_variables has at least one item
    spacing = 5  # This value controls the gap. Increase it to space out nodes more.
    pos = {node: (0, i + spacing) for i, node in enumerate(G_variables)}
    pos.update({node: (1, i) for i, node in enumerate(E_variables)})
    pos.update({node: (2, i + spacing) for i, node in enumerate(O_variables)})

    # Visualization
    node_colors = ["#1f77b4" if node in G_variables else "#ff7f0e" if node in E_variables else "#2ca02c" for node in
                   graph.nodes()]

    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=600, node_color=node_colors, font_size=20,
            width=edge_weights, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1,
            font_weight="bold", alpha=0.9)

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, alpha=0.7)

    plt.title(f"Feature Relationships Graph for {target}")
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()  # Adjust the layout
    # plt.savefig(f"../results/modeling/{target}_neighborhood_regression_layers.jpg", dpi=600)
    plt.show()


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    ols(target=target_1)