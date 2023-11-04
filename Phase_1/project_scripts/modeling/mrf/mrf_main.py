import logging
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import MarkovNetwork
from tqdm import tqdm

from Phase_1.config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.modeling.mrf.mrf_utils import create_mrf_structure, logistic_regression_pairwise_potential, \
    unary_potential
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing_without_interaction_mrf
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, save_results

pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "markov_random_field"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")


def get_evidence_bin(value, bin_edges):
    """
    Get the bin index for the given value based on bin edges.
    """
    bin_index = np.digitize(value, bin_edges) - 1
    return min(bin_index, len(bin_edges) - 2)  # ensure the bin_index doesn't exceed the number of bins - 1


def main(target):
    logger.info(f"Starting Markov Random Field for {target}...\n")

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

    # Compute the correlation_matrix
    correlation_matrix = data_combined.corr()

    # Determine which pairs of features have a correlation below a threshold
    threshold = 0.2
    weakly_correlated_pairs = [(col1, col2) for col1 in correlation_matrix.columns
                               for col2 in correlation_matrix.columns
                               if abs(correlation_matrix.loc[col1, col2]) < threshold]

    # Define the MRF graphical model
    logger.info("Defining the MRF graphical model...\n")
    outcome_variables = [target]
    mrf_graph_list = create_mrf_structure(features, "PolygenicScoreEXT", outcome_variables)
    mrf_graph_list = [edge for edge in mrf_graph_list if edge not in weakly_correlated_pairs]
    mrf_graph = nx.Graph(mrf_graph_list)
    model = MarkovNetwork()
    model.add_nodes_from(data_combined.columns)
    nx.draw(mrf_graph, with_labels=True)

    # path_for_mrf = get_path_from_root("results", "multi_class", "markov_random_field_results")
    # plt.savefig(os.path.join(path_for_mrf, 'mrf_graph.png'), dpi=300)

    # Convert the edge list DataFrame
    # edge_df = pd.DataFrame(list(mrf_graph.edges()), columns=['From', 'To'])
    # edge_df.to_csv(os.path.join(path_for_mrf, 'graph_edge_list.csv'), index=False)

    # Debugging step 2
    # Cardinality of a node refers to the number of possible states or values that the variable can take on.
    for variable in model.nodes():
        # print(variable, model.get_cardinality(variable))
        pass

    # Set up potential functions
    logger.info("Setting up unary potential functions...\n")
    for node in tqdm(mrf_graph.nodes(), desc="Setting up unary potentials"):
        states = len(data_combined[node].unique())
        factor_values = [unary_potential(data_combined, node, i) for i in range(states)]
        factor = DiscreteFactor([node], [states], factor_values)
        model.add_factors(factor)

    logger.info("Unary potential functions have been set up...\n")

    factor_PGS = model.get_factors('PolygenicScoreEXT')
    factor_Age = model.get_factors('Age')

    factor_PGS = factor_PGS[0]
    factor_Age = factor_Age[0]

    interaction_factor_values = np.outer(factor_PGS.values, factor_Age.values).flatten()
    states_PGS = len(data_combined['PolygenicScoreEXT'].unique())
    states_Age = len(data_combined['Age'].unique())
    interaction_factor = DiscreteFactor(['PolygenicScoreEXT', 'Age'], [states_PGS, states_Age],
                                        interaction_factor_values)
    model.add_factors(interaction_factor)

    print()

    # Pairwise potentials using logistic regression
    logger.info("Setting up pairwise potential functions using logistic regression...\n")
    edges = list(mrf_graph.edges())
    batch_size = 100

    for i in tqdm(range(0, len(edges), batch_size), desc="Setting up pairwise potentials"):
        batch_edges = edges[i:i + batch_size]
        for edge in batch_edges:
            node1, node2 = edge
            potential_function = logistic_regression_pairwise_potential(data_preprocessed, node1, node2)
            states_node1 = len(data_combined[node1].unique())
            states_node2 = len(data_combined[node2].unique())
            factor_values = [
                [potential_function(i, j) for j in range(states_node2)] for i in range(states_node1)
            ]
            factor = DiscreteFactor([node1, node2], [states_node1, states_node2], factor_values)
            model.add_factors(factor)

    logger.info("Pairwise potential functions using logistic regression have been set up...\n")

    # Checking if factors for all variables have been defined or not
    defined_factors_vars = set(var for factor in model.factors for var in factor.scope())
    all_vars = set(model.nodes())

    missing_factors_vars = all_vars - defined_factors_vars

    if missing_factors_vars:
        logger.error(f"Factors not defined for variables: {missing_factors_vars}")

    # Inference (as an example)
    logger.info("Performing inference...\n")
    bp = BeliefPropagation(model)

    # Define bin edges for discretized features
    n_bins = 10  # For example, if you want five bins
    features_to_bin = ['DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth', 'PolygenicScoreEXT',
                       'Age']

    continuous_feature_bins = {feature: np.histogram_bin_edges(data_combined[feature], bins=n_bins) for feature in
                               features_to_bin}

    logger.info(f"Bin edges:\n{continuous_feature_bins}")

    logger.info("Making predictions...\n")

    y_pred = []
    results = []

    for index, row in X_test.iterrows():
        evidence = {}
        for feature in features_to_bin:
            bin_value = get_evidence_bin(row[feature], continuous_feature_bins[feature])
            evidence[feature] = bin_value

            # Print out the bins assigned to values of PolygenicScoreEXT_x_Age
            if feature == 'PolygenicScoreEXT_x_Age' and bin_value == 4:
                print(f"Value: {row[feature]}, Bin: {bin_value}")

        for col, value in row.items():
            if value >= 2:  # Because we are getting error for value '3', let's check for all values >= 2
                print(f"Feature: {col}, Value in X_test: {value}")

        logger.debug(f"Evidence for row {index}:\n{evidence}")

        if target == "AntisocialTrajectory":
            prediction = bp.map_query(variables=["AntisocialTrajectory"], evidence=evidence)
            y_pred.append(prediction["AntisocialTrajectory"])
        elif target == "SubstanceUseTrajectory":
            prediction = bp.map_query(variables=["SubstanceUseTrajectory"], evidence=evidence)
            y_pred.append(prediction["SubstanceUseTrajectory"])

    if target == "AntisocialTrajectory":
        metrics = calculate_metrics(y_test, y_pred, model_name="MRF", target="AntisocialTrajectory",
                                    test_or_train="Test")
        results.append({"test_metrics": metrics})
    elif target == "SubstanceUseTrajectory":
        metrics = calculate_metrics(y_test, y_pred, model_name="MRF", target="SubstanceUseTrajectory",
                                    test_or_train="Test")
        results.append({"test_metrics": metrics})

    # Save results using the provided save_results function
    save_results(target, "mrf", results, metrics_dir, interaction=False, model_name=MODEL_NAME)

    logger.info(f"Markov Random Field modeling for {target} completed.\\n")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    main(target=target_1)
