import logging
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import MarkovNetwork

from Phase_1.config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.modeling.MRF.mrf_utils import create_mrf_structure, logistic_regression_pairwise_potential, \
    unary_potential
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing_without_interaction_mrf
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "markov_random_field"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")


def get_evidence_bin(value, bin_edges):
    """
    Get the bin index for the given value based on bin edges.
    """
    return np.digitize(value, bin_edges) - 1


def main(target):
    logger.info(f"Starting Markov Random Field for {target}...\n")

    # Subdirectories for metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    ensure_directory_exists(metrics_dir)

    # Load the data
    df = load_data_old()

    # Preprocess data
    if target == "AntisocialTrajectory":
        features_to_use = FEATURES_FOR_AST
    else:
        features_to_use = FEATURES_FOR_SUT

    # Applying primary preprocessing
    data_preprocessed, features = primary_preprocessing_mrf(df, features_to_use, target)

    X = pd.DataFrame(data_preprocessed[features])   # feature dataset
    y = pd.DataFrame(data_preprocessed[target])     # outcome dataset

    # Applying secondary preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test = secondary_preprocessing_without_interaction_mrf(X, y, features)

    # Combining data to replicate processed 'data_processed'
    X_combined = pd.concat([X_train, X_val, X_test], axis=0)
    y_combined = pd.concat([y_train, y_val, y_test], axis=0)
    data_combined = pd.concat([X_combined, y_combined], axis=1)

    # Define the MRF graphical model
    logger.info("Defining the MRF graphical model...\n")
    outcome_variables = ["AntisocialTrajectory", "SubstanceUseTrajectory"]
    mrf_graph_list = create_mrf_structure(features, "PolygenicScoreEXT", outcome_variables)
    mrf_graph = nx.Graph(mrf_graph_list)
    model = MarkovNetwork()
    model.add_nodes_from(data_combined.columns)
    model.add_edges_from(mrf_graph.edges())

    # Set up potential functions
    logger.info("Setting up unary potential functions...\n")
    for node in mrf_graph.nodes():
        factor = DiscreteFactor([node], [2],
                                [unary_potential(data_combined, node, 0),
                                 unary_potential(data_combined, node, 1)])
        model.add_factors(factor)

    # Pairwise potentials using logistic regression
    logger.info("Setting up pairwise potential functions using logistic regression...\n")
    for edge in mrf_graph.edges():
        node1, node2 = edge
        potential_function = logistic_regression_pairwise_potential(data_preprocessed, node1, node2)

        factor_values = [
            [potential_function(i, j) for j in [0, 1]] for i in [0, 1]
        ]

        factor = DiscreteFactor([node1, node2], [2, 2], factor_values)
        model.add_factors(factor)

    for variable in model.nodes():
        print(variable, model.get_cardinality(variable))

    # Inference (as an example)
    logger.info("Performing inference...\n")
    bp = BeliefPropagation(model)

    # Define bin edges for discretized features
    n_bins = 5  # For example, if you want five bins
    features_to_bin = ['DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth', 'PolygenicScoreEXT',
                       'Age']

    continuous_feature_bins = {feature: np.histogram_bin_edges(data_combined[feature], bins=n_bins) for feature in
                               features_to_bin}

    logger.info("Making predictions...\n")

    y_pred = []
    results = []

    for index, row in X_test.iterrows():
        evidence = {}
        for feature in features_to_bin:
            evidence[feature] = get_evidence_bin(row[feature], continuous_feature_bins[feature])

        print(evidence)

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
    save_results(target, "binary", results, metrics_dir, interaction=False, model_name=MODEL_NAME)

    logger.info(f"Markov Random Field modeling for {target} completed.\\n")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    main(target=target_1)
