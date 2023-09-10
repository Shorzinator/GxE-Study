import logging
import os
import warnings

import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import MarkovNetwork

from Phase_1.config import FEATURES_FOR_AST, FEATURES_FOR_SUT
from Phase_1.project_scripts import get_path_from_root, load_data_old
from Phase_1.project_scripts.modeling.MRF.mrf_utils import create_mrf_structure, logistic_regression_pairwise_potential, \
    unary_potential
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import apply_preprocessing_without_interaction_terms_mrf, \
    preprocess_for_mrf
from Phase_1.project_scripts.utility.model_utils import ensure_directory_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "markov_random_field"
RESULTS_DIR = get_path_from_root("results", "one_vs_all", f"{MODEL_NAME}_results")


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
    data_preprocessed, features = preprocess_for_mrf(df, features_to_use, target)

    X = pd.DataFrame(data_preprocessed[features])   # feature dataset
    y = pd.DataFrame(data_preprocessed[target])     # outcome dataset

    # Applying secondary preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test = apply_preprocessing_without_interaction_terms_mrf(X, y, features)

    # Define the MRF graphical model
    mrf_graph_list = create_mrf_structure(data_preprocessed.columns)
    mrf_graph = nx.Graph(mrf_graph_list)
    model = MarkovNetwork()
    model.add_nodes_from(data_preprocessed.columns)
    model.add_edges_from(mrf_graph.edges())

    # Set up potential functions
    for node in mrf_graph.nodes():
        factor = DiscreteFactor([node], [2],
                                [unary_potential(data_preprocessed, node, 0),
                                 unary_potential(data_preprocessed, node, 1)])
        model.add_factors(factor)

    # Pairwise potentials using logistic regression
    for edge in mrf_graph.edges():
        node1, node2 = edge
        potential_function = logistic_regression_pairwise_potential(data_preprocessed, node1, node2)

        factor_values = [
            [potential_function(i, j) for j in [0, 1]] for i in [0, 1]
        ]

        factor = DiscreteFactor([node1, node2], [2, 2], factor_values)
        model.add_factors(factor)

    # Inference (as an example)
    bp = BeliefPropagation(model)

    # Evidence variables for the model. These can be adjusted based on the available data or experiment design.
    evidence_variables = {
        'DelinquentPeer': 1,
        'SchoolConnect': 1,
        'NeighborConnect': 1,
        'ParentalWarmth': 1
    }

    if target == "AntisocialTrajectory":
        prediction_AST = bp.query(variables=["AntisocialTrajectory"], evidence=evidence_variables)
        print("Predicted Marginals for AST:", prediction_AST)

    if target == "SubstanceUseTrajectory":
        prediction_SUT = bp.query(variables=["SubstanceUseTrajectory"], evidence=evidence_variables)
        print("Predicted Marginals for SUT:", prediction_SUT)

    # Save results
    # Placeholder: results should be the metrics or any other relevant data you want to save
    # results = {}
    # save_results(target, "MRF", results, metrics_dir, MODEL_NAME)

    logger.info(f"Markov Random Field modeling for {target} completed.\n")


if __name__ == '__main__':
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"
    main(target=target_1)
