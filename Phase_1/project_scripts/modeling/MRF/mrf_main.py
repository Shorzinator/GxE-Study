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
from Phase_1.project_scripts.preprocessing.mrf_preprocessing import primary_preprocessing_mrf, \
    secondary_preprocessing_without_interaction_mrf
from Phase_1.project_scripts.utility.model_utils import calculate_metrics, ensure_directory_exists, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "markov_random_field"
RESULTS_DIR = get_path_from_root("results", "multi_class", f"{MODEL_NAME}_results")


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

    logger.info("Checking for null values...\n")
    if X_train.isnull().any().any() or y_train.isnull().any().any():
        print("NaN values detected in the training set after secondary preprocessing!")
    if X_val.isnull().any().any() or y_val.isnull().any().any():
        print("NaN values detected in the validation set after secondary preprocessing!")
    if X_test.isnull().any().any() or y_test.isnull().any().any():
        print("NaN values detected in the test set after secondary preprocessing!")
    logger.info("Check for null values complete...\n")

    # Define the MRF graphical model
    logger.info("Defining the MRF graphical model...\n")
    mrf_graph_list = create_mrf_structure(data_preprocessed.columns)
    mrf_graph = nx.Graph(mrf_graph_list)
    model = MarkovNetwork()
    model.add_nodes_from(data_preprocessed.columns)
    model.add_edges_from(mrf_graph.edges())

    # Set up potential functions
    logger.info("Setting up unary potential functions...\n")
    for node in mrf_graph.nodes():
        factor = DiscreteFactor([node], [2],
                                [unary_potential(data_preprocessed, node, 0),
                                 unary_potential(data_preprocessed, node, 1)])
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

    # Inference (as an example)
    logger.info("Performing inference...\n")
    bp = BeliefPropagation(model)

    # Evidence variables for the model. These can be adjusted based on the available data or experiment design.
    evidence_variables = {
        'DelinquentPeer': 1,
        'SchoolConnect': 1,
        'NeighborConnect': 1,
        'ParentalWarmth': 1
    }

    logger.info("Making predictions...\n")
    y_pred = []
    results = []
    if target == "AntisocialTrajectory":
        for index, row in X_test.iterrows():
            evidence = {var: row[var] for var in evidence_variables}
            prediction = bp.map_query(variables=["AntisocialTrajectory"], evidence=evidence)
            y_pred.append(prediction["AntisocialTrajectory"])

        metrics = calculate_metrics(y_test, y_pred, model_name="MRF", target="AntisocialTrajectory",
                                    test_or_train="Test")
        results.append({"test_metrics": metrics})

    if target == "SubstanceUseTrajectory":
        for index, row in X_test.iterrows():
            evidence = {var: row[var] for var in evidence_variables}
            prediction = bp.map_query(variables=["SubstanceUseTrajectory"], evidence=evidence)
            y_pred.append(prediction["SubstanceUseTrajectory"])

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
