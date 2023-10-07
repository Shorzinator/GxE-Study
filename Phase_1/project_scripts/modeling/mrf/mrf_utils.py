import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_nan_values(data, context_msg=""):
    """Utility function to check and log NaN values."""
    nan_columns = data.columns[data.isnull().any()].tolist()
    if nan_columns:
        nan_percentage = data[nan_columns].isnull().mean() * 100
        for column, percentage in nan_percentage.items():
            logger.warning(f"NaN percentage in column '{column}' ({context_msg}): {percentage:.2f}%")
        return True
    return False


def create_mrf_structure(variables, genetic_variable, outcome_variables):
    """
    Create a graph structure for the MRF. The structure is designed such that:
    1. All environmental nodes are fully connected.
    2. The genetic node is connected to every environmental node.
    3. The outcome nodes are connected to every environmental and the genetic node.
    """
    edges = []

    # Connecting environmental variables among themselves
    env_vars = [v for v in variables if v not in outcome_variables and v != genetic_variable]
    for i in range(len(env_vars)):
        for j in range(i+1, len(env_vars)):
            edges.append((env_vars[i], env_vars[j]))

    # Connecting genetic variable to every environmental variable
    for var in env_vars:
        edges.append((genetic_variable, var))

    # Connecting outcome variables to every other node (genetic + environmental)
    for outcome in outcome_variables:
        for var in variables:
            if var != outcome:
                edges.append((outcome, var))

    return list(set(edges))


def unary_potential(data, variable, value):
    """
    Calculate the unary potential for a given variable and value based on its distribution in the dataset
    """
    if check_nan_values(data[[variable]], f"unary_potential for {variable}"):
        logger.error(f"NaN values detected in data for unary_potential calculation for variable {variable}.")
        raise ValueError(f"NaN values detected in data for unary_potential calculation for variable {variable}.")

    data_clean = data.dropna(subset=[variable])
    p = np.mean(data_clean[variable] == value)
    return p


def logistic_regression_pairwise_potential(data, variable1, variable2):
    """
    Calculate the pairwise potential using logistic regression if both nodes are categorical.
    If one of the nodes is continuous, return a dummy potential function.
    """

    if data[variable1].dtype == 'float64' or data[variable2].dtype == 'float64':
        # If one of the variables is continuous, we return a dummy potential function
        def continuous_dummy_potential(value1, value2):
            return 1
        return continuous_dummy_potential

    X = data[[variable2]]
    y = data[variable1]

    if check_nan_values(X, f"logistic_regression_pairwise_potential for X with nodes {variable1} and {variable2}"):
        logger.error(f"Nan values detected in y before pairwise potential calculation for node {variable1}")
        raise ValueError(f"Nan values detected in y before pairwise potential calculation for node {variable1}")

    clf = LogisticRegression(max_iter=10000, multi_class="ovr")
    clf.fit(X, y)

    # Log some details about the logistic regression model
    logger.debug(f"Classes for {variable1} as predicted by logistic regression: {clf.classes_}")  # ADDED

    def potential(value1, value2):
        prob = clf.predict_proba([[value2]])[0]
        value1_index = list(clf.classes_).index(value1)
        return prob[value1_index]

    return potential
