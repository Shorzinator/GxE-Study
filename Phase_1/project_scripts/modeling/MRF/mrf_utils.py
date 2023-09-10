import numpy as np
from sklearn.linear_model import LogisticRegression


def create_mrf_structure(variables):
    """
    Create a fully connected graph structure for the MRF.

    Args:
    - variables: List of variables for the MRF.

    Returns:
    - A list of edges representing the MRF structure.
    """
    edges = []
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            if variables[i] != variables[j]:
                edges.append((variables[i], variables[j]))

    edges = list(set(edges))
    return edges


def unary_potential(data, variable, value):
    """
    Calculate the unary potential for a given variable and value based on its distribution in the dataset.

    Args:
    - data: The dataset.
    - variable: The variable of interest.
    - value: The value of the variable.

    Returns:
    - Unary potential for the given variable and value.
    """
    p = np.mean(data[variable] == value)
    return p


def logistic_regression_pairwise_potential(data, variable1, variable2, covariates=[]):
    """
    Calculate the pairwise potential using logistic regression.

    Args:
    - data: The dataset.
    - variable1: The dependent variable.
    - variable2: The independent variable.
    - covariates: List of covariate variables to be included in the regression.

    Returns:
    - A function that gives pairwise potential based on values of variable1 and variable2.
    """

    # Prepare the data
    X = data[[variable2] + covariates]
    y = data[variable1]

    # Train a logistic regression model
    clf = LogisticRegression(max_iter=10000, multi_class='ovr')
    clf.fit(X, y)

    def potential(value1, value2, covariate_values=[]):
        # Prepare the input data including covariate values
        input_data = [value2] + covariate_values
        prob = clf.predict_proba([input_data])[0]

        # Find the index of the value1 in the classes_ attribute of the classifier
        value1_index = list(clf.classes_).index(value1)

        # Return the probability of variable1 taking value1 given variable2=value2 and covariates
        return prob[value1_index]

    return potential

# Older pairwise potential function (retained for reference)
# def pairwise_potential(data, variable1, variable2, value1, value2):
#    joint_probability = np.mean((data[variable1] == value1) & (data[variable2] == value2))
#    return joint_probability
