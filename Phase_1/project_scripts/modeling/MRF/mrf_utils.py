import logging

logging.basicConfig(level=logging.INFO)
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


def create_mrf_structure(variables):
    """
    Create a fully connected graph structure for teh MRF

    :param variables: Input values for nodes.
    :return: Edges
    """

