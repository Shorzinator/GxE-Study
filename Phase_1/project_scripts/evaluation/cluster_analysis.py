from Phase_1.project_scripts.utility.path_utils import get_path_from_root
from Phase_1.project_scripts.utility.model_utils import *
from Phase_1.project_scripts.utility.data_loader import *
from Phase_1.config import FEATURES

RESULTS_DIR = get_path_from_root("results", "evaluation")

if __name__ == "__main__":
    logger.info("Starting cluster analysis...")

    # Load data
    df = load_data_old()

    # Preprocess the data specific for OvR
    datasets = preprocess_ovr(df, "AntisocialTrajectory")

    # Preparing for interaction terms
    features = FEATURES.copy()

    features.remove("PolygenicScoreEXT")
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]

    for key, X in datasets.items():
        results = []

        logging.info(f"Starting analysis for {key} ...\n")

        for feature_pair in feature_pairs:

            # Performing imputation
            impute = imputation_pipeline()




