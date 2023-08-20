import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from Phase_1.project_scripts.utility.path_utils import get_path_from_root
from Phase_1.project_scripts.utility.model_utils import *
from Phase_1.project_scripts.utility.data_loader import *
from Phase_1.config import FEATURES

RESULTS_DIR = get_path_from_root("results", "evaluation", "cluster_analysis")
ensure_directory_exists(RESULTS_DIR)


def optimal_number_of_clusters(wcss):
    x1, y1 = 1, wcss[0]
    x2, y2 = len(wcss), wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 1


def plot_elbow(X_scaled):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    n_optimal = optimal_number_of_clusters(wcss)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.axvline(x=n_optimal, color='red', linestyle='--')
    plt.title('K-means Elbow Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join(RESULTS_DIR, "elbow_plot.png"))
    plt.close()


def plot_clusters(X, clusters, feature_pair, pca_transformed_data, centroids_pca):
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], color='red', s=150, marker='X')
    plt.title(f"Clusters for Interaction: {feature_pair[0]} x {feature_pair[1]}")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.colorbar()
    filename = f"clusters_{feature_pair[0]}_x_{feature_pair[1]}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


if __name__ == "__main__":
    logger.info("Starting cluster analysis...")

    df = load_data_old()
    X = preprocess_general(df, "AntisocialTrajectory")

    features = FEATURES.copy()
    features.remove("PolygenicScoreEXT")
    fixed_element = "PolygenicScoreEXT"

    feature_pairs = [(fixed_element, x) for x in features if x != fixed_element]
    results = []

    for feature_pair in feature_pairs:
        logging.info(f"Starting analysis for {feature_pair} ...\n")

        impute = imputation_pipeline()
        X_imputed = imputation_applier(impute, X)

        X_final = add_interaction_terms(X_imputed, feature_pair)
        transformed_columns = X_final.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_final)

        plot_elbow(X_scaled)

        kmeans = KMeans(n_clusters=3, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        pca_transformed_data = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(kmeans.cluster_centers_)

        plot_clusters(X, clusters, feature_pair, pca_transformed_data, centroids_pca)

        # Extract cluster statistics
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)

        cluster_summary = {
            "interaction term": f"{feature_pair[0]}_x_{feature_pair[1]}",
            "silhouette_score": silhouette_score(X_scaled, clusters),
        }

        cluster_stats = X_final.groupby(clusters).agg(['mean', 'std'])
        for i in range(3):  # for each cluster
            for column in X_final.columns:
                cluster_summary[f"cluster_{i}_{column}_mean"] = cluster_stats[column]['mean'].iloc[i]
                cluster_summary[f"cluster_{i}_{column}_std"] = cluster_stats[column]['std'].iloc[i]

        results.append(cluster_summary)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "clustering_analysis.csv"))
    logging.info("Clustering analysis completed...")