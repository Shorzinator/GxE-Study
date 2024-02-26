import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from statsmodels.multivariate.manova import MANOVA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Define functions for each test

def perform_mann_whitney_u_test(old_data, new_data, feature):
    stat, p_value = mannwhitneyu(old_data, new_data)
    print(f"Mann-Whitney U Test for {feature}: p-value = {p_value}")


def perform_ks_test(old_data, new_data, feature):
    stat, p_value = ks_2samp(old_data, new_data)
    print(f"Kolmogorov-Smirnov Test for {feature}: p-value = {p_value}\n")


def perform_manova(old_data, new_data, features):
    combined_data = pd.concat([old_data, new_data])
    formula = ' + '.join(features) + ' ~ Group'
    manova = MANOVA.from_formula(formula, data=combined_data)
    print(manova.mv_test())


def perform_pca_and_clustering(data, features, n_clusters=2):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data[features].dropna())
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_results)
    print("Cluster centers:", kmeans.cluster_centers_)


def main():
    # Load the datasets
    new_data_path = '../../data/raw/Data_GxE_on_EXT_trajectories_new.csv'
    old_data_path = '../../data/raw/Data_GxE_on_EXT_trajectories_old.csv'

    new_data = pd.read_csv(new_data_path)
    old_data = pd.read_csv(old_data_path)

    # Filter the new data for Race 1.0 and assign group labels
    new_data_race_1 = new_data[new_data['Race'] == 1.0].assign(Group='New')
    old_data = old_data.assign(Group='Old')

    # Specify the features to test (excluding PGS)
    features_to_test = ['Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth']

    # Perform tests
    for feature in features_to_test:
        old_data_feature = old_data[feature].dropna()
        new_data_race_1_feature = new_data_race_1[feature].dropna()
        perform_mann_whitney_u_test(old_data_feature, new_data_race_1_feature, feature)
        perform_ks_test(old_data_feature, new_data_race_1_feature, feature)

    # MANOVA
    perform_manova(old_data[features_to_test + ['Group']], new_data_race_1[features_to_test + ['Group']],
                   features_to_test)

    # PCA and K-Means Clustering
    combined_data = pd.concat([old_data[features_to_test], new_data_race_1[features_to_test]])
    perform_pca_and_clustering(combined_data, features_to_test)


if __name__ == "__main__":
    main()


# Results -
#
# Mann-Whitney U Test for Age: p-value = 0.0352786562784368
# Kolmogorov-Smirnov Test for Age: p-value = 0.2017307673768053
#
# Mann-Whitney U Test for DelinquentPeer: p-value = 0.24093844781732876
# Kolmogorov-Smirnov Test for DelinquentPeer: p-value = 0.5891553867735705
#
# Mann-Whitney U Test for SchoolConnect: p-value = 0.4622205221094181
# Kolmogorov-Smirnov Test for SchoolConnect: p-value = 0.9884613190812435
#
# Mann-Whitney U Test for NeighborConnect: p-value = 0.0025124597077828214
# Kolmogorov-Smirnov Test for NeighborConnect: p-value = 0.040047451949915325
#
# Mann-Whitney U Test for ParentalWarmth: p-value = 0.3572160286036602
# Kolmogorov-Smirnov Test for ParentalWarmth: p-value = 0.9698738412614925
#
#                    Multivariate linear model
# ===============================================================
#
# ---------------------------------------------------------------
#        Intercept        Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9913 5.0000 21780.0000 38.3854 0.0000
#          Pillai's trace 0.0087 5.0000 21780.0000 38.3854 0.0000
#  Hotelling-Lawley trace 0.0088 5.0000 21780.0000 38.3854 0.0000
#     Roy's greatest root 0.0088 5.0000 21780.0000 38.3854 0.0000
# ---------------------------------------------------------------
#
# ---------------------------------------------------------------
#          Group          Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9992 5.0000 21780.0000  3.6580 0.0026
#          Pillai's trace 0.0008 5.0000 21780.0000  3.6580 0.0026
#  Hotelling-Lawley trace 0.0008 5.0000 21780.0000  3.6580 0.0026
#     Roy's greatest root 0.0008 5.0000 21780.0000  3.6580 0.0026
# ===============================================================
#
# Cluster centers: [[ 1.26622414 -0.02199232]
#  [-0.89958878  0.01562444]]
