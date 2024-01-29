import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, levene, shapiro, stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# Load the dataset
file_path = ("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\Phase_2\\preprocessed_data\\with_PGS\\AST_new"
             "\\X_train_new_AST.csv")
data = pd.read_csv(file_path)

# Identify the race variable and target variable
race_column = 'Race'
# target_variable = 'SubstanceUseTrajectory'
target_variable = 'AntisocialTrajectory'


# Descriptive Statistics and Visualization
def plot_distributions(data, race_column, feature_columns):
    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=race_column, y=feature, data=data)
        title = f'Distribution of {feature} by Race for SUT'
        plt.title(title)
        plt.savefig(f"../results/new_data/{title}.png")
        plt.show()


# Correlation Analysis
def plot_correlations(data, race_column, feature_columns, target_variable):
    for race in data[race_column].unique():
        race_data = data[data[race_column] == race]
        correlations = race_data[feature_columns + [target_variable]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm')
        title = f'Correlation Matrix for {race} for SUT'
        plt.title(title)
        plt.savefig(f"../results/new_data/{title}.png")
        plt.show()


# Advanced Statistical Testing

# Perform Tukey's HSD test for a feature
def tukeys_test(data, race_column, feature):
    # Drop NaN values in the feature and race column
    clean_data = data[[feature, race_column]].dropna()

    # Check for normality in each group
    for race in clean_data[race_column].unique():
        _, p_normal = shapiro(clean_data[clean_data[race_column] == race][feature])
        if p_normal < 0.05:
            print(f"Normality assumption not met for race {race} and feature {feature}.")
            return

    # Check for homogeneity of variances
    race_groups = [group[feature].values for name, group in clean_data.groupby(race_column)]
    _, p_levene = levene(*race_groups)
    if p_levene < 0.05:
        print(f"Homogeneity of variances assumption not met for feature {feature}.")
        return

    # Check for any infinite values and remove them
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure the data type for race column is categorical
    clean_data[race_column] = clean_data[race_column].astype('category')

    # Check if there are enough samples for each group
    if clean_data[race_column].value_counts().min() < 2:
        print(f"Not enough samples for each group in {feature}.")
        return

    tukey = pairwise_tukeyhsd(endog=clean_data[feature], groups=clean_data[race_column], alpha=0.05)
    print(tukey)

    # Ensure there are at least two different groups
    if clean_data[race_column].nunique() >= 2:
        fig = tukey.plot_simultaneous(figsize=(10, 6))
        title = f'Tukey HSD Test for {feature}'
        fig.get_axes()[0].title.set_text(title)
        fig.get_axes()[0].set_xlabel(f'Means for {feature} by {race_column}')
        fig.get_axes()[0].set_ylabel('Group')
        plt.savefig(f"..\\results\\new_data\\{title}.png")
        plt.show()
    else:
        print("Not enough groups for comparison.")


def kruskal_dunn_test(data, race_column, feature):
    # Drop NaN values in the feature and race column
    clean_data = data[[feature, race_column]].dropna()

    # Prepare the data for the Kruskal-Wallis test
    race_groups = [group[feature].values for name, group in clean_data.groupby(race_column)]

    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*race_groups)
    print(f'Kruskal-Wallis test for {feature}: H={stat}, p={p_value}')

    if p_value < 0.05:
        # If the test is significant, perform Dunn's post-hoc test
        print(f"Performing Dunn's post-hoc test for {feature}")
        dunn_pvals = sp.posthoc_dunn(clean_data, val_col=feature, group_col=race_column, p_adjust='bonferroni')

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(dunn_pvals, annot=True, cmap='viridis', linewidths=.5, fmt=".2f")
        title = f'Significant Dunns Test p-values for {feature}'
        plt.title(title)

        # Annotate significant comparisons
        for y in range(dunn_pvals.shape[0]):
            for x in range(dunn_pvals.shape[1]):
                if dunn_pvals.iloc[y, x] < 0.05:
                    plt.text(x + 0.5, y + 0.5, '*', ha='center', va='center', color='red')

        plt.xlabel('Race Group')
        plt.ylabel('Race Group')
        plt.tight_layout()  # Adjust the plot to ensure complete visibility
        plt.savefig(f"..\\results\\new_data\\{title}_X_train_AST.png")
        plt.show()

    else:
        print(f'No significant differences found by Kruskal-Wallis test for {feature}.')


def main():
    # Adjust these feature columns to your dataset
    feature_columns = ['Age', 'DelinquentPeer', 'ParentalWarmth', 'SchoolConnect', 'NeighborConnect']

    # plot_distributions(data, race_column, feature_columns)
    # plot_correlations(data, race_column, feature_columns, target_variable)
    #
    # # Perform ANOVA for each feature across races
    # for feature in feature_columns:
    #     perform_anova(data, race_column, feature)

    # Call this function after the ANOVA test for each feature
    # for feature in feature_columns:
    #     tukeys_test(data, race_column, feature)

    for feature in feature_columns:
        kruskal_dunn_test(data, race_column, feature)

    # clustering_within_races(data, race_column, feature_columns)


if __name__ == "__main__":
    main()
