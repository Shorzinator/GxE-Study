import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)


def basic_info(data):
    """
    Print basic information and display first few rows of the dataset.
    """
    print(data.info())
    print(data.head())


def plot_distribution(data, column, title):
    """
    Plot the distribution of a specified column.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


def missing_values_analysis(data):
    """
    Analyze and print the missing values in the dataset.
    """
    missing_values = data.isnull().sum()
    print("Missing Values:\\n", missing_values)


def plot_correlation_matrix(data):
    """
    Plot the correlation matrix for the dataset.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Variables')
    plt.show()


def handle_missing_values(data):
    """
    Handle missing values by dropping columns with >50% missing and imputing others.
    """
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    columns_to_drop = missing_percentage[missing_percentage > 50].index
    data_cleaned = data.drop(columns=columns_to_drop)
    data_imputed = data_cleaned.fillna(data_cleaned.median())
    return data_imputed


def subgroup_analysis(data, group_column):
    """
    Perform a subgroup analysis based on a specified grouping column.
    """
    grouped_data = data.groupby(group_column).mean()
    return grouped_data


# Example usage
if __name__ == "__main__":
    file_path = 'C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\raw\\Data_GxE_on_EXT_trajectories_new.csv'
    data = load_data(file_path)
    # basic_info(data)
    plot_distribution(data, 'Race', 'Distribution of Races in the Dataset')
    plot_distribution(data, 'AntisocialTrajectory', 'Distribution of Antisocial Trajectory')
    plot_distribution(data, 'SubstanceUseTrajectory', 'Distribution of Substance Use Trajectory')
    missing_values_analysis(data)
    plot_correlation_matrix(data)
    data_imputed = handle_missing_values(data)
    race_analysis = subgroup_analysis(data_imputed, 'Race')
    print("Subgroup Analysis by Race:\\n", race_analysis.head())
