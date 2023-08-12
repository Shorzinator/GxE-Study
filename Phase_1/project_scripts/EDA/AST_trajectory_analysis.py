import matplotlib.pyplot as plt
import seaborn as sns

# Importing utility functions
from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.path_utils import get_path_from_root


def analyze_trajectory_distribution():
    # Load the dataset
    df = load_data()

    # Plot the distribution of AntisocialTrajectory
    plt.figure(figsize=(10, 7))
    ax = sns.countplot(data=df, x='AntisocialTrajectory', palette="viridis")
    plt.title('Distribution of AntisocialTrajectory Categories', fontsize=15)
    plt.xlabel('Antisocial Trajectory Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    # Annotate counts on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig(get_path_from_root("results", "figures",  "eda_plots", "AST_trajectory_distribution.png"))
    plt.show()

    # Examine the association between AntisocialTrajectory and focal environmental variables
    focal_variables = ['Age', 'Race', 'PolygenicScoreEXT', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect',
                       'ParentalWarmth']

    for var in focal_variables:
        plt.figure(figsize=(10, 7))
        if var == "Race":
            sns.barplot(data=df, x='AntisocialTrajectory', y=var, estimator=len, palette="viridis")
        else:
            sns.boxplot(data=df, x='AntisocialTrajectory', y=var, order=df.groupby('AntisocialTrajectory')[var].median().sort_values().index, palette="viridis")
        plt.title(f'{var} Distribution by AntisocialTrajectory Category', fontsize=15)
        plt.xlabel('Antisocial Trajectory Category', fontsize=12)
        plt.ylabel(var, fontsize=12)
        plt.savefig(get_path_from_root("results", "figures", "eda_plots", "trajectory_analysis", f"AST_boxplot_{var}.png"))
        plt.show()


if __name__ == "__main__":
    analyze_trajectory_distribution()
