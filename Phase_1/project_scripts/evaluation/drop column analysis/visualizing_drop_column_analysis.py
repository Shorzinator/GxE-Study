import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Phase_1.project_scripts import get_path_from_root

# Load the data
data_path = get_path_from_root("results", "evaluation", "column_drop_evaluation.csv")
df = pd.read_csv(data_path)

# Select the metrics and group by type and column_dropped
grouped = df.groupby(['type', 'column_dropped']).mean().reset_index()
RESULTS_DIR = get_path_from_root("results", "evaluation")


def plot_spider_chart(grouped, metric):
    # Number of features
    N = len(grouped['column_dropped'].unique())
    theta = [n / float(N) * 2 * 3.141592653589793 for n in range(N)]
    theta += theta[:1]

    # Initialise the spider plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # Plot data for each dataset type
    for key in grouped['type'].unique():
        values = grouped[grouped['type'] == key][metric].values.tolist()
        values += values[:1]
        ax.plot(theta, values, linewidth=2, label=key)
        ax.fill(theta, values, alpha=0.25)

    # Add a title
    plt.title(f'Spider chart of {metric} across datasets', size=15, y=1.1)

    # Configure the labels and legend
    labels = grouped['column_dropped'].unique().tolist() + grouped['column_dropped'].unique().tolist()[:1]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, size=12)
    ax.set_yticklabels([])
    ax.yaxis.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    output_path = get_path_from_root("results", "evaluation", f"spider_chart_{metric}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_joint_plot(df):
    g = sns.jointplot(x="change_in_accuracy", y="change_in_custom_score", data=df, hue="type",
                      height=8, space=0.2, palette="muted")

    g.ax_joint.set_xlabel('Change in Accuracy', fontweight='bold', size=12)
    g.ax_joint.set_ylabel('Change in Custom Score', fontweight='bold', size=12)

    g.ax_joint.legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Joint plot of Change in Accuracy vs Change in Custom Score", fontsize=14)

    output_path = get_path_from_root("results", "evaluation", "joint_plot.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def visualize_results(df):
    """
    Visualizes the change in metrics and saves the plots with enhanced styling and annotations.
    """
    # Set style for the plots
    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    # Function to annotate the bars
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    # Bar plot for change in accuracy
    plt.figure(figsize=(18, 7))
    ax1 = sns.barplot(data=df, x="column_dropped", y="change_in_accuracy", hue="type")
    plt.title("Change in Accuracy by Column Dropped", fontsize=18)
    plt.xlabel("Columns Dropped", fontsize=15)
    plt.ylabel("Change in Accuracy", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title="Dataset Type", loc="upper right", fontsize=12)
    annotate_bars(ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_drop_column.png"))
    plt.close()

    # Bar plot for change in custom score
    plt.figure(figsize=(18, 7))
    ax2 = sns.barplot(data=df, x="column_dropped", y="change_in_custom_score", hue="type")
    plt.title("Change in Custom Score by Column Dropped", fontsize=18)
    plt.xlabel("Columns Dropped", fontsize=15)
    plt.ylabel("Change in Custom Score", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title="Dataset Type", loc="upper right", fontsize=12)
    annotate_bars(ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "custom_score_drop_column.png"))
    plt.close()


plot_spider_chart(grouped, 'change_in_accuracy')
plot_spider_chart(grouped, 'change_in_custom_score')
plot_joint_plot(df)
visualize_results(df)
