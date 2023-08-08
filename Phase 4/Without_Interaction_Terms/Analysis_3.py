# plotting the classification reports per model per outcome

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Get list of files
files = glob.glob('C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\AST_*\\Without IT\\class_report_*.csv')

df_list = []
for file in files:
    df = pd.read_csv(file)
    # The following line extracts model and outcome from the filename.
    # Adjust the indices based on your actual filenames.
    filename_parts = file.split('\\')[-1].replace('class_report_', '').replace('.csv', '').split('_')
    model = filename_parts[0]
    outcome = '_'.join(filename_parts[1:])  # Join all parts other than the first one as the outcome
    df['model'] = model
    df['outcome'] = outcome
    df_list.append(df)

# Concatenate all the dataframes
df = pd.concat(df_list)

# Melt the dataframe to long format for easier plotting
df_melt = df.melt(id_vars=['model', 'outcome'], value_vars=['precision', 'recall', 'f1-score'],
                  var_name='metric', value_name='score')

# Set seaborn style for better looking plots
sns.set(style="whitegrid")


# Set a color palette
palette = sns.color_palette("Set1", 3)  # change the number based on unique metrics

# Iterate over outcomes
for outcome in df_melt['outcome'].unique():
    plt.figure(figsize=(10, 6))
    df_subset = df_melt[df_melt['outcome'] == outcome]
    bar_plot = sns.barplot(data=df_subset, x='model', y='score', hue='metric', palette=palette)

    # Loop over the bars, and adjust the labels
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          size=10,
                          xytext=(0, -12),
                          textcoords='offset points')

    plt.title(f'Metrics for outcome: {outcome}', size=14)
    plt.xlabel('Model', size=12)
    plt.ylabel('Score', size=12)
    plt.legend(title='Metric', title_fontsize='13', loc='upper right')

    plt.show()
