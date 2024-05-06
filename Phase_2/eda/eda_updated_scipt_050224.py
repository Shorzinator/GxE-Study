import pandas as pd
pd.set_option("display.max_rows", None)

df = pd.read_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\merged_without_PGS.csv")

na_counts = df.isna().sum()
na_dict = na_counts.to_dict()
filtered_dict = {k: v for k, v in na_dict.items() if v != 0}
sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))

# print(sorted_dict)
# {'AntisocialTrajectory': 106, 'SubstanceUseTrajectory': 106, 'H1GI4': 62, 'H1GI6A': 41, 'H1GI6B': 41, 'H1GI6C': 41, 'H1GI6D': 41, 'H1GI6E': 41, 'H1ED9': 41, 'H1GI1Y': 17, 'H1TO10': 10, 'H1TO11': 10, 'H1TO40': 8, 'H1ED7': 7, 'H1TO37': 5, 'H1TO9': 4, 'H1TO29': 4, 'H1ED2': 3, 'H1FP13B2': 3, 'H1TO41': 3, 'BIO_SEX': 2, 'H1TO34': 2, 'H1TO42': 2, 'H1TO43': 2, 'H1TO45': 2, 'H1CO10': 1, 'H1WS3A': 1, 'H1WS5A': 1, 'H1TO2': 1, 'H1TO4': 1, 'H1TO17': 1, 'H1TO18': 1}

# print(len(df))    # 20745

mask = df.isna()
unique_ids_with_na = df.loc[mask.any(axis=1), "ID"].unique()
count = len(unique_ids_with_na)
# print(count)  # 284
# 1.37 % of unique IDs have a missing value

# print(df.shape)
