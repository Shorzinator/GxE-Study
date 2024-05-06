import pandas as pd

df1 = pd.read_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\updated_raw\\AddH_ML_ld.csv")

df2 = pd.read_csv("C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\raw\\Data_GxE_on_EXT_trajectories_new.csv")

df2 = df2[["AntisocialTrajectory", "SubstanceUseTrajectory", "ID", "PolygenicScoreEXT"]]

# Merge the two dataframes on 'ID'
merged_df = pd.merge(df1, df2, on='ID')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\data\\merged_with_PGS.csv', index=False)
