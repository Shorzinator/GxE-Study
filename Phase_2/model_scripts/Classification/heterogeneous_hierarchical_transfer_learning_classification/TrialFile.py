import pandas as pd
from scipy.stats import pearsonr

from utility.data_loader import load_data_old

df = load_data_old()

df.drop(["FamilyID", "ID"], axis=1, inplace=True)
df.dropna(inplace=True)

# Extract columns for PGS and outcomes
pgs = df['PolygenicScoreEXT']
outcome1 = df['AntisocialTrajectory']
outcome2 = df['SubstanceUseTrajectory']

# Calculate Pearson correlation
r1, p1 = pearsonr(pgs, outcome1)
r2, p2 = pearsonr(pgs, outcome2)

# Print results
print(f'Correlation between PGS and Outcome1: r={r1:.3f}, p={p1:.3f}')
print(f'Correlation between PGS and Outcome2: r={r2:.3f}, p={p2:.3f}')

# Check for high correlation
if abs(r1) > 0.7 or abs(r2) > 0.7:
    print('High collinearity detected between PGS and outcomes')
else:
    print('No high collinearity between PGS and outcomes')
