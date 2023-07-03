# Summary for ASB_Analysis.py

1. **Loading the Dataset:** The first step was to load the data from a CSV file into a pandas DataFrame. The dataset contained various features along with the target variable 'AntisocialTrajectory'.
    - Similiar code will be repeated for 'SubstanceUseBehaviour' with minor changes.

2. **Preprocessing the Data:** After loading the data, the next step was preprocessing, which included:
    - Handling missing values with K-Nearest Neighbors Imputation.
    	- I used '_n_neighbours = 4_' but it can be tuned to explore changes in results.
    	- Experimental iterativeimputer would be used in case of SUB
    - Converting the '_Sex_' variable into a binary '_Is_Male_' variable, indicating 1 for males and 0 for females.
    	- Initially I created two separate dummy variables for male and female but that was resulting in high collinearity and low accuracy.
    - Defining the feature set (X) and the target variable (y). Here, certain columns like 'ID', 'FamilyID', 'AntisocialTrajectory', 'SubstanceUseTrajectory', 'Sex' were dropped from X.
    - Subtracting 1 from the 'AntisocialTrajectory' labels to standardize the labels to start from 0.

3. **Handling Errors:**
	- **KeyError during imputation:**
		- Cause: The KeyError occurred because the imputation was being performed directly on the DataFrame, and it seemed like the KNNImputer was not able to handle non-numeric columns.
		- Resolution: To resolve this, the DataFrame was converted into a numpy array before performing imputation. After imputation, the numpy array was converted back to a DataFrame. This was achieved by replacing:

	- **MemoryError during model evaluation:**
		- Cause: The MemoryError was caused by the creation of a large number of interaction terms for each feature combination, resulting in an extensive memory requirement for storing the feature matrix. This was compounded by the fact that all combinations of interaction terms and models were being evaluated at once.
		- Resolution: Instead of storing the entire feature matrix with all interaction terms in memory, the code was modified to create interaction terms one-by-one for each model and interaction term pair. This reduces the memory requirement by not keeping all
