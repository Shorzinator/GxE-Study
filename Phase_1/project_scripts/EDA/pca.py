from Phase_1.project_scripts.utility.data_loader import load_data
from Phase_1.project_scripts.utility.path_utils import get_path_from_root
from Phase_1.project_scripts.preprocessing.preprocessing import preprocess_data, imputation_pipeline, split_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

output_dir_plots = get_path_from_root("results", "figures", "eda_plots", "pca")

df = load_data()
temp = input("AST or SUT?:\n")

df_processed = preprocess_data(df, temp)
target = "AntisocialTrajectory" if temp == "AST" else "SubstanceUseTrajectory"
X, y = df_processed.drop(target, axis=1), df_processed[target]

# Splitting the data
X_train, X_test, y_train, y_test = split_data(df_processed, target)

# Apply imputation
preprocessor = imputation_pipeline(X_train)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# You're only doing PCA for visualization, so consider just using the train set
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_train_transformed)

plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_train)
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

plot_filename = os.path.join(output_dir_plots, "pca_results.png")

# Ensure the directory exists
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

plt.savefig(plot_filename)
plt.close()
