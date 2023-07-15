# Literature Review

1. “RDoC Mechanisms of Transdiagnostic Polygenic Risk for Trajectories of Depression: From Early Adolescence to Adulthood”

### Abstract -
1. Summary:<p>
        The study aims to understand the heterogeneity in the development of depression from adolescence to adulthood and the underlying genetic     and behavioral risk factors. Previous longitudinal studies identified distinct patterns of depression development, yet the etiological processes remain unknown. Genome-wide association studies (GWAS) revealed genetic variants for depression, but the significant overlap between depression and other mental disorders suggests a possible transdiagnostic and complex etiology. The study examines the relationship between a transdiagnostic polygenic score (p-factor PGS), representing genetic covariation across multiple mental disorders, and different developmental trajectories of depression. This is examined in the context of the Research Domains Criteria (RDoC) framework, focusing on negative valence, positive valence, and cognitive systems. The study uses data from the National Longitudinal Study of Adolescent to Adult Health and tests multiple mediation models to investigate the direct and indirect effects of the p-factor PGS on depression trajectories via the RDoC subconstructs.</p>
        
2. Points of Attention:
       <p> The key aspect to pay attention to is the concept of a transdiagnostic approach, both in terms of the p-factor PGS, representing genetic risk across multiple mental disorders, and the RDoC framework. Understanding the interaction between these genetic risks and various developmental and behavioral factors can provide insight into the development of depression. Another point of attention is the different trajectories of depression identified in previous studies, and their associations with the p-factor PGS and the RDoC subconstructs. The methodology of these studies, including longitudinal design and the mediation models, could be useful for exploring a potential 5th trajectory. </p>

<br>

### Method:
1. Summary:
        <p>The paper presents a methodology employed to study adolescents in the US using the Add Health database, with the goal of understanding the trajectory of depression. Data collected over four waves from various groups are analyzed for the subset of individuals where genotypic and phenotypic information are available. The variables used in the study include depression symptoms, cognitive system measures, negative and positive valence systems, and polygenic scores (PGS). The statistical analysis comprises three steps: Latent class growth models of depression were fit using Mplus, a multinomial logistic regression was performed, and multiple mediation models were executed in R.</p>

2. Points of attention:
          <p>Understanding the Sample: The demographics of the subset being studied and the details of data collection. It is important to note the characteristics of the subjects, the period covered, and the size of the sample.
Variables Involved: The chosen variables (Depression symptoms, cognitive system, positive and negative valence systems, and PGS) and their definitions/measurements, are critical in understanding the overall research study.</p>
  
3. Statistical Analysis:
          <p>Each of the three steps (Latent class growth models, multinomial logistic regression, multiple mediation models) undertaken in the analysis are crucial to understand, as they explain how the researchers interpreted the data.</p>

### Results: 
1. Summary:
          <p>The results section of the paper delineates the trajectory of depressive symptoms in adolescents, culled from the Add Health dataset, presenting four distinctive classes identified using latent class growth (LCG) models: low depression, low increasing depression, high declining depression, and early adult peak depression. Each of these trajectories is characterized by unique patterns of depressive symptom development over time. The vast majority of individuals (78.9%) exhibited consistently low depression symptoms (low depression class). Other classes showed variations such as steady increase in symptoms (low increasing depression), an initial high level followed by a significant decline (high declining depression), and an increase until age 23 followed by a decrease (early adult peak depression).

    The paper also examines the association of each Research Domain Criteria (RDoC) measure (negative emotionality, picture vocabulary, novelty seeking) and the p-factor polygenic score (PGS) on the relative risk of belonging to each depression class, using multinomial logistic regression. The results indicate distinctive associations for each class: for the low increasing and early adult peak depression classes, higher negative emotionality and novelty seeking were associated with increased relative risk, while lower picture vocabulary scores were associated only with the early adult peak and high declining depression classes. The p-factor PGS was associated only with the low increasing depression class.

      The results of the multiple mediation analysis suggest that the indirect effects of p-factor PGS on depression class through RDoC mediators were significant only for negative emotionality in the early adult peak and high declining depression classes. No significant indirect effects emerged for the low increasing depression class. </p>

2. Points of Attention:
          <p>You should pay close attention to the characteristics and composition of each identified class and the role of RDoC measures and the p-factor PGS in determining class membership. Recognizing these factors can help in understanding the factors associated with distinct depressive symptom trajectories, as well as the potential influence of genetics and different psychological constructs. This could aid in identifying the nature of the 5th trajectory you are seeking to identify.</p>

### Characterization of each class:

In this study, the researchers identified four distinct classes of depression trajectories based on latent growth curve (LGC) modeling. Here is a detailed characterization of each class:

1. **Low Depression Class:** This class represented the largest proportion of the sample (78.9%) and was characterized by consistently low levels of depression symptoms over time.

2. **Low Increasing Depression Class:** This class comprised 7.3% of the sample. Individuals in this class had similarly low levels of depression at baseline (age 13) as the low depression class, but they exhibited a steady increase in depression beginning in their early 20s.

3. **High Declining Depression Class:** This class accounted for 8.2% of the sample. Members of this class had a high initial status of depression in early adolescence but exhibited a steep decline in depression during the teenage years, reaching levels of depression by their mid-20s that were virtually indistinguishable from those in the low depression class.

4. **Early Adult Peak Depression Class:** This class represented 5.7% of the sample and was characterized by low levels of depression at baseline that steadily increased up until approximately age 23, followed by a steady decline in depression to low depression levels after age 23.

<p>The association of each Research Domain Criteria (RDoC) measure and the p-factor polygenic score (PGS) with class membership was investigated using multinomial logistic regression:</p>

1. **Low Increasing Depression Class:** Higher negative emotionality, novelty seeking, and the p-factor PGS, but not picture vocabulary, increased one’s relative risk of belonging to this class relative to the low depression class.

2. **Early Adult Peak Depression Class:** Higher negative emotionality and novelty seeking, lower picture vocabulary scores, but not the p-factor PGS, were associated with membership in this class relative to the low depression class.

3. **High Declining Depression Class:** Only higher negative emotionality and lower picture vocabulary were associated with membership in this class relative to the low depression class. Neither novelty seeking nor the p-factor PGS were associated with high declining depression class membership.

<p>Finally, in the multiple mediation analysis, indirect effects of the p-factor PGS on each depression class through the RDoC constructs were evaluated. For the low increasing depression class, higher p-factor PGS increased the risk of class membership by 5.3% (total effect), which reduced slightly to 4.7% after accounting for RDoC mediators (direct effect), but no significant total indirect effect of p-factor PGS emerged via RDoC mediators. However, there were specific indirect effects of the p-factor PGS on early adult peak and high declining depression classes via negative emotionality.</p>

### Discussion: 
1. Summary:
          <p> In the discussion section, the authors explicate the relevance of their results, comparing them to previous studies, and highlighting some unexpected outcomes. They identified four distinctive trajectories of depression from adolescence to adulthood, of which the low-increasing trajectory was particularly notable as it was uniquely associated with the p-factor PGS. This contradicts the earlier research that suggested a stronger link between early-onset depression and higher genetic liability. The authors hypothesize that this low-increasing trajectory might be linked to greater overall genetic liability that interacts with other psychopathologies that may become prominent in adulthood.

   Further, the discussion delves into the role of Research Domain Criteria (RDoC) measures in these trajectories. It notes that these measures are differentially associated with the depression trajectories, reflecting unique cognitive, behavioral, and emotional processes. Specifically, the indirect effect of the p-factor PGS on the "early adult peak" and "high declining" depression was partially mediated by negative emotionality, but not by picture vocabulary or novelty seeking.

   The authors also mention certain limitations of the study, such as the reliance on self-reporting for depression and RDoC measures and the temporal issue with novelty seeking being assessed at Wave III only. They also acknowledge the Eurocentric bias in discovery GWASs that affects the polygenic prediction signal for non-Eurocentric target populations. </p>

2. Points of Attention:

   <p> An important focus should be on the identification of the four trajectories and the unique link of the p-factor PGS with the low-increasing depression trajectory. Understanding the differential associations of RDoC measures with these trajectories and their role in mediating the impact of p-factor PGS is crucial. The limitations mentioned by the authors should also be taken into account, as they can affect the interpretation of the findings and their generalizability. </p>

---

### Project Outline:

**Analysis 1: Single-Study Analysis**
In this phase, we will work with either the ABCD or the Add Health dataset individually. The goal here is to ascertain the impact of different sets of features on the predictive performance of a model trained on that dataset. We will do this by training a model on the full set of features and then removing certain sets of features, retraining the model each time, and noting the impact on the model's predictive accuracy.

**Analysis 2: Combined Study Analysis**
In this phase, the focus is on creating a combined model using both the ABCD and Add Health datasets. This analysis involves three steps:

<p>
        a. Step 1: Matrix Completion/Collaborative Filtering**
        In this step, we will apply matrix completion techniques or collaborative filtering to impute missing features across both                   datasets. This is necessary because the two datasets may not have the same set of features, and you want to create a unified dataset         that includes all features from both.
</p>
<p>
        b. Step 2: Model Training and Comparison
        Once the datasets are unified and missing values are imputed, we will train a model using this combined dataset and compare its              predictive accuracy to the models trained on the individual datasets (from Analysis 1). The expectation is that the model trained on         the combined dataset will have superior predictive accuracy because it has access to more data.
</p>  
<p>
        c. Step 3: Feature Removal from Combined Dataset
        Similar to what was done in Analysis 1, we will now remove sets of features from the combined dataset and assess the impact on the           predictive performance of the model trained on this dataset. This will help determine which sets of features are most critical for           accurate predictions in the combined dataset setting.
</p>

---

### Experimenting with the data - 

### PHASE 1 -
The experimentation process started with an objective of understanding the influence of genetic and environmental factors on the antisocial and substance use trajectory of individuals. We used a dataset from a study that was investigating Gene x Environment interactions (GxE) on externalizing trajectories, and our primary aim was to find meaningful interactions between these variables to better understand their collective impact on antisocial behavior and substance use behavior.

In order to analyze the data and get useful insights, we adopted the following steps:

Absolutely, let's break down the process in greater detail:

1. **Exploratory Data Analysis (EDA)**: We started off with a high-level analysis of the dataset structure, content, and the variable distributions. The data contained information related to the individual's family background, gender, age, Polygenic Score (a measure of genetic influence), and two key outcomes: antisocial behavior trajectory and substance use trajectory. 

    Identifying outliers, missing values, distributions of the variables, and their interrelationships were the primary tasks in this stage. This enabled us to get an initial understanding of the dataset, the variables, their types, and their possible relationships with the outcomes of interest.

2. **Data Cleaning**: The EDA stage revealed that the dataset contained missing values. To handle these, we used the K-Nearest Neighbors (KNN) imputation method, where the missing value of an attribute is determined by the values of that attribute in 'k' nearest neighbors. The 'k' is usually a small positive integer; in our case, we chose k=4.

3. **Data Preprocessing**: Following the cleaning stage, we carried out preprocessing tasks to make the data suitable for modeling. This included scaling certain numerical features using the StandardScaler, which standardizes features by removing the mean and scaling to unit variance. The features scaled included 'PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect', 'NeighborConnect', 'ParentalWarmth'. 

    For the 'Sex' variable, which was categorical (Male or Female), we converted it into binary values (1 for Male, 0 for Female). This conversion is necessary because machine learning models require numerical input.

4. **Initial Model - Multinomial Logistic Regression (mnlogit)**: The initial modeling started with Multinomial Logistic Regression because our target variable, 'AntisocialTrajectory', was multiclass, and mnlogit is often a good starting point for such problems. However, we faced a crucial challenge: mnlogit does not natively support interaction terms, and the primary aim of this study was to investigate the effect of these interaction terms on the outcome.

5. **Transition to Tree-based Models**: To overcome the limitation of mnlogit, we transitioned to tree-based machine learning models. Tree-based models are powerful because they can naturally capture interaction effects between variables without the need for explicit specification, and they are robust to outliers and can handle non-linear data, which is a common characteristic of real-world datasets.

    We chose a variety of tree-based models to ensure robustness and generalizability of our results, including RandomForest, GradientBoosting, ExtraTrees, XGBoost, and LightGBM. Each of these models has its strengths, so this ensemble approach allowed us to capture different aspects of the data.

6. **Handling Imbalanced Data**: After transitioning to tree-based models, we identified another issue - our dataset was imbalanced with respect to the target variable, 'AntisocialTrajectory' and 'SubstanceUseTrajectory'. This can lead to biased results, as models tend to favor the majority class. We used the ADASYN (Adaptive Synthetic) technique to rectify this issue. ADASYN generates synthetic examples in the feature space of the minority class to increase its population.

7. **Creating Interaction Terms**: Given the goal of the study, we needed to generate interaction terms between the variables. An interaction term is a variable that represents a combination of two or more other variables. This could provide us valuable information about how different variables together influence the outcome.

    We generated all possible pairs of interaction terms among the variables in the dataset. These were then added one by one to the model to evaluate their impact.

8. **Evaluating Model Performance**: For each combination of model and interaction term, we evaluated the model's performance using 5-fold cross-validation, a robust technique to assess model performance and stability across different subsets of the data.

    We used several metrics for evaluation: Precision, Recall, F1 Score, ROC AUC Score, Accuracy, Log Loss, and a custom score. This comprehensive list of metrics ensured we could assess the model from different perspectives, understanding its ability to correctly classify each class, its trade-off between precision and recall, and its overall accuracy.

9. **Facing and Overcoming Errors**: During this extensive evaluation, we encountered several errors due to certain interaction terms causing issues in model training. To handle this, we added a 'try-except' block in our code. If an error occurred, it was logged into the output file, and the loop moved on to the next interaction term. This ensured that the entire process did not stop due to an error with a particular interaction term.

10. **Custom Scoring Metric**: In addition to the standard scoring metrics, we also created a custom scoring metric. This metric was a weighted sum of Precision, Recall, F1 Score, and ROC AUC Score, allowing us to consider all important aspects of the model's performance in a single score. Weights were set to 0.3 for Precision and F1, and 0.2 for Recall and ROC AUC, reflecting the relative importance we placed on these metrics.

Throughout this process, our main goal was to understand the complex interactions of genetic and environmental factors on antisocial behavior. By moving from logistic regression to tree-based models, we managed to overcome the limitations of traditional regression models. Implementing multiple evaluation metrics gave us a more complete view of the performance of different interaction terms and models, and the custom score provided a single comparable number that considered all important performance aspects.

As of now, the process of model evaluation with interaction terms is ongoing, and this comprehensive approach will likely provide robust and valuable insights into the role and interplay of genetics and environment in antisocial behavior.

---

### PHASE 2 - 

1. **Models and Interaction Terms**: In Phase 1, we only considered the model's performance with all interaction terms added at once. We found that data imbalance caused issues with certain models, such as Naive Bayes. To improve this, in Phase 2, we decided to focus on a selection of models that are known to handle imbalanced data well. The chosen models were RandomForest, GradientBoosting, ExtraTrees, XGBoost, and LightGBM. The model list was flexible and could be changed to suit different needs.

   Another major change was the approach to interaction terms. Instead of adding all interaction terms at once, we generated pairs of interaction terms using itertools.combinations. We then tested each interaction term individually with each model. This allowed us to see which specific interaction terms improved model performance. 

2. **Custom Scoring**: The initial Phase 1 code used common metrics (like precision, recall, etc.) to evaluate model performance. In Phase 2, we introduced a custom scoring system to better encapsulate the performance based on specific needs. The `custom_score` function was introduced, which combined the weighted sum of precision, recall, f1-score, and ROC_AUC. This method allowed more control over the scoring system to reflect what mattered most in our specific context.

3. **Model Evaluation**: Model evaluation underwent several changes from Phase 1 to Phase 2. Initially, we used 'cross_val_score' to evaluate models, but this approach was not compatible with our custom scoring function which required additional parameters (predicted probabilities). To resolve this, we used the 'make_scorer' function from sklearn to create a custom scorer compatible with 'cross_val_score'. 

   Later, we shifted to a train-test split approach to facilitate more control over the evaluation process, including the calculation of metrics post-prediction. This change also made it easier to use predicted probabilities in the custom scoring function, which was not possible with the 'cross_val_score' method.

4. **Error Handling**: During the testing of interaction terms, we encountered errors where certain interaction terms caused models to fail during training. This was a significant shift from Phase 1, where we didn't consider individual interaction terms' potential to cause errors. 

   In Phase 2, we introduced a try-except block around the model fitting and evaluation loop. This meant that if an error occurred with one specific interaction term or model, the code would continue to the next term or model rather than stop execution. This allowed us to robustly test a large number of interaction terms without manual intervention.

5. **Performance Logging and Comparative Analysis**: Phase 2 introduced a system for logging the performance of each model and interaction term combination. This logging was achieved by writing the results to a CSV file. This file served as a comprehensive record of the evaluation process and enabled further analysis outside of the Python environment.

   The final part of Phase 2 was to compare the models and interaction terms based on the metrics logged in the CSV file. A comparative analysis was performed, which included normalization and ranking of each metric, and then calculating an 'Average Rank' across all metrics for each model and interaction term combination. The output provided valuable insights about which models and interaction terms performed the best overall.

The journey from Phase 1 to Phase 2 was characterized by iterative refinement and expansion of the code, with a significant focus on flexibility and resilience. The changes allowed the testing of a larger set of models and interaction terms, offered a more comprehensive evaluation system, and enabled us to identify the most promising interaction terms and models for the dataset at hand.
