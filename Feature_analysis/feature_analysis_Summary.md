1. Anti-Learning - 
    1. The mnlogit was indeed anti-learning. The very based mnlogit was
giving me 6.2% accuracy and the baseline was 80%
    2. I tuned it a little, ran a grid search, rectified the imbalance
of data, even after that my accuracy bumped up to 47% which was still
much less than baseline. 


2. Feature importance - 
    1. By default it is based on mean decrees in impurity which is Gini
importance. 
    Let's analyze the results you posted. 

    2. For **SubstanceUseTrajectory**:
    
    **Mean Decrease Impurity (MDI)** based feature importance:
    
    - DelinquentPeer, SchoolConnect, and PolygenicScoreEXT have relatively higher importance compared to others. This suggests that the interaction of the individual with delinquent peers, their connection to school, and their polygenic score for externalizing traits are important factors in determining Substance Use Trajectory.
    - NeighbourConnect and ParentalWarmth are in the mid-range, suggesting they have a moderate impact.
    - is_male has the least importance, suggesting that gender is not very significant in determining Substance Use Trajectory based on this particular dataset and the model used.
    
    **Permutation-based** feature importance:
    
    - DelinquentPeer has very high importance. This reinforces the notion that the company one keeps, especially if they are involved in delinquent behavior, is very significant in Substance Use Trajectory.
    - Age also has some importance.
    - Other features like NeighbourConnect and SchoolConnect have very small importance.
    - PolygenicScoreEXT has negative importance, which is interesting. It suggests that shuffling the values in this feature actually improved the performance, indicating that maybe this feature is not predictive for this model.
    
    3. For **AntisocialTrajectory**:
    
    **Mean Decrease Impurity (MDI)** based feature importance:
    
    - Similar to the SubstanceUseTrajectory, DelinquentPeer, SchoolConnect, and ParentalWarmth have relatively higher importance.
    - is_male has the least importance, again suggesting that gender might not be very significant in determining Antisocial Trajectory.
    
    **Permutation-based** feature importance:
    
    - DelinquentPeer again has very high importance, consistently with what we observed for SubstanceUseTrajectory.
    - is_male has some importance in this case, suggesting that gender has a role in determining Antisocial Trajectory, though not very high.
    - Other features have very small or negative importance.
    
    #### Key Takeaways:
    
    1. **DelinquentPeer** is consistently the most important feature in both the SubstanceUseTrajectory and AntisocialTrajectory. This could suggest that individuals who are in the company of delinquent peers are more likely to have a higher propensity for substance use and antisocial behavior.
    
    2. **SchoolConnect** and **ParentalWarmth** seem to have a moderate impact, particularly in MDI. This may suggest that a strong connection to school and having parental support may play protective roles against substance use and antisocial behavior.
    
    3. **is_male** does not appear to be a strong predictor in these models. This suggests that gender may not be the most critical determinant in these trajectories.
    
    4. The **negative values in Permutation-based** importance suggest that some features might not be contributing positively to the model's performance. These features might not be good predictors for the target variables.
    
    5. The discrepancy between **MDI and Permutation-based** importances indicates that relying solely on one method could be misleading. Permutation-based feature importance is often considered to be more reliable as it directly measures the impact on performance. However, it's also good to analyze both to get an understanding of the structure and the actual performance impact.
    
    6. It is also important to note that feature importance doesn't imply causality. Just because a feature is important in predicting the target variable, it doesn't mean it is causing the outcome. There could be underlying factors that are not present in the dataset.
