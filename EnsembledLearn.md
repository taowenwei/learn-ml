# Ensemble learning 
Ensemble learning in `scikit-learn` refers to the technique of combining multiple machine learning models to improve overall performance. The idea is that by aggregating the predictions of several models, the ensemble can often achieve better accuracy and robustness than any single model. Ensemble methods can be broadly categorized into two main types: **bagging** and **boosting**.

### Types of Ensemble Learning

1. **Bagging (Bootstrap Aggregating)**:
   - **Random Forest**: A popular bagging method where multiple decision trees are trained on different subsets of the data. The final prediction is made by averaging the predictions of all the trees (for regression) or by majority vote (for classification).
   - **Bagging Classifier/Regressor**: A more general form where any base estimator (not just decision trees) can be used.

2. **Boosting**:
   - **AdaBoost**: An adaptive boosting method where each subsequent model focuses more on the errors made by the previous models.
   - **Gradient Boosting**: Builds models sequentially, with each new model trying to correct the errors of the previous ones. Implementations include `GradientBoostingClassifier` and `GradientBoostingRegressor`.
   - **XGBoost**: An optimized and scalable implementation of gradient boosting.

3. **Stacking**:
   - Combines the predictions of multiple models (level-0 models) using another model (level-1 model), often referred to as the meta-learner.

4. **Voting**:
   - Combines the predictions of multiple models by averaging (for regression) or majority voting (for classification). `VotingClassifier` is used for classification, and `VotingRegressor` for regression.
