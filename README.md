# DevelopsHub-DataAnalysis-Cycle2
<b>Task01<b>
Employee Attrition Prediction Report
1. Dataset Description and Preprocessing
The IBM HR Analytics Dataset contains 1,470 employee records with 35 features, including demographic, job, and satisfaction metrics. The target variable, Attrition, indicates whether an employee left the company (1) or stayed (0).
Preprocessing Steps:

Removed irrelevant columns: EmployeeNumber, EmployeeCount, StandardHours, Over18.
Encoded categorical variables (e.g., Department, JobRole) using LabelEncoder.
Scaled numerical features using StandardScaler.
Split data into 80% training and 20% testing sets.

2. Models Implemented and Rationale
Two classification models were trained:

Random Forest: Chosen for its ability to handle non-linear relationships and feature interactions, which are common in HR data.
Logistic Regression: Selected for its interpretability and effectiveness in binary classification tasks.

Evaluation Metrics:

ROC-AUC score to assess model performance on imbalanced data.
Precision, recall, and F1-score for detailed performance analysis.

3. Key Insights and Visualizations
EDA Insights:

Attrition is imbalanced (~16% left vs. 84% stayed).
Key factors influencing attrition: Overtime, JobSatisfaction, YearsAtCompany, and MonthlyIncome.
Visualizations (saved as eda_plots.png):
Attrition distribution showed class imbalance.
Correlation heatmap revealed relationships between numerical features.



Model Performance:

Random Forest outperformed Logistic Regression with a higher ROC-AUC score (~0.85 vs. 0.78).
SHAP analysis (saved as shap_summary.png) highlighted:
Overtime and JobSatisfaction as top contributors to attrition risk.
Employees with lower MonthlyIncome and fewer YearsAtCompany are more likely to leave.



Actionable Retention Strategies (saved as retention_strategies.txt):

Reduce overtime to improve work-life balance.
Offer clear career progression paths.
Adjust compensation for long-tenure employees.
Implement engagement programs to boost satisfaction.

4. Challenges Faced and Solutions

Challenge: Class imbalance in the dataset.
Solution: Used ROC-AUC as the primary metric and considered SMOTE (not implemented due to sufficient model performance).


Challenge: Interpreting complex model predictions.
Solution: Employed SHAP for explainable insights, focusing on key features.


Challenge: Handling categorical variables with high cardinality (e.g., JobRole).
Solution: Applied LabelEncoder and validated model performance to ensure no significant information loss.



Conclusion
The Random Forest model effectively predicts employee attrition, with SHAP providing interpretable insights into key drivers. HR can leverage the proposed retention strategies to reduce turnover and enhance employee satisfaction.
