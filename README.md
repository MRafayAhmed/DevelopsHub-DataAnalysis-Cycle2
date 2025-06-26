# DevelopsHub-DataAnalysis-Cycle2
<Task01>
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


<Task02>
Text Summarization System Report
1. Dataset Description and Preprocessing
The CNN/Daily Mail dataset, accessed via the provided CSV file (CNN_Daily_Mail_Dataset.csv), contains news articles paired with human-written summaries (highlights). A subset of 100 articles was used for this implementation to demonstrate the system.
Preprocessing Steps:

Loaded the dataset from the specified CSV path.
Cleaned text by removing extra whitespace and special characters using regex.
Truncated articles to 1024 tokens to fit BART’s input limits.
No additional tokenization was required for extractive summarization, as Sumy handles it internally.

2. Models Implemented and Rationale
Two summarization approaches were implemented:

Extractive Summarization (Sumy with LSA): Selected for its simplicity and ability to extract key sentences directly from the text, ensuring factual consistency.
Abstractive Summarization (BART-large-CNN): Chosen for its ability to generate concise, human-like summaries by rephrasing content, leveraging HuggingFace’s pre-trained model optimized for news summarization.

Evaluation Metrics:

ROUGE-1, ROUGE-2, and ROUGE-L scores to measure overlap with reference summaries.
Manual inspection of summaries for coherence and readability.

3. Key Insights and Visualizations
EDA Insights:

Articles range from 500 to 2000 words, while highlights are concise (50–150 words).
Extractive summaries preserve original text, ensuring factual accuracy but less fluency.
Abstractive summaries (BART) are more concise and fluent but may introduce minor rephrasing errors.

Model Performance:

BART-based abstractive summaries achieved higher ROUGE scores (e.g., ROUGE-1 F1 0.45) compared to extractive summaries (0.35).
Visualizations (saved as rouge_scores_extractive.png and rouge_scores_abstractive.png) confirm BART’s superior performance across ROUGE metrics.
Sample summaries saved in sample_summaries.txt demonstrate coherence and relevance to the original article.

Actionable Insights:

Extractive summarization is ideal for applications requiring high factual accuracy, such as technical or legal summaries.
Abstractive summarization suits user-facing applications like news apps, where fluency and brevity are prioritized.

4. Challenges Faced and Solutions

Challenge: Handling long articles exceeding BART’s 1024-token limit.
Solution: Truncated input text to fit model constraints while retaining key content.


Challenge: Limited computational resources for fine-tuning BART.
Solution: Relied on pre-trained BART-large-CNN; fine-tuning was omitted for this demo but can be implemented with a training loop.


Challenge: Assessing summary coherence beyond ROUGE metrics.
Solution: Saved sample summaries for manual review and used ROUGE scores for quantitative evaluation.



Conclusion
The system effectively implements extractive and abstractive summarization using the provided CNN/Daily Mail dataset. BART provides more fluent and concise summaries, while extractive methods ensure factual accuracy. Future enhancements could involve fine-tuning BART and incorporating advanced coherence metrics like BERTScore.


<br>Task03<br>

![image](https://github.com/user-attachments/assets/a925a0fc-05fb-4fe3-aaf2-bec421fc9b26)
