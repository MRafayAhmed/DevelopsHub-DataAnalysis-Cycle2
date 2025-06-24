import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Preprocessing
def load_and_preprocess_data():
    """Load and preprocess the IBM HR Analytics dataset"""
    df = pd.read_csv("C:/Users/Administrator/.cache/kagglehub/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/versions/1/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Drop irrelevant columns
    df = df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'])
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# 2. Exploratory Data Analysis
def perform_eda(df):
    """Perform EDA to identify key factors influencing attrition"""
    
    # Ensure 'Attrition' is numeric for correlation
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    plt.figure(figsize=(12, 6))
    
    # Attrition distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='Attrition', data=df)
    plt.title('Attrition Distribution')
    
    # Correlation heatmap for numerical features
    plt.subplot(1, 2, 2)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(df[numerical_cols].corr(numeric_only=True), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png')
    plt.close()
    
    # Feature importance based on correlation with Attrition
    correlations = df.corr(numeric_only=True)['Attrition'].sort_values(ascending=False)
    print("Top features correlated with Attrition:\n", correlations.head(10))

# 3. Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and Logistic Regression models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'roc_auc': roc_auc_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        print(f"\n{name} Results:\n", results[name]['classification_report'])
    
    return results

# 4. SHAP Explanations
# 4. SHAP Explanations
def explain_model(model, X_train, X_test, feature_names):
    """Use SHAP to explain model predictions"""
    explainer = shap.TreeExplainer(model) if isinstance(model, RandomForestClassifier) else shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary vs multi-class properly
    if isinstance(shap_values, list):
        # For classification, take the SHAP values for class 1 (Attrition = Yes)
        shap_summary = shap_values[1]
    else:
        shap_summary = shap_values

    # Convert X_test to DataFrame (with feature names) if needed
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Plot SHAP summary
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_summary, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance for Attrition Prediction')
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    return shap_values


# 5. Main execution
if __name__ == '__main__':
    # Load and preprocess
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Perform EDA (load again separately to preserve original labels for visuals)
    df = pd.read_csv("C:/Users/Administrator/.cache/kagglehub/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/versions/1/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    perform_eda(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Explain model (using Random Forest)
    explain_model(results['Random Forest']['model'], X_train, X_test, feature_names)
    
    # Save actionable insights
    insights = """
    Actionable Retention Strategies:
    1. Enhance work-life balance: Reduce overtime work to decrease burnout.
    2. Career development: Offer clear promotion paths to increase job satisfaction.
    3. Improve compensation: Adjust salaries for employees with longer tenures.
    4. Engagement programs: Foster a positive work environment to boost retention.
    """
    with open('retention_strategies.txt', 'w') as f:
        f.write(insights)
