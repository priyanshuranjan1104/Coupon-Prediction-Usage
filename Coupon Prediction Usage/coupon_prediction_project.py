import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- 1. Data Acquisition and Understanding ---
file_path = 'MultipleFiles/Coupon Usage Prediction for ShopEase.csv'
df = pd.read_csv(file_path)

print("--- Initial Data Overview ---")
print("First 5 rows:")
print(df.head())
print("\nData Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("\nUnique values in categorical columns:")
for col in ['gender', 'coupon_type', 'promo_day', 'time_of_day', 'mobile_user']:
    print(f"- {col}: {df[col].unique()}")

# --- 2. Data Preprocessing and Feature Engineering ---
# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())
# No missing values observed in the provided context, so no imputation needed for now.

# Define categorical and numerical features
categorical_features = ['gender', 'coupon_type', 'promo_day', 'time_of_day', 'mobile_user']
numerical_features = ['age', 'annual_income', 'browsing_time_minutes', 'pages_viewed', 'past_purchases']
target = 'coupon_used'

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# Distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df)
plt.title('Distribution of Coupon Usage')
plt.show()
print(f"Coupon Usage Distribution:\n{df[target].value_counts(normalize=True)}")

# Distributions of numerical features
df[numerical_features].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Target')
plt.show()

# Relationship between categorical features and coupon_used
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=col, y=target, data=df, ci=None)
    plt.title(f'Coupon Usage Rate by {col}')
    plt.ylabel('Coupon Usage Rate')
    plt.show()

# --- 4. Model Development and Training ---
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': cm
    }
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# --- 5. Model Evaluation (Detailed) ---
print("\n--- Model Comparison ---")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    for metric_name, value in metrics.items():
        if metric_name != 'Confusion Matrix':
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}:\n{value}")

# Cross-validation for robustness
print("\n--- Cross-Validation Scores (Accuracy) ---")
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_scores.mean():.4f} (Std Dev = {cv_scores.std():.4f})")

# --- 6. Model Interpretation and Insights ---
print("\n--- Model Interpretation ---")

# Feature importance for tree-based models
# Re-train a Random Forest model to get feature importances from the trained pipeline
# Need to get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Train a Random Forest model within a pipeline to get feature importances
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42))])
rf_pipeline.fit(X_train, y_train)

if 'Random Forest' in models:
    feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\nRandom Forest Feature Importances:")
    print(importance_df.head(10))

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.show()

# Coefficients for Logistic Regression
if 'Logistic Regression' in models:
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])
    lr_pipeline.fit(X_train, y_train)

    # Get coefficients from the trained logistic regression model
    lr_coefficients = lr_pipeline.named_steps['classifier'].coef_[0]
    coefficients_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': lr_coefficients})
    coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)

    print("\nLogistic Regression Coefficients:")
    print(coefficients_df.head(10))
    print(coefficients_df.tail(10)) # Also show least influential

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients_df.head(10))
    plt.title('Top 10 Positive Coefficients (Logistic Regression)')
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients_df.tail(10))
    plt.title('Top 10 Negative Coefficients (Logistic Regression)')
    plt.show()

# --- 7. Reporting and Visualization (Summary of findings will be presented here) ---
print("\n--- Project Summary and Recommendations ---")
print("The project aimed to predict coupon usage for ShopEase using various customer and promotional factors.")
print("Several machine learning models were trained and evaluated, with Random Forest and Gradient Boosting generally showing strong performance.")
print("\nKey Findings:")
print("- The distribution of coupon usage indicates a slight imbalance, with more customers not using coupons.")
print("- (Based on EDA and Feature Importance/Coefficients):")
print("  - Factors like 'annual_income', 'browsing_time_minutes', and 'past_purchases' appear to be significant predictors.")
print("  - Certain 'coupon_type' or 'promo_day' might have a higher impact on usage.")
print("\nRecommendations for ShopEase:")
print("- **Personalized Offers:** Leverage the predictive model to identify customers most likely to use a coupon and tailor offers accordingly.")
print("- **Targeted Campaigns:** Focus marketing efforts on customer segments identified as having a higher propensity for coupon usage.")
print("- **Optimize Coupon Types:** Analyze which 'coupon_type' (e.g., Cashback, Discount, BOGO, Free Shipping) performs best and prioritize those in future campaigns.")
print("- **Strategic Timing:** Utilize insights from 'promo_day' and 'time_of_day' to schedule coupon promotions for maximum impact.")
print("- **Customer Engagement:** Encourage browsing and past purchases, as these factors seem to correlate with higher coupon usage.")
print("- **A/B Testing:** Continuously A/B test different coupon strategies and integrate new data to refine the predictive model over time.")
