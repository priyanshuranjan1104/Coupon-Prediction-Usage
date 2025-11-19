import warnings
warnings.filterwarnings("ignore")  # optional: suppress warnings (e.g. convergence)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# --- 1. Data Acquisition and Understanding ---
file_path = 'Dataset.csv'
df = pd.read_csv(file_path)

print("--- Initial Data Overview ---")
print("First 5 rows:")
print(df.head())
print("\nData Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe(include='all'))
print("\nUnique values in categorical columns:")
for col in ['gender', 'coupon_type', 'promo_day', 'time_of_day', 'mobile_user']:
    if col in df.columns:
        print(f"- {col}: {df[col].unique()}")
    else:
        print(f"- {col}: (column not present)")

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Define features (make sure columns exist)
categorical_features = [c for c in ['gender', 'coupon_type', 'promo_day', 'time_of_day', 'mobile_user'] if c in df.columns]
numerical_features = [c for c in ['age', 'annual_income', 'browsing_time_minutes', 'pages_viewed', 'past_purchases'] if c in df.columns]
target = 'coupon_used'
if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found in dataset")

# Ensure target is 0/1 numeric
if df[target].dtype == object or not np.issubdtype(df[target].dtype, np.number):
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])
    print(f"Label-encoded target. Classes: {list(le_target.classes_)}")

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'  # drop any other columns not specified
)

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# Distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df)
plt.title('Distribution of Coupon Usage')
plt.show()
print(f"Coupon Usage Distribution:\n{df[target].value_counts(normalize=True)}")

# Distributions of numerical features
if numerical_features:
    df[numerical_features].hist(bins=15, figsize=(15, 10))
    plt.suptitle('Histograms of Numerical Features')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("No numerical features found for histogram plotting.")

# Correlation matrix for numerical features + target
corr_cols = numerical_features + [target]
if len(numerical_features) >= 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features and Target')
    plt.show()

# Relationship between categorical features and coupon_used
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    # convert target mean to show usage rate per category
    usage = df.groupby(col)[target].mean().reset_index()
    sns.barplot(x=col, y=target, data=df, estimator=np.mean, ci=None)
    plt.title(f'Coupon Usage Rate by {col}')
    plt.ylabel('Coupon Usage Rate (mean)')
    plt.show()

# --- 4. Model Development and Training ---
X = df.drop(columns=[target])
y = df[target]

# simple train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
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

    # try to get probabilities for ROC AUC
    y_proba = None
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
        try:
            # some models have decision_function instead of predict_proba
            y_scores = pipeline.decision_function(X_test)
            # if binary, use scores directly for roc_auc
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)
        except Exception:
            y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
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
    print(f"ROC AUC: {roc_auc if not np.isnan(roc_auc) else 'N/A'}")
    print(f"Confusion Matrix:\n{cm}")

# --- 5. Model Evaluation (Detailed) ---
print("\n--- Model Comparison ---")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    for metric_name, value in metrics.items():
        if metric_name != 'Confusion Matrix':
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
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

# Fit a Random Forest pipeline to extract feature importances (we already trained one earlier; re-fit for clarity)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
rf_pipeline.fit(X_train, y_train)

# extract feature names from fitted preprocessor
num_names = numerical_features.copy()
cat_names = []
if categorical_features:
    ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    try:
        cat_names = list(ohe.get_feature_names_out(categorical_features))
    except Exception:
        # fallback for older sklearn versions
        cat_names = []
        # attempt to reconstruct names from categories_
        if hasattr(ohe, 'categories_'):
            for feat, cats in zip(categorical_features, ohe.categories_):
                cat_names.extend([f"{feat}_{val}" for val in cats])
all_feature_names = num_names + cat_names

# Random Forest feature importances
feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importances (top 10):")
print(importance_df.head(10))

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()

# Logistic Regression coefficients
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))])
lr_pipeline.fit(X_train, y_train)
lr_coefs = lr_pipeline.named_steps['classifier'].coef_[0]

coefficients_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': lr_coefs})
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
print("\nLogistic Regression Coefficients (top 10 positive):")
print(coefficients_df.head(10))
print("\nLogistic Regression Coefficients (top 10 negative):")
print(coefficients_df.tail(10))

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
print("Several machine learning models were trained and evaluated; ensemble models (Random Forest / Gradient Boosting) tend to perform well in such tabular tasks.")

print("\nKey Findings:")
print("- The distribution of coupon usage indicates whether there's an imbalance; consider class weighting or resampling if imbalanced.")
print("- (Based on EDA and Feature Importance/Coefficients):")
print("  - Factors like 'annual_income', 'browsing_time_minutes', and 'past_purchases' may be significant predictors (see feature importances).")
print("  - Specific levels of categorical features (coupon_type, promo_day, time_of_day) can show meaningful differences in usage rates.")

print("\nRecommendations for ShopEase:")
print("- Personalize offers using predicted propensity scores (use probabilities from the best model).")
print("- Target campaigns to customer segments with higher predicted coupon uptake.")
print("- Monitor which coupon types and timing produce the best conversion; A/B test changes.")
print("- If dataset is imbalanced, try class_weight or resampling (SMOTE/undersampling) and compare results.")
print("- Continuously retrain model with new data to keep predictions fresh.")
