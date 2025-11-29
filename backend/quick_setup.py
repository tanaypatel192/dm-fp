"""
Quick setup script to download data, preprocess, and train models with GPU
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("="*80)
print(" QUICK SETUP - Diabetes Prediction System")
print("="*80)
print()

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results/decision_tree", exist_ok=True)
os.makedirs("results/random_forest", exist_ok=True)
os.makedirs("results/xgboost", exist_ok=True)

print("Step 1: Loading Pima Indians Diabetes Dataset...")
print("-" * 80)

# Try to load from online source or use sklearn's diabetes dataset
try:
    # Download Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    df = pd.read_csv(url, names=column_names)
    print(f"[OK] Downloaded Pima Indians Diabetes Dataset: {df.shape}")
    
except Exception as e:
    print(f"[WARN] Could not download from online source: {e}")
    print("Creating synthetic dataset for demonstration...")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(70, 20, n_samples).clip(0, 122),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 99),
        'Insulin': np.random.normal(80, 115, n_samples).clip(0, 846),
        'BMI': np.random.normal(32, 7, n_samples).clip(0, 67),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples).clip(0, 2.42),
        'Age': np.random.randint(21, 81, n_samples),
    }
    
    df = pd.DataFrame(data)
    # Create outcome based on risk factors
    risk_score = (
        (df['Glucose'] > 140).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) +
        (df['Age'] > 50).astype(int) +
        (df['BloodPressure'] > 80).astype(int)
    )
    df['Outcome'] = (risk_score + np.random.randint(-1, 2, n_samples) > 2).astype(int)
    
    print(f"[OK] Created synthetic dataset: {df.shape}")

# Save raw data
df.to_csv("data/raw/diabetes.csv", index=False)
print(f"[OK] Saved to data/raw/diabetes.csv")
print()

print("Step 2: Preprocessing Data...")
print("-" * 80)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X)}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Handle zero values in medical measurements
zero_replacements = {
    'Glucose': X[X['Glucose'] > 0]['Glucose'].median(),
    'BloodPressure': X[X['BloodPressure'] > 0]['BloodPressure'].median(),
    'SkinThickness': X[X['SkinThickness'] > 0]['SkinThickness'].median(),
    'Insulin': X[X['Insulin'] > 0]['Insulin'].median(),
    'BMI': X[X['BMI'] > 0]['BMI'].median()
}

for col, median_val in zero_replacements.items():
    X.loc[X[col] == 0, col] = median_val

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# Save processed data
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False, header=['Outcome'])
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False, header=['Outcome'])

# Save scaler and feature info
joblib.dump(scaler, "data/processed/scaler.pkl")
feature_info = {
    'feature_columns': list(X.columns),
    'n_features': len(X.columns)
}
joblib.dump(feature_info, "data/processed/feature_info.pkl")

print(f"[OK] Train set: {X_train_scaled.shape}")
print(f"[OK] Test set: {X_test_scaled.shape}")
print(f"[OK] Scaler saved")
print()

print("="*80)
print(" SETUP COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Train models with GPU: python train_models_gpu.py")
print("2. Start API server: python app.py")
print()

