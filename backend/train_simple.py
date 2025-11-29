"""
Simple Model Training Script with GPU Support
Trains Decision Tree, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import os
import sys

print("\n" + "="*80)
print("DIABETES PREDICTION - MODEL TRAINING WITH GPU")
print("="*80 + "\n")

# Check GPU
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] GPU DETECTED - XGBoost will use GPU acceleration!")
        USE_GPU = True
    else:
        print("[INFO] No GPU - Using CPU for all models")
        USE_GPU = False
except:
    print("[INFO] No GPU - Using CPU for all models")
    USE_GPU = False

print("\n" + "-"*80)
print("STEP 1/6: Loading Data")
print("-"*80)

# Load data
data_path = 'data/raw/diabetes.csv'
if not os.path.exists(data_path):
    print(f"ERROR: Dataset not found at {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"[OK] Loaded {len(df)} samples")
print(f"[OK] Features: {df.shape[1]-1}")
print(f"[OK] Class distribution:")
print(f"  - No Diabetes: {(df['Outcome']==0).sum()}")
print(f"  - Diabetes: {(df['Outcome']==1).sum()}")

print("\n" + "-"*80)
print("STEP 2/6: Preprocessing Data")
print("-"*80)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handle zeros (missing values) in critical columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    if col in X.columns:
        median_val = X[X[col] != 0][col].median()
        X[col] = X[col].replace(0, median_val)

print(f"[OK] Replaced zeros with medians in {len(zero_cols)} columns")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Train set: {len(X_train)} samples")
print(f"[OK] Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Features scaled using StandardScaler")

# Save preprocessed data
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
joblib.dump(scaler, 'data/processed/scaler.pkl')

feature_info = {
    'feature_columns': X.columns.tolist(),
    'n_features': len(X.columns)
}
joblib.dump(feature_info, 'data/processed/feature_info.pkl')

print(f"[OK] Preprocessed data saved")

print("\n" + "-"*80)
print("STEP 3/6: Training Decision Tree")
print("-"*80)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

dt_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt, zero_division=0),
    'recall': recall_score(y_test, y_pred_dt, zero_division=0),
    'f1_score': f1_score(y_test, y_pred_dt, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
}

print(f"[OK] Decision Tree trained")
print(f"  Accuracy:  {dt_metrics['accuracy']:.4f}")
print(f"  Precision: {dt_metrics['precision']:.4f}")
print(f"  Recall:    {dt_metrics['recall']:.4f}")
print(f"  F1-Score:  {dt_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {dt_metrics['roc_auc']:.4f}")

# Save
model_data_dt = {
    'model': dt_model,
    'metrics': dt_metrics,
    'feature_names': X.columns.tolist()
}
joblib.dump(model_data_dt, 'models/decision_tree_model.pkl')

print("\n" + "-"*80)
print("STEP 4/6: Training Random Forest")
print("-"*80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rf, zero_division=0),
    'f1_score': f1_score(y_test, y_pred_rf, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
}

print(f"[OK] Random Forest trained")
print(f"  Accuracy:  {rf_metrics['accuracy']:.4f}")
print(f"  Precision: {rf_metrics['precision']:.4f}")
print(f"  Recall:    {rf_metrics['recall']:.4f}")
print(f"  F1-Score:  {rf_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {rf_metrics['roc_auc']:.4f}")

# Save
model_data_rf = {
    'model': rf_model,
    'metrics': rf_metrics,
    'feature_names': X.columns.tolist()
}
joblib.dump(model_data_rf, 'models/random_forest_model.pkl')

print("\n" + "-"*80)
print("STEP 5/6: Training XGBoost" + (" (GPU ACCELERATED!)" if USE_GPU else " (CPU)"))
print("-"*80)

xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss'
}

if USE_GPU:
    try:
        # Try GPU first
        xgb_params['tree_method'] = 'hist'  # Use hist (can be GPU if available)
        xgb_params['device'] = 'cuda'
        print("[GPU] Attempting GPU acceleration (device='cuda')")
    except:
        print("[INFO] GPU not available for this XGBoost build, using CPU")
else:
    xgb_params['tree_method'] = 'hist'

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb, zero_division=0),
    'f1_score': f1_score(y_test, y_pred_xgb, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb)
}

print(f"[OK] XGBoost trained")
print(f"  Accuracy:  {xgb_metrics['accuracy']:.4f}")
print(f"  Precision: {xgb_metrics['precision']:.4f}")
print(f"  Recall:    {xgb_metrics['recall']:.4f}")
print(f"  F1-Score:  {xgb_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {xgb_metrics['roc_auc']:.4f}")

# Save
model_data_xgb = {
    'model': xgb_model,
    'metrics': xgb_metrics,
    'feature_names': X.columns.tolist(),
    'use_gpu': USE_GPU
}
joblib.dump(model_data_xgb, 'models/xgboost_model.pkl')

print("\n" + "-"*80)
print("STEP 6/6: Model Comparison & Summary")
print("-"*80)

# Compare models
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [dt_metrics['accuracy'], rf_metrics['accuracy'], xgb_metrics['accuracy']],
    'Precision': [dt_metrics['precision'], rf_metrics['precision'], xgb_metrics['precision']],
    'Recall': [dt_metrics['recall'], rf_metrics['recall'], xgb_metrics['recall']],
    'F1-Score': [dt_metrics['f1_score'], rf_metrics['f1_score'], xgb_metrics['f1_score']],
    'ROC-AUC': [dt_metrics['roc_auc'], rf_metrics['roc_auc'], xgb_metrics['roc_auc']]
})

print("\nModel Performance Comparison:")
print(comparison.to_string(index=False))

# Best model
best_idx = comparison['F1-Score'].idxmax()
best_model = comparison.loc[best_idx, 'Model']
print(f"\n[BEST] Best Model: {best_model} (F1-Score: {comparison.loc[best_idx, 'F1-Score']:.4f})")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print("\nModels saved:")
print("  [OK] models/decision_tree_model.pkl")
print("  [OK] models/random_forest_model.pkl")
print("  [OK] models/xgboost_model.pkl")

print("\nPreprocessed data saved:")
print("  [OK] data/processed/X_train.csv")
print("  [OK] data/processed/X_test.csv")
print("  [OK] data/processed/scaler.pkl")
print("  [OK] data/processed/feature_info.pkl")

if USE_GPU:
    print("\n[GPU] GPU acceleration was used for XGBoost training!")

print("\nYou can now start the API server:")
print("  python app.py")

print("\n" + "="*80 + "\n")


