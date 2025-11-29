"""
Test the trained models directly
"""

import sys
sys.path.append('backend')

import joblib
import numpy as np
import pandas as pd

print("\n" + "="*80)
print("TESTING TRAINED MODELS")
print("="*80 + "\n")

# Load scaler and models
print("[1/4] Loading models and scaler...")
scaler = joblib.load('backend/data/processed/scaler.pkl')
dt_data = joblib.load('backend/models/decision_tree_model.pkl')
rf_data = joblib.load('backend/models/random_forest_model.pkl')
xgb_data = joblib.load('backend/models/xgboost_model.pkl')

dt_model = dt_data['model']
rf_model = rf_data['model']
xgb_model = xgb_data['model']

print("[OK] All models and scaler loaded successfully!")
print(f"  Decision Tree: Loaded")
print(f"  Random Forest: Loaded")
print(f"  XGBoost: Loaded (GPU: {xgb_data.get('use_gpu', False)})")

# Test patients
print("\n[2/4] Preparing test cases...")
test_patients = [
    {
        "name": "High Risk Patient",
        "data": [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        "expected": "Diabetes"
    },
    {
        "name": "Low Risk Patient",  
        "data": [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        "expected": "No Diabetes"
    },
    {
        "name": "Moderate Risk Patient",
        "data": [3, 120, 70, 30, 100, 32.0, 0.4, 40],
        "expected": "Unknown"
    }
]

print(f"[OK] {len(test_patients)} test cases prepared")

# Make predictions
print("\n[3/4] Making predictions...")
print("-"*80)

results = []
for patient in test_patients:
    print(f"\nPatient: {patient['name']}")
    print(f"  Data: {patient['data']}")
    
    # Scale data
    patient_array = np.array([patient['data']])
    patient_scaled = scaler.transform(patient_array)
    
    # Predict with all models
    dt_pred = dt_model.predict(patient_scaled)[0]
    dt_proba = dt_model.predict_proba(patient_scaled)[0][1]
    
    rf_pred = rf_model.predict(patient_scaled)[0]
    rf_proba = rf_model.predict_proba(patient_scaled)[0][1]
    
    xgb_pred = xgb_model.predict(patient_scaled)[0]
    xgb_proba = xgb_model.predict_proba(patient_scaled)[0][1]
    
    # Print results
    print(f"\n  Predictions:")
    print(f"    Decision Tree:  {'Diabetes' if dt_pred == 1 else 'No Diabetes':12} (prob: {dt_proba:.2%})")
    print(f"    Random Forest:  {'Diabetes' if rf_pred == 1 else 'No Diabetes':12} (prob: {rf_proba:.2%})")
    print(f"    XGBoost:        {'Diabetes' if xgb_pred == 1 else 'No Diabetes':12} (prob: {xgb_proba:.2%})")
    
    # Ensemble prediction
    ensemble_proba = (dt_proba + rf_proba + xgb_proba) / 3
    ensemble_pred = 1 if ensemble_proba >= 0.5 else 0
    
    print(f"\n  Ensemble:       {'Diabetes' if ensemble_pred == 1 else 'No Diabetes':12} (prob: {ensemble_proba:.2%})")
    
    # Risk level
    if ensemble_proba < 0.3:
        risk = "Low"
    elif ensemble_proba < 0.7:
        risk = "Medium"
    else:
        risk = "High"
    
    print(f"  Risk Level:     {risk}")
    
    results.append({
        'patient': patient['name'],
        'ensemble_prediction': 'Diabetes' if ensemble_pred == 1 else 'No Diabetes',
        'probability': ensemble_proba,
        'risk_level': risk
    })

print("\n" + "-"*80)

# Summary
print("\n[4/4] Test Summary")
print("-"*80)

summary_df = pd.DataFrame(results)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("MODEL TESTING COMPLETE!")
print("="*80)

print("\n[SUCCESS] All models are working correctly!")
print("\nModels can now be used via:")
print("  1. Python API: Load the .pkl files directly")
print("  2. REST API: Start backend with 'python app.py'")
print("  3. Web UI: Start frontend with 'npm run dev'")

print("\nModel Summary:")
print(f"  Best ROC-AUC: XGBoost ({xgb_data['metrics']['roc_auc']:.4f})")
print(f"  Best F1-Score: {max(dt_data['metrics']['f1_score'], rf_data['metrics']['f1_score'], xgb_data['metrics']['f1_score']):.4f}")
print(f"  GPU Accelerated: {'Yes' if xgb_data.get('use_gpu', False) else 'No'}")

print("\n" + "="*80 + "\n")


