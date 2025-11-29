"""Simple API Test Script - No Unicode"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("\n" + "="*60)
print("  API TESTING - ALL ENDPOINTS")
print("="*60 + "\n")

# Test 1: Health Check
print("[Test 1/7] Health Check")
try:
    r = requests.get(f"{BASE_URL}/health")
    data = r.json()
    print(f"  [PASS] Status: {data['status']}")
    print(f"  Models: {data['models_loaded']}")
    print(f"  Available: {', '.join(data['available_models'])}\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# Test 2: Single Prediction - XGBoost
print("[Test 2/7] Single Prediction (XGBoost)")
patient = {
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
    "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627, "Age": 50
}
try:
    r = requests.post(f"{BASE_URL}/api/predict?model_name=xgboost", json=patient)
    data = r.json()
    print(f"  [PASS] Prediction: {data['prediction_label']}")
    print(f"  Probability: {data['probability']:.1%}")
    print(f"  Risk: {data['risk_level']}\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# Test 3: All Models
print("[Test 3/7] All Three Models")
for model in ['decision_tree', 'random_forest', 'xgboost']:
    try:
        r = requests.post(f"{BASE_URL}/api/predict?model_name={model}", json=patient)
        data = r.json()
        print(f"  {model:15} - {data['prediction_label']:12} ({data['probability']:.1%})")
    except Exception as e:
        print(f"  {model:15} - FAILED")
print()

# Test 4: Model Comparison
print("[Test 4/7] Model Comparison")
try:
    r = requests.post(f"{BASE_URL}/api/compare-models", json=patient)
    data = r.json()
    print(f"  [PASS] Consensus: {data['consensus_label']}")
    print(f"  Agreement: {data['agreement_percentage']:.0f}%\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# Test 5: Batch Prediction
print("[Test 5/7] Batch Prediction")
batch = {"patients": [patient, {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}]}
try:
    r = requests.post(f"{BASE_URL}/api/predict-batch?model_name=xgboost", json=batch)
    data = r.json()
    print(f"  [PASS] Processed: {data['total_processed']} patients")
    print(f"  Time: {data['processing_time_ms']:.0f}ms\n")
except Exception as e:
    print(f"  [FAIL] {e}\n")

# Test 6: Feature Importance
print("[Test 6/7] Feature Importance")
try:
    r = requests.get(f"{BASE_URL}/api/model/xgboost/feature-importance?top_n=5")
    data = r.json()
    print(f"  [PASS] Top 5 features:")
    for feat in data[:3]:
        print(f"    {feat['rank']}. {feat['feature']}: {feat['importance']:.4f}")
    print()
except Exception as e:
    print(f"  [FAIL] {e}\n")

# Test 7: List Models
print("[Test 7/7] List All Models")
try:
    r = requests.get(f"{BASE_URL}/api/models")
    data = r.json()
    print(f"  [PASS] Found {len(data)} models:")
    for m in data:
        print(f"    {m['model_name']:15} - ROC-AUC: {m['roc_auc']:.4f}")
    print()
except Exception as e:
    print(f"  [FAIL] {e}\n")

print("="*60)
print("  ALL TESTS COMPLETED!")
print("="*60)
print("\nServers are running at:")
print("  Frontend: http://localhost:3000")
print("  Backend:  http://localhost:8000/docs")
print("  Health:   http://localhost:8000/health")
print("\n")


