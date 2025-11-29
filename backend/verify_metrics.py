import joblib
import os
import sys

def verify_models():
    models_dir = 'models'
    model_files = {
        'Decision Tree': 'decision_tree_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    all_passed = True

    print("Verifying model metrics...")
    print("="*60)

    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            print(f"[FAIL] {name}: File not found at {path}")
            all_passed = False
            continue

        try:
            data = joblib.load(path)
            metrics = data.get('metrics')
            
            if metrics:
                print(f"[PASS] {name}: Metrics found")
                print(f"       Accuracy: {metrics.get('accuracy', 'N/A')}")
                print(f"       Precision: {metrics.get('precision', 'N/A')}")
                print(f"       Recall: {metrics.get('recall', 'N/A')}")
                
                # Check for non-zero values
                if metrics.get('accuracy', 0) == 0:
                     print(f"       [WARNING] Accuracy is 0!")
            else:
                print(f"[FAIL] {name}: 'metrics' key missing in model data")
                all_passed = False
                
        except Exception as e:
            print(f"[FAIL] {name}: Error loading model - {str(e)}")
            all_passed = False
            
    print("="*60)
    if all_passed:
        print("VERIFICATION SUCCESSFUL: All models have metrics.")
    else:
        print("VERIFICATION FAILED: Some models are missing metrics.")
        sys.exit(1)

if __name__ == "__main__":
    verify_models()
