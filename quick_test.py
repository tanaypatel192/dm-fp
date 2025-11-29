#!/usr/bin/env python3
"""
Quick Test Script for Diabetes Prediction System
Runs a series of basic tests to verify the system is working correctly
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} | {test_name}")
    if message:
        print(f"       {message}")


def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_loaded = data.get('models_loaded', 0)
            print_test("Backend Health Check", True, 
                      f"Status: {data.get('status')}, Models: {models_loaded}")
            return True, data
        else:
            print_test("Backend Health Check", False, 
                      f"HTTP {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError:
        print_test("Backend Health Check", False, 
                  "Cannot connect - Is the backend running?")
        return False, None
    except Exception as e:
        print_test("Backend Health Check", False, str(e))
        return False, None


def test_frontend():
    """Test frontend accessibility"""
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print_test("Frontend Accessibility", True, 
                      f"Available at {FRONTEND_URL}")
            return True
        else:
            print_test("Frontend Accessibility", False, 
                      f"HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_test("Frontend Accessibility", False, 
                  "Cannot connect - Is the frontend running?")
        return False
    except Exception as e:
        print_test("Frontend Accessibility", False, str(e))
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    patient_data = {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/predict",
            params={"model_name": "xgboost"},
            json=patient_data,
            timeout=10
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print_test("Single Prediction (XGBoost)", True, 
                      f"Prediction: {data.get('prediction_label')}, "
                      f"Probability: {data.get('probability'):.2%}, "
                      f"Time: {elapsed:.0f}ms")
            return True, data
        else:
            print_test("Single Prediction (XGBoost)", False, 
                      f"HTTP {response.status_code}")
            return False, None
    except Exception as e:
        print_test("Single Prediction (XGBoost)", False, str(e))
        return False, None


def test_all_models():
    """Test predictions from all models"""
    patient_data = {
        "Pregnancies": 2,
        "Glucose": 100.0,
        "BloodPressure": 70.0,
        "SkinThickness": 25.0,
        "Insulin": 80.0,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.3,
        "Age": 35
    }
    
    models = ["decision_tree", "random_forest", "xgboost"]
    results = []
    
    for model_name in models:
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict",
                params={"model_name": model_name},
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append((model_name, True, data))
                print_test(f"Prediction - {model_name.replace('_', ' ').title()}", 
                          True, 
                          f"{data.get('prediction_label')} (prob: {data.get('probability'):.2%})")
            else:
                results.append((model_name, False, None))
                print_test(f"Prediction - {model_name.replace('_', ' ').title()}", 
                          False, 
                          f"HTTP {response.status_code}")
        except Exception as e:
            results.append((model_name, False, None))
            print_test(f"Prediction - {model_name.replace('_', ' ').title()}", 
                      False, str(e))
    
    return all(passed for _, passed, _ in results)


def test_batch_prediction():
    """Test batch prediction endpoint"""
    batch_data = {
        "patients": [
            {
                "Pregnancies": 6,
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "Insulin": 0.0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            },
            {
                "Pregnancies": 1,
                "Glucose": 85.0,
                "BloodPressure": 66.0,
                "SkinThickness": 29.0,
                "Insulin": 0.0,
                "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.351,
                "Age": 31
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict-batch",
            params={"model_name": "xgboost"},
            json=batch_data,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print_test("Batch Prediction", True, 
                      f"Processed: {data.get('total_processed')} patients, "
                      f"Time: {data.get('processing_time_ms'):.0f}ms")
            return True
        else:
            print_test("Batch Prediction", False, 
                      f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test("Batch Prediction", False, str(e))
        return False


def test_model_comparison():
    """Test model comparison endpoint"""
    patient_data = {
        "Pregnancies": 5,
        "Glucose": 120.0,
        "BloodPressure": 75.0,
        "SkinThickness": 30.0,
        "Insulin": 100.0,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/compare-models",
            json=patient_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_test("Model Comparison", True, 
                      f"Consensus: {data.get('consensus_label')}, "
                      f"Agreement: {data.get('agreement_percentage'):.0f}%")
            return True
        else:
            print_test("Model Comparison", False, 
                      f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test("Model Comparison", False, str(e))
        return False


def test_feature_importance():
    """Test feature importance endpoint"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/model/xgboost/feature-importance",
            params={"top_n": 5},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            top_feature = data[0]['feature'] if data else "N/A"
            print_test("Feature Importance", True, 
                      f"Top feature: {top_feature}")
            return True
        else:
            print_test("Feature Importance", False, 
                      f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test("Feature Importance", False, str(e))
        return False


def test_data_stats():
    """Test data statistics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/data-stats", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            total_samples = data.get('total_samples', 0)
            print_test("Dataset Statistics", True, 
                      f"Total samples: {total_samples}")
            return True
        else:
            print_test("Dataset Statistics", False, 
                      f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test("Dataset Statistics", False, str(e))
        return False


def test_comprehensive_prediction():
    """Test comprehensive prediction with explanations"""
    patient_data = {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/predict-explain",
            json=patient_data,
            timeout=15
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            shap_available = data.get('shap_available', False)
            risk_factors_count = len(data.get('risk_factors', []))
            recommendations_count = len(data.get('recommendations', []))
            
            print_test("Comprehensive Prediction", True, 
                      f"Risk: {data.get('risk_level')}, "
                      f"SHAP: {'✓' if shap_available else '✗'}, "
                      f"Risk Factors: {risk_factors_count}, "
                      f"Recommendations: {recommendations_count}, "
                      f"Time: {elapsed:.0f}ms")
            return True
        else:
            print_test("Comprehensive Prediction", False, 
                      f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_test("Comprehensive Prediction", False, str(e))
        return False


def test_error_handling():
    """Test error handling with invalid input"""
    invalid_data = {
        "Pregnancies": 6,
        "Glucose": 500.0,  # Invalid - out of range
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            params={"model_name": "xgboost"},
            json=invalid_data,
            timeout=10
        )
        
        # Should return 422 for validation error
        if response.status_code == 422:
            print_test("Error Handling (Invalid Input)", True, 
                      "Correctly rejected invalid input")
            return True
        else:
            print_test("Error Handling (Invalid Input)", False, 
                      f"Expected 422, got {response.status_code}")
            return False
    except Exception as e:
        print_test("Error Handling (Invalid Input)", False, str(e))
        return False


def print_summary(results):
    """Print test summary"""
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print_header("TEST SUMMARY")
    
    print(f"Total Tests:  {total}")
    print(f"{Colors.GREEN}Passed:       {passed}{Colors.END}")
    if failed > 0:
        print(f"{Colors.RED}Failed:       {failed}{Colors.END}")
    
    percentage = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {percentage:.1f}%")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some tests failed. Check the details above.{Colors.END}")


def print_instructions():
    """Print usage instructions"""
    print(f"\n{Colors.BOLD}Quick Test Instructions:{Colors.END}")
    print(f"\n{Colors.YELLOW}Before running this script, make sure:{Colors.END}")
    print("1. Backend is running:  python backend/app.py")
    print("2. Frontend is running: npm run dev (in frontend directory)")
    print(f"\n{Colors.CYAN}Or run them in separate terminals:{Colors.END}")
    print(f"{Colors.BOLD}Terminal 1:{Colors.END} cd backend && python app.py")
    print(f"{Colors.BOLD}Terminal 2:{Colors.END} cd frontend && npm run dev")
    print(f"{Colors.BOLD}Terminal 3:{Colors.END} python quick_test.py")


def main():
    """Run all tests"""
    print_header("DIABETES PREDICTION SYSTEM - QUICK TEST")
    
    print(f"{Colors.BOLD}Testing at:{Colors.END}")
    print(f"  Backend:  {BASE_URL}")
    print(f"  Frontend: {FRONTEND_URL}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Backend Health
    print_header("BACKEND TESTS")
    health_passed, health_data = test_backend_health()
    results['Backend Health'] = health_passed
    
    if not health_passed:
        print(f"\n{Colors.RED}{Colors.BOLD}Backend is not running!{Colors.END}")
        print_instructions()
        sys.exit(1)
    
    # Test 2: Frontend
    print_header("FRONTEND TESTS")
    results['Frontend'] = test_frontend()
    
    # Test 3-9: API Tests
    print_header("API FUNCTIONALITY TESTS")
    
    single_pred_passed, _ = test_single_prediction()
    results['Single Prediction'] = single_pred_passed
    
    results['All Models'] = test_all_models()
    results['Batch Prediction'] = test_batch_prediction()
    results['Model Comparison'] = test_model_comparison()
    results['Comprehensive Prediction'] = test_comprehensive_prediction()
    results['Feature Importance'] = test_feature_importance()
    results['Data Statistics'] = test_data_stats()
    results['Error Handling'] = test_error_handling()
    
    # Print Summary
    print_summary(results)
    
    # Print URLs
    print(f"\n{Colors.BOLD}Application URLs:{Colors.END}")
    print(f"  Frontend:        {Colors.CYAN}{FRONTEND_URL}{Colors.END}")
    print(f"  API Docs:        {Colors.CYAN}{BASE_URL}/docs{Colors.END}")
    print(f"  Health Check:    {Colors.CYAN}{BASE_URL}/health{Colors.END}")
    
    # Exit code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        sys.exit(1)




