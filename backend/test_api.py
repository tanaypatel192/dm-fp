"""
Test script for Diabetes Prediction API

This script demonstrates how to interact with the API endpoints.

Usage:
    python test_api.py

Make sure the API server is running:
    python app.py
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_response(response: requests.Response):
    """Print formatted API response."""
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_health_check():
    """Test health check endpoint."""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)


def test_single_prediction():
    """Test single prediction endpoint."""
    print_section("2. Single Prediction")

    # High-risk patient
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

    print(f"Patient Data: {json.dumps(patient_data, indent=2)}")

    for model in ["xgboost", "random_forest", "decision_tree"]:
        print(f"\nUsing model: {model}")
        response = requests.post(
            f"{BASE_URL}/api/predict",
            params={"model_name": model},
            json=patient_data
        )
        print_response(response)


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_section("3. Batch Prediction")

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
            },
            {
                "Pregnancies": 8,
                "Glucose": 183.0,
                "BloodPressure": 64.0,
                "SkinThickness": 0.0,
                "Insulin": 0.0,
                "BMI": 23.3,
                "DiabetesPedigreeFunction": 0.672,
                "Age": 32
            }
        ]
    }

    print(f"Batch size: {len(batch_data['patients'])} patients")

    response = requests.post(
        f"{BASE_URL}/api/predict-batch",
        params={"model_name": "xgboost"},
        json=batch_data
    )
    print_response(response)


def test_list_models():
    """Test list models endpoint."""
    print_section("4. List All Models")

    response = requests.get(f"{BASE_URL}/api/models")
    print_response(response)


def test_model_metrics():
    """Test model metrics endpoint."""
    print_section("5. Model Metrics")

    for model in ["xgboost", "random_forest", "decision_tree"]:
        print(f"\nModel: {model}")
        response = requests.get(f"{BASE_URL}/api/model/{model}/metrics")
        print_response(response)


def test_feature_importance():
    """Test feature importance endpoint."""
    print_section("6. Feature Importance")

    for model in ["xgboost", "random_forest", "decision_tree"]:
        print(f"\nModel: {model} (Top 5 Features)")
        response = requests.get(
            f"{BASE_URL}/api/model/{model}/feature-importance",
            params={"top_n": 5}
        )
        print_response(response)


def test_compare_models():
    """Test model comparison endpoint."""
    print_section("7. Compare All Models")

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

    print(f"Patient Data: {json.dumps(patient_data, indent=2)}")

    response = requests.post(
        f"{BASE_URL}/api/compare-models",
        json=patient_data
    )
    print_response(response)


def test_data_stats():
    """Test data statistics endpoint."""
    print_section("8. Dataset Statistics")

    response = requests.get(f"{BASE_URL}/api/data-stats")
    print_response(response)


def test_error_handling():
    """Test error handling."""
    print_section("9. Error Handling")

    # Test invalid model name
    print("\nTest 1: Invalid model name")
    response = requests.post(
        f"{BASE_URL}/api/predict",
        params={"model_name": "invalid_model"},
        json={
            "Pregnancies": 6,
            "Glucose": 148.0,
            "BloodPressure": 72.0,
            "SkinThickness": 35.0,
            "Insulin": 0.0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    )
    print_response(response)

    # Test invalid input (out of range)
    print("\nTest 2: Invalid input (Glucose out of range)")
    response = requests.post(
        f"{BASE_URL}/api/predict",
        params={"model_name": "xgboost"},
        json={
            "Pregnancies": 6,
            "Glucose": 500.0,  # Out of range
            "BloodPressure": 72.0,
            "SkinThickness": 35.0,
            "Insulin": 0.0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    )
    print_response(response)

    # Test missing field
    print("\nTest 3: Missing required field")
    response = requests.post(
        f"{BASE_URL}/api/predict",
        params={"model_name": "xgboost"},
        json={
            "Pregnancies": 6,
            "Glucose": 148.0,
            # Missing BloodPressure
            "SkinThickness": 35.0,
            "Insulin": 0.0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    )
    print_response(response)


def main():
    """Run all tests."""
    print("="*80)
    print("  DIABETES PREDICTION API - TEST SUITE")
    print("="*80)
    print(f"\nTesting API at: {BASE_URL}")
    print("\nMake sure the API server is running!")
    print("Start with: python app.py")

    try:
        # Run all tests
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_list_models()
        test_model_metrics()
        test_feature_importance()
        test_compare_models()
        test_data_stats()
        test_error_handling()

        print_section("TEST SUITE COMPLETED")
        print("All tests executed successfully!")

    except requests.exceptions.ConnectionError:
        print("\n" + "="*80)
        print("  ERROR: Cannot connect to API server")
        print("="*80)
        print("\nPlease make sure the API server is running:")
        print("  python app.py")
        print("\nOr:")
        print("  uvicorn app:app --reload")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")


if __name__ == "__main__":
    main()
