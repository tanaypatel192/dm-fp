"""
FastAPI Backend for Diabetes Prediction System

This module implements a comprehensive REST API for diabetes prediction using
Decision Tree, Random Forest, and XGBoost models.

Endpoints:
- POST /api/predict: Single prediction
- POST /api/predict-batch: Batch predictions
- GET /api/models: List all models with metrics
- GET /api/model/{model_name}/metrics: Get model metrics
- GET /api/model/{model_name}/feature-importance: Get feature importance
- POST /api/compare-models: Compare all models
- GET /api/data-stats: Get dataset statistics
- GET /health: Health check

Author: Diabetes Prediction Project
Date: 2025
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import joblib
import logging
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="REST API for diabetes prediction using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessor
models_cache = {}
scaler = None
feature_names = None
model_metrics = {}


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PatientInput(BaseModel):
    """Input schema for patient data with validation."""
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Glucose level (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    Insulin: float = Field(..., ge=0, le=900, description="Insulin level (mu U/ml)")
    BMI: float = Field(..., ge=0, le=70, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "Insulin": 0.0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }

    @validator('Glucose', 'BloodPressure', 'BMI')
    def check_critical_values(cls, v, field):
        """Validate critical health measurements."""
        if v == 0:
            logger.warning(f"Zero value detected for {field.name}, will be imputed")
        return v


class BatchPatientInput(BaseModel):
    """Input schema for batch predictions."""
    patients: List[PatientInput] = Field(..., min_items=1, max_items=100)

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    prediction: int = Field(..., description="Predicted class (0: No Diabetes, 1: Diabetes)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., ge=0, le=1, description="Probability of diabetes")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str = Field(..., description="Name of model used for prediction")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "Diabetes",
                "probability": 0.78,
                "risk_level": "High",
                "confidence": 0.78,
                "model_used": "xgboost"
            }
        }


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    total_processed: int
    processing_time_ms: float


class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    is_loaded: bool
    last_updated: Optional[str]


class FeatureImportance(BaseModel):
    """Schema for feature importance."""
    feature: str
    importance: float
    rank: int


class ModelComparisonOutput(BaseModel):
    """Output schema for model comparison."""
    input_data: Dict[str, Any]
    predictions: Dict[str, PredictionOutput]
    consensus_prediction: int
    consensus_label: str
    agreement_percentage: float


class DataStats(BaseModel):
    """Schema for dataset statistics."""
    total_samples: int
    features_count: int
    class_distribution: Dict[str, int]
    feature_statistics: Dict[str, Dict[str, float]]


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    timestamp: str
    models_loaded: int
    available_models: List[str]


# ============================================================================
# Startup Event - Load Models and Preprocessor
# ============================================================================

@app.on_event("startup")
async def load_models_and_preprocessor():
    """Load all trained models and preprocessor at startup."""
    global models_cache, scaler, feature_names, model_metrics

    logger.info("="*80)
    logger.info("LOADING MODELS AND PREPROCESSOR")
    logger.info("="*80)

    try:
        # Define model paths
        models_dir = "models"
        model_files = {
            "decision_tree": "decision_tree_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "xgboost": "xgboost_model.pkl"
        }

        # Load scaler
        scaler_path = os.path.join("data", "processed", "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"✗ Scaler not found at {scaler_path}")

        # Load feature info
        feature_info_path = os.path.join("data", "processed", "feature_info.pkl")
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            feature_names = feature_info.get('feature_columns')
            logger.info(f"✓ Feature info loaded: {len(feature_names)} features")
        else:
            logger.warning(f"✗ Feature info not found at {feature_info_path}")
            # Fallback to basic features
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Load models
        for model_name, model_file in model_files.items():
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    models_cache[model_name] = model_data.get('model')

                    # Load metrics if available
                    cv_results = model_data.get('cv_results', {})
                    if cv_results:
                        model_metrics[model_name] = {
                            'cv_score': cv_results.get('mean_cv_score', 0.0),
                            'cv_std': cv_results.get('std_cv_score', 0.0)
                        }

                    logger.info(f"✓ {model_name.replace('_', ' ').title()} model loaded")
                except Exception as e:
                    logger.error(f"✗ Error loading {model_name}: {str(e)}")
            else:
                logger.warning(f"✗ {model_name.replace('_', ' ').title()} model not found at {model_path}")

        # Load evaluation metrics from files
        results_dir = "results"
        for model_name in model_files.keys():
            metrics_file = os.path.join(results_dir, model_name.replace('_', ''), 'evaluation_metrics.txt')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        # Parse metrics from file
                        content = f.read()
                        # This is a simplified parser - adjust based on actual file format
                        if model_name not in model_metrics:
                            model_metrics[model_name] = {}
                except Exception as e:
                    logger.error(f"Error reading metrics for {model_name}: {str(e)}")

        logger.info("="*80)
        logger.info(f"STARTUP COMPLETE - {len(models_cache)} models loaded")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_input(patient_data: PatientInput) -> np.ndarray:
    """
    Preprocess patient input data.

    Args:
        patient_data: PatientInput object

    Returns:
        np.ndarray: Preprocessed feature array
    """
    # Convert to DataFrame
    data_dict = patient_data.dict()
    df = pd.DataFrame([data_dict])

    # Handle zero values (missing data) by replacing with median
    # These medians should ideally come from training data
    zero_replacements = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 23.0,
        'Insulin': 30.0,
        'BMI': 32.0
    }

    for col, median_val in zero_replacements.items():
        if df[col].iloc[0] == 0:
            df[col] = median_val

    # Apply scaling if scaler is available
    if scaler is not None:
        # Ensure columns match scaler's expected features
        if feature_names and len(feature_names) > 8:
            # If we have engineered features, we need to create them
            # For now, just use the base features
            base_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            df = df[base_features]

        scaled_data = scaler.transform(df.values)
        return scaled_data
    else:
        return df.values


def calculate_risk_level(probability: float) -> str:
    """
    Calculate risk level based on probability.

    Args:
        probability: Probability of diabetes

    Returns:
        str: Risk level (Low/Medium/High)
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def get_model_or_404(model_name: str):
    """Get model from cache or raise 404."""
    if model_name not in models_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available models: {list(models_cache.keys())}"
        )
    return models_cache[model_name]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and loaded models information.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models_cache),
        available_models=list(models_cache.keys())
    )


@app.post("/api/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(
    patient: PatientInput,
    model_name: str = "xgboost"
):
    """
    Make a single prediction for a patient.

    - **patient**: Patient data with all required features
    - **model_name**: Model to use (decision_tree, random_forest, xgboost)

    Returns prediction with probability and risk level.
    """
    try:
        logger.info(f"Prediction request received for model: {model_name}")

        # Get model
        model = get_model_or_404(model_name)

        # Preprocess input
        processed_data = preprocess_input(patient)

        # Make prediction
        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0][1])

        # Calculate risk level
        risk_level = calculate_risk_level(probability)

        # Prepare response
        response = PredictionOutput(
            prediction=prediction,
            prediction_label="Diabetes" if prediction == 1 else "No Diabetes",
            probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(max(model.predict_proba(processed_data)[0]), 4),
            model_used=model_name
        )

        logger.info(f"Prediction successful: {response.prediction_label} (prob: {probability:.4f})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/api/predict-batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(
    batch_input: BatchPatientInput,
    model_name: str = "xgboost"
):
    """
    Make predictions for multiple patients.

    - **batch_input**: List of patient data (max 100)
    - **model_name**: Model to use for predictions

    Returns list of predictions with processing time.
    """
    try:
        start_time = datetime.now()
        logger.info(f"Batch prediction request received: {len(batch_input.patients)} patients")

        # Get model
        model = get_model_or_404(model_name)

        # Process all predictions
        predictions = []
        for patient in batch_input.patients:
            processed_data = preprocess_input(patient)
            prediction = int(model.predict(processed_data)[0])
            probability = float(model.predict_proba(processed_data)[0][1])

            predictions.append(PredictionOutput(
                prediction=prediction,
                prediction_label="Diabetes" if prediction == 1 else "No Diabetes",
                probability=round(probability, 4),
                risk_level=calculate_risk_level(probability),
                confidence=round(max(model.predict_proba(processed_data)[0]), 4),
                model_used=model_name
            ))

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = BatchPredictionOutput(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )

        logger.info(f"Batch prediction successful: {len(predictions)} predictions in {processing_time:.2f}ms")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch prediction: {str(e)}"
        )


@app.get("/api/models", response_model=List[ModelMetrics], tags=["Models"])
async def list_models():
    """
    List all available models with their performance metrics.

    Returns list of models with accuracy, precision, recall, F1, and ROC-AUC scores.
    """
    try:
        models_list = []

        for model_name in models_cache.keys():
            # Get metrics from cache or set defaults
            metrics = model_metrics.get(model_name, {})

            models_list.append(ModelMetrics(
                model_name=model_name,
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', metrics.get('cv_score', 0.0)),
                roc_auc=metrics.get('roc_auc', 0.0),
                is_loaded=True,
                last_updated=datetime.now().isoformat()
            ))

        logger.info(f"Retrieved {len(models_list)} models")
        return models_list

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@app.get("/api/model/{model_name}/metrics", response_model=ModelMetrics, tags=["Models"])
async def get_model_metrics(model_name: str):
    """
    Get performance metrics for a specific model.

    - **model_name**: Name of the model (decision_tree, random_forest, xgboost)

    Returns detailed metrics for the specified model.
    """
    try:
        # Verify model exists
        get_model_or_404(model_name)

        # Get metrics
        metrics = model_metrics.get(model_name, {})

        response = ModelMetrics(
            model_name=model_name,
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', metrics.get('cv_score', 0.0)),
            roc_auc=metrics.get('roc_auc', 0.0),
            is_loaded=True,
            last_updated=datetime.now().isoformat()
        )

        logger.info(f"Retrieved metrics for {model_name}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model metrics: {str(e)}"
        )


@app.get("/api/model/{model_name}/feature-importance",
         response_model=List[FeatureImportance],
         tags=["Models"])
async def get_feature_importance(model_name: str, top_n: int = 10):
    """
    Get feature importance for a specific model.

    - **model_name**: Name of the model
    - **top_n**: Number of top features to return (default: 10)

    Returns ranked list of features with importance scores.
    """
    try:
        # Get model
        model = get_model_or_404(model_name)

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_booster'):
            # XGBoost
            importance_dict = model.get_booster().get_score(importance_type='gain')
            importances = np.zeros(len(feature_names) if feature_names else 8)
            for i in range(len(importances)):
                importances[i] = importance_dict.get(f'f{i}', 0)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {model_name} does not support feature importance"
            )

        # Create feature importance list
        base_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        features_used = feature_names if feature_names else base_features
        importance_list = [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in zip(features_used, importances)
        ]

        # Sort by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)

        # Take top N and add rank
        top_features = [
            FeatureImportance(
                feature=item['feature'],
                importance=round(item['importance'], 6),
                rank=idx + 1
            )
            for idx, item in enumerate(importance_list[:top_n])
        ]

        logger.info(f"Retrieved top {len(top_features)} features for {model_name}")
        return top_features

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting feature importance: {str(e)}"
        )


@app.post("/api/compare-models", response_model=ModelComparisonOutput, tags=["Predictions"])
async def compare_models(patient: PatientInput):
    """
    Compare predictions from all available models.

    - **patient**: Patient data for prediction

    Returns predictions from all models with consensus.
    """
    try:
        logger.info("Model comparison request received")

        # Preprocess input once
        processed_data = preprocess_input(patient)

        # Get predictions from all models
        predictions = {}
        prediction_values = []

        for model_name, model in models_cache.items():
            prediction = int(model.predict(processed_data)[0])
            probability = float(model.predict_proba(processed_data)[0][1])

            predictions[model_name] = PredictionOutput(
                prediction=prediction,
                prediction_label="Diabetes" if prediction == 1 else "No Diabetes",
                probability=round(probability, 4),
                risk_level=calculate_risk_level(probability),
                confidence=round(max(model.predict_proba(processed_data)[0]), 4),
                model_used=model_name
            )

            prediction_values.append(prediction)

        # Calculate consensus
        consensus_prediction = int(np.round(np.mean(prediction_values)))
        agreement_count = sum(1 for p in prediction_values if p == consensus_prediction)
        agreement_percentage = (agreement_count / len(prediction_values)) * 100

        response = ModelComparisonOutput(
            input_data=patient.dict(),
            predictions=predictions,
            consensus_prediction=consensus_prediction,
            consensus_label="Diabetes" if consensus_prediction == 1 else "No Diabetes",
            agreement_percentage=round(agreement_percentage, 2)
        )

        logger.info(f"Model comparison complete: {len(predictions)} models, "
                   f"{agreement_percentage:.1f}% agreement")
        return response

    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}"
        )


@app.get("/api/data-stats", response_model=DataStats, tags=["Data"])
async def get_data_stats():
    """
    Get statistics about the training dataset.

    Returns dataset size, feature count, class distribution, and feature statistics.
    """
    try:
        # Try to load training data
        data_path = os.path.join("data", "processed", "X_train.csv")
        labels_path = os.path.join("data", "processed", "y_train.csv")

        if os.path.exists(data_path) and os.path.exists(labels_path):
            X_train = pd.read_csv(data_path)
            y_train = pd.read_csv(labels_path)

            # Calculate statistics
            base_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            # Filter to base features if we have engineered features
            if len(X_train.columns) > 8:
                X_base = X_train[base_features] if all(f in X_train.columns for f in base_features) else X_train.iloc[:, :8]
                X_base.columns = base_features
            else:
                X_base = X_train
                X_base.columns = base_features

            feature_stats = {}
            for col in X_base.columns:
                feature_stats[col] = {
                    'mean': float(X_base[col].mean()),
                    'std': float(X_base[col].std()),
                    'min': float(X_base[col].min()),
                    'max': float(X_base[col].max()),
                    'median': float(X_base[col].median())
                }

            response = DataStats(
                total_samples=len(X_train),
                features_count=len(base_features),
                class_distribution={
                    'No Diabetes': int((y_train.values == 0).sum()),
                    'Diabetes': int((y_train.values == 1).sum())
                },
                feature_statistics=feature_stats
            )

            logger.info("Retrieved dataset statistics")
            return response
        else:
            # Return dummy stats if data not available
            logger.warning("Training data not found, returning default statistics")
            return DataStats(
                total_samples=0,
                features_count=8,
                class_distribution={'No Diabetes': 0, 'Diabetes': 0},
                feature_statistics={}
            )

    except Exception as e:
        logger.error(f"Error getting data stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting data stats: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*80)
    print("DIABETES PREDICTION API")
    print("="*80)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("ReDoc Documentation: http://localhost:8000/redoc")
    print("Health Check: http://localhost:8000/health")
    print("="*80)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
