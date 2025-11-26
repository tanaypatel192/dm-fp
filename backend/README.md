# Diabetes Prediction API - Backend

FastAPI-based REST API for diabetes prediction using machine learning models.

## Features

- **Multiple ML Models**: Decision Tree, Random Forest, XGBoost
- **Comprehensive Endpoints**: Single/batch predictions, model comparison, metrics
- **Input Validation**: Pydantic models with range checking
- **Auto Documentation**: Swagger UI and ReDoc
- **CORS Support**: Ready for frontend integration
- **Error Handling**: Detailed error messages with proper status codes
- **Model Caching**: Fast predictions with pre-loaded models

## API Endpoints

### Health & Status

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "models_loaded": 3,
  "available_models": ["decision_tree", "random_forest", "xgboost"]
}
```

### Predictions

#### `POST /api/predict`
Make a single prediction

**Parameters:**
- `model_name` (query): Model to use (default: "xgboost")

**Request Body:**
```json
{
  "Pregnancies": 6,
  "Glucose": 148.0,
  "BloodPressure": 72.0,
  "SkinThickness": 35.0,
  "Insulin": 0.0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Diabetes",
  "probability": 0.78,
  "risk_level": "High",
  "confidence": 0.78,
  "model_used": "xgboost"
}
```

#### `POST /api/predict-batch`
Make predictions for multiple patients (max 100)

**Parameters:**
- `model_name` (query): Model to use

**Request Body:**
```json
{
  "patients": [
    {
      "Pregnancies": 6,
      "Glucose": 148.0,
      ...
    },
    {
      "Pregnancies": 1,
      "Glucose": 85.0,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_processed": 2,
  "processing_time_ms": 45.23
}
```

#### `POST /api/compare-models`
Compare predictions from all models

**Request Body:** Same as single prediction

**Response:**
```json
{
  "input_data": {...},
  "predictions": {
    "decision_tree": {...},
    "random_forest": {...},
    "xgboost": {...}
  },
  "consensus_prediction": 1,
  "consensus_label": "Diabetes",
  "agreement_percentage": 100.0
}
```

### Model Information

#### `GET /api/models`
List all available models with metrics

**Response:**
```json
[
  {
    "model_name": "xgboost",
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.79,
    "f1_score": 0.80,
    "roc_auc": 0.88,
    "is_loaded": true,
    "last_updated": "2025-01-15T10:30:00"
  },
  ...
]
```

#### `GET /api/model/{model_name}/metrics`
Get metrics for specific model

**Response:** Same as single model object above

#### `GET /api/model/{model_name}/feature-importance`
Get feature importance

**Parameters:**
- `top_n` (query): Number of top features (default: 10)

**Response:**
```json
[
  {
    "feature": "Glucose",
    "importance": 0.2543,
    "rank": 1
  },
  {
    "feature": "BMI",
    "importance": 0.1876,
    "rank": 2
  },
  ...
]
```

### Data Statistics

#### `GET /api/data-stats`
Get dataset statistics

**Response:**
```json
{
  "total_samples": 614,
  "features_count": 8,
  "class_distribution": {
    "No Diabetes": 400,
    "Diabetes": 214
  },
  "feature_statistics": {
    "Glucose": {
      "mean": 121.7,
      "std": 30.4,
      "min": 44.0,
      "max": 199.0,
      "median": 117.0
    },
    ...
  }
}
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r ../requirements.txt
```

3. Ensure models are trained and saved in `backend/models/`

4. Ensure preprocessed data is in `backend/data/processed/`

## Running the API

### Development Mode

```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Accessing Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/api/predict?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148.0,
    "BloodPressure": 72.0,
    "SkinThickness": 35.0,
    "Insulin": 0.0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'

# List models
curl http://localhost:8000/api/models

# Feature importance
curl "http://localhost:8000/api/model/xgboost/feature-importance?top_n=5"
```

### Using Python requests

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make prediction
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

response = requests.post(
    f"{BASE_URL}/api/predict",
    params={"model_name": "xgboost"},
    json=patient_data
)
print(response.json())

# Compare models
response = requests.post(
    f"{BASE_URL}/api/compare-models",
    json=patient_data
)
print(response.json())
```

### Using Swagger UI

1. Open http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"

## Input Validation

All patient inputs are validated with the following constraints:

| Feature | Type | Min | Max | Description |
|---------|------|-----|-----|-------------|
| Pregnancies | int | 0 | 20 | Number of pregnancies |
| Glucose | float | 0 | 300 | Glucose level (mg/dL) |
| BloodPressure | float | 0 | 200 | Blood pressure (mm Hg) |
| SkinThickness | float | 0 | 100 | Skin thickness (mm) |
| Insulin | float | 0 | 900 | Insulin level (mu U/ml) |
| BMI | float | 0 | 70 | Body Mass Index |
| DiabetesPedigreeFunction | float | 0 | 3 | Diabetes pedigree function |
| Age | int | 1 | 120 | Age in years |

Zero values in critical fields (Glucose, BloodPressure, BMI, Insulin, SkinThickness) are automatically replaced with median values during preprocessing.

## Risk Levels

Risk levels are calculated based on prediction probability:

- **Low**: < 30%
- **Medium**: 30% - 70%
- **High**: > 70%

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (model not found)
- **500**: Internal Server Error

Error responses include:
```json
{
  "error": "Error description",
  "status_code": 404,
  "timestamp": "2025-01-15T10:30:00"
}
```

## CORS Configuration

CORS is configured to allow all origins by default. For production, modify in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-frontend-domain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance

- **Single Prediction**: ~10-50ms
- **Batch Prediction (100)**: ~500-2000ms
- **Model Comparison**: ~30-150ms

Times vary based on model complexity and hardware.

## Logging

All API requests and errors are logged with timestamps. Logs include:

- Request details
- Model used
- Prediction results
- Error messages (if any)

## Directory Structure

```
backend/
├── app.py                  # FastAPI application
├── README.md              # This file
├── data/
│   └── processed/         # Preprocessed data and scaler
├── models/                # Trained model files
├── src/                   # Model implementations
└── results/               # Model results and metrics
```

## Troubleshooting

### Models not loading

- Ensure models are trained and saved in `backend/models/`
- Check file names match: `decision_tree_model.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl`

### Scaler not found

- Run preprocessing pipeline to generate `backend/data/processed/scaler.pkl`

### Port already in use

```bash
# Use different port
uvicorn app:app --port 8001
```

### CORS errors

- Check CORS configuration in `app.py`
- Ensure frontend origin is allowed

## Production Deployment

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Security Considerations

For production deployment:

1. **Enable HTTPS**: Use reverse proxy (nginx, Apache)
2. **Rate Limiting**: Implement rate limiting middleware
3. **Authentication**: Add API key or JWT authentication
4. **Input Sanitization**: Already handled by Pydantic
5. **CORS**: Restrict to specific origins
6. **Monitoring**: Add logging and monitoring (e.g., Prometheus, Grafana)

## License

MIT License

## Support

For issues or questions, please refer to the main project documentation.
