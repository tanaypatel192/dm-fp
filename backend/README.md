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

### Prerequisites

Install production dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Configure environment variables:**
Edit `.env` with your production values (database, Redis, API keys, etc.)

### Production Features

#### 1. Redis Caching

**Setup Redis:**
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
redis-server

# Test connection
redis-cli ping
```

**Cache Configuration** (`src/cache.py`):
- Prediction caching with 1-hour TTL
- Model metrics caching with 2-hour TTL
- Feature importance caching
- Automatic cache invalidation
- Exponential backoff retry logic

**Usage:**
```python
from src.cache import cache, cached, CacheTTL

# Cached decorator
@cached(ttl=CacheTTL.PREDICTION)
async def get_prediction(data):
    return prediction

# Direct cache access
await cache.set("key", value, ttl=3600)
result = await cache.get("key")
```

#### 2. Database Integration

**Setup Database** (`src/database.py`):

**SQLite (development):**
```bash
# Automatic - no setup required
# Database file: diabetes_predictions.db
```

**PostgreSQL (production):**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb diabetes_db

# Update DATABASE_URL in .env:
DATABASE_URL=postgresql+asyncpg://user:password@localhost/diabetes_db
```

**Database Models:**
- `Prediction` - All predictions for auditing
- `BatchJob` - Batch processing jobs
- `BatchPrediction` - Individual batch predictions
- `ModelMetric` - Model performance history
- `APILog` - API request logs

**Features:**
- Automatic table creation
- Async SQLAlchemy ORM
- Indexed queries for performance
- Relationship mapping

#### 3. Rate Limiting

**Configuration** (`src/rate_limit.py`):
- Global limits: 200/hour, 50/minute
- Prediction endpoints: 100/minute
- Batch endpoints: 10/hour
- Customizable per endpoint

**Rate Limits:**
```python
from src.rate_limit import rate_limit, RateLimits

@app.post("/api/predict")
@rate_limit(RateLimits.PREDICTION)
async def predict():
    ...
```

**Client Identification:**
- API key (if provided)
- IP address (fallback)

**Headers Returned:**
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `Retry-After` (when exceeded)

#### 4. Structured Logging

**Setup** (`src/logging_config.py`):
```python
from src.logging_config import setup_logging

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="logs",
    json_format=False  # True for production
)
```

**Log Files:**
- `logs/app.log` - General application logs
- `logs/error.log` - Error logs only
- `logs/api_requests.log` - API request logs (JSON)
- `logs/performance.log` - Performance metrics (JSON)

**Features:**
- Structured logging with loguru
- Automatic log rotation (500MB)
- Log retention (30/60 days)
- Compression (zip)
- JSON format support
- Performance timing
- Request ID tracing

#### 5. Security

**Features** (`src/security.py`):
- Input validation and sanitization
- SQL injection prevention
- XSS prevention
- Security headers (CSP, HSTS, etc.)
- API key authentication (optional)
- Content-Type validation
- Rate limiting per IP
- Request logging

**Security Headers:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security`
- `Referrer-Policy`

**Input Validation:**
```python
from src.security import InputValidator

# Validate patient data
validated = InputValidator.validate_patient_data(data)

# Validate model name
model = InputValidator.validate_model_name("xgboost")
```

#### 6. Async Optimization

**Features:**
- All database operations are async
- Async Redis cache operations
- Background tasks for batch processing
- Concurrent request handling
- Connection pooling

#### 7. Deployment with Gunicorn

**Using gunicorn.conf.py:**
```bash
gunicorn -c gunicorn.conf.py app:app
```

**Configuration:**
- Multiple workers (CPU count * 2 + 1)
- Uvicorn worker class
- Worker recycling after 1000 requests
- Graceful worker restarts
- Health check hooks

**Manual deployment:**
```bash
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

#### 8. Nginx Reverse Proxy

**Setup:**
```bash
# Copy nginx config
sudo cp nginx.conf /etc/nginx/sites-available/diabetes-api

# Create symlink
sudo ln -s /etc/nginx/sites-available/diabetes-api /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

**Features:**
- SSL/TLS termination
- Rate limiting by endpoint
- Connection limiting
- Request buffering
- Compression (gzip)
- Static file serving
- Load balancing support
- Security headers
- Health check endpoint

#### 9. Systemd Service

Create `/etc/systemd/system/diabetes-api.service`:
```ini
[Unit]
Description=Diabetes Prediction API
After=network.target redis.service postgresql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/diabetes-api
Environment="PATH=/var/www/diabetes-api/venv/bin"
ExecStart=/var/www/diabetes-api/venv/bin/gunicorn -c gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**Manage service:**
```bash
sudo systemctl enable diabetes-api
sudo systemctl start diabetes-api
sudo systemctl status diabetes-api
sudo systemctl restart diabetes-api
```

### Performance Monitoring

**Metrics tracked:**
- Request/response times
- Cache hit/miss rates
- Database query performance
- Model prediction times
- Error rates
- Active connections

**Logging queries:**
```bash
# View API requests
tail -f logs/api_requests.log

# View errors only
tail -f logs/error.log

# View performance metrics
tail -f logs/performance.log

# Nginx access logs
tail -f /var/log/nginx/diabetes-api-access.log
```

### Environment Variables Reference

See `.env.example` for all available configuration options:

**Critical Settings:**
- `ENVIRONMENT=production`
- `DEBUG=False`
- `DATABASE_URL` - Production database
- `REDIS_URL` - Redis connection
- `SECRET_KEY` - Change from default
- `CORS_ORIGINS` - Restrict to your frontend
- `API_KEYS` - Enable API authentication

### Health Checks

**Application health:**
```bash
curl http://localhost:8000/health
```

**Redis health:**
```bash
redis-cli ping
```

**Database health:**
```bash
# SQLite
ls -lh diabetes_predictions.db

# PostgreSQL
psql -U user -d diabetes_db -c "SELECT 1"
```

**Nginx health:**
```bash
curl http://localhost:8080/health
```

### Backup and Recovery

**Database backup:**
```bash
# SQLite
cp diabetes_predictions.db diabetes_predictions.db.backup

# PostgreSQL
pg_dump diabetes_db > backup.sql
```

**Redis backup:**
```bash
redis-cli SAVE
cp /var/lib/redis/dump.rdb dump.rdb.backup
```

### Security Checklist

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Configure `API_KEYS` if using authentication
- [ ] Restrict `CORS_ORIGINS` to frontend domain
- [ ] Enable HTTPS with valid SSL certificate
- [ ] Set `HTTPS_ONLY=True` in production
- [ ] Configure firewall (UFW, iptables)
- [ ] Set up log rotation
- [ ] Enable database backups
- [ ] Configure Redis password
- [ ] Review and adjust rate limits
- [ ] Set up monitoring (Sentry, etc.)

### Troubleshooting

**Check logs:**
```bash
# Application logs
tail -100 logs/app.log

# Gunicorn errors
tail -100 logs/error.log

# Nginx errors
tail -100 /var/log/nginx/diabetes-api-error.log

# System logs
journalctl -u diabetes-api -n 100
```

**Common issues:**
- **502 Bad Gateway**: Check if Gunicorn is running
- **429 Too Many Requests**: Rate limit exceeded
- **Connection refused**: Redis/PostgreSQL not running
- **Slow responses**: Check database/Redis connection

## License

MIT License

## Support

For issues or questions, please refer to the main project documentation.
