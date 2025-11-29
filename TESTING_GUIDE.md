# Complete Testing Guide for Diabetes Prediction System

This comprehensive guide walks you through testing the entire diabetes prediction project, from setup to deployment testing.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Backend Testing](#backend-testing)
5. [Frontend Testing](#frontend-testing)
6. [Integration Testing](#integration-testing)
7. [API Testing Tools](#api-testing-tools)
8. [Production Testing](#production-testing)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

For the impatient, here's the fastest way to get the entire system running:

```bash
# 1. Backend Setup
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# OR
source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt

# 2. Train models (if not already done)
python src/preprocessing.py
python src/decision_tree_model.py
python src/random_forest_model.py
python src/xgboost_model.py

# 3. Start backend
python app.py

# In a NEW terminal:
# 4. Frontend Setup
cd frontend
npm install
npm run dev

# 5. Open browser
# http://localhost:5173 (Frontend)
# http://localhost:8000/docs (API Documentation)
```

---

## Prerequisites

### Required Software

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Git**: For version control

### Optional Tools

- **Postman** or **Insomnia**: For API testing
- **Redis**: For caching (production)
- **PostgreSQL**: For database (production)

### Verify Installations

```bash
# Check Python
python --version

# Check Node.js
node --version

# Check npm
npm --version
```

---

## Project Setup

### 1. Clone/Navigate to Project

```bash
cd c:\Users\tanay\OneDrive\Desktop\dmfp\dm-fp
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Prepare Data and Models

The project needs:
1. Dataset (Pima Indians Diabetes Dataset)
2. Preprocessed data
3. Trained models

**Option A: Download Pre-trained Models** (if available)
- Place model files in `backend/models/`
  - `decision_tree_model.pkl`
  - `random_forest_model.pkl`
  - `xgboost_model.pkl`

**Option B: Train Models from Scratch**

```bash
# Step 1: Get the dataset
# Download diabetes.csv from Kaggle:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Place it in: backend/data/raw/diabetes.csv

# Step 2: Run preprocessing
python src/preprocessing.py

# Step 3: Train all models
python src/decision_tree_model.py
python src/random_forest_model.py
python src/xgboost_model.py
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file (if needed)
# The default API URL is http://localhost:8000
```

---

## Backend Testing

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

**Expected Output:**
```
================================================================================
DIABETES PREDICTION API
================================================================================

Starting FastAPI server...
API Documentation: http://localhost:8000/docs
ReDoc Documentation: http://localhost:8000/redoc
Health Check: http://localhost:8000/health
================================================================================
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading models...
✓ Scaler loaded
✓ Decision Tree model loaded
✓ Random Forest model loaded
✓ XGBoost model loaded
INFO:     Application startup complete.
```

### 2. Verify Backend Health

Open your browser or use curl:

```bash
# Browser
http://localhost:8000/health

# Or curl
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "models_loaded": 3,
  "available_models": ["decision_tree", "random_forest", "xgboost"]
}
```

### 3. Use Built-in Test Script

The project includes a comprehensive test script:

```bash
cd backend
python test_api.py
```

This will test:
- ✓ Health check
- ✓ Single predictions (all models)
- ✓ Batch predictions
- ✓ Model listing
- ✓ Model metrics
- ✓ Feature importance
- ✓ Model comparison
- ✓ Dataset statistics
- ✓ Error handling

### 4. Explore Interactive API Documentation

Visit the Swagger UI for interactive testing:

```
http://localhost:8000/docs
```

**What you can do:**
1. Click on any endpoint
2. Click "Try it out"
3. Enter parameters
4. Click "Execute"
5. See the response

**Key Endpoints to Test:**

#### Health Check
```
GET /health
```

#### Single Prediction
```
POST /api/predict
Parameters: model_name (xgboost, random_forest, decision_tree)
Body: {
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

#### Comprehensive Prediction with Explanations
```
POST /api/predict-explain
Body: Same as above
```

#### Compare All Models
```
POST /api/compare-models
Body: Same as above
```

#### Batch Predictions
```
POST /api/predict-batch
Body: {
  "patients": [
    { /* patient 1 */ },
    { /* patient 2 */ }
  ]
}
```

#### List Models
```
GET /api/models
```

#### Feature Importance
```
GET /api/model/xgboost/feature-importance?top_n=10
```

#### Dataset Statistics
```
GET /api/data-stats
```

### 5. Command Line API Testing

#### Using curl (Windows PowerShell)

```powershell
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/api/predict?model_name=xgboost" `
  -H "Content-Type: application/json" `
  -d '{
    \"Pregnancies\": 6,
    \"Glucose\": 148.0,
    \"BloodPressure\": 72.0,
    \"SkinThickness\": 35.0,
    \"Insulin\": 0.0,
    \"BMI\": 33.6,
    \"DiabetesPedigreeFunction\": 0.627,
    \"Age\": 50
  }'

# List models
curl http://localhost:8000/api/models
```

#### Using Python Requests

Create a file `quick_test.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Single prediction
patient = {
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
    json=patient
)
print("Prediction:", json.dumps(response.json(), indent=2))
```

Run it:
```bash
python quick_test.py
```

---

## Frontend Testing

### 1. Start the Frontend Development Server

```bash
cd frontend
npm run dev
```

**Expected Output:**
```
  VITE v5.0.8  ready in 823 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

### 2. Open in Browser

Navigate to: **http://localhost:5173**

### 3. Manual Testing Checklist

#### Dashboard Page
- [ ] Page loads without errors
- [ ] System status cards display
- [ ] Model cards show all 3 models
- [ ] Statistics cards display correct data
- [ ] Quick prediction form works
- [ ] Charts render properly
- [ ] Theme toggle works (dark/light mode)

#### Single Prediction Page
- [ ] Form displays all 8 input fields
- [ ] Sliders and number inputs are synchronized
- [ ] Example patient buttons work (High Risk, Low Risk, Moderate)
- [ ] Random patient generator works
- [ ] Form validation shows errors for invalid inputs
- [ ] Submit button triggers prediction
- [ ] Loading spinner appears during prediction
- [ ] Results section displays with:
  - Risk assessment gauge
  - Model predictions comparison table
  - Feature importance chart
  - SHAP waterfall chart (if available)
  - Risk factors grid
  - Recommendations list
  - Similar patients section
- [ ] Export functionality works (PDF, CSV)
- [ ] Feature Explorer works with real-time predictions

#### Batch Analysis Page
- [ ] File upload area displays
- [ ] Drag-and-drop works
- [ ] CSV file selection works
- [ ] File validation catches invalid formats
- [ ] Preview shows first 5 rows
- [ ] Process button starts batch prediction
- [ ] Progress bar updates
- [ ] Statistics dashboard displays:
  - Total patients count
  - Risk distribution pie chart
  - Probability histogram
  - Top 5 high-risk patients
- [ ] Results table displays with:
  - Sortable columns
  - Search functionality
  - Risk level filtering
  - Color-coded risk badges
- [ ] Click on patient row opens detailed view
- [ ] Export results as CSV works
- [ ] Export summary report works

#### Model Comparison Page
- [ ] Input form displays
- [ ] Submit triggers comparison
- [ ] All 3 models' predictions display
- [ ] Consensus prediction calculated
- [ ] Agreement percentage shown
- [ ] Comparison table formatted correctly
- [ ] Visualizations render

#### Visualization Dashboard Page
- [ ] Summary statistics cards load
- [ ] Feature distribution charts display
- [ ] Histogram shows diabetic vs non-diabetic
- [ ] Box plots render
- [ ] Correlation heatmap loads
- [ ] Heatmap is interactive (hover shows values)
- [ ] 3D scatter plot works
- [ ] Can rotate and zoom 3D plot
- [ ] Pairplot matrix displays
- [ ] Color scheme selector works
- [ ] Filter by outcome works
- [ ] Download chart buttons work

#### Model Explainability Page
- [ ] Education section displays
- [ ] Model selector tabs work
- [ ] Decision tree visualization renders
- [ ] Feature importance comparison loads
- [ ] Interactive bar chart works
- [ ] SHAP values section displays
- [ ] Example predictions load automatically
- [ ] Try-it-yourself form works
- [ ] Real-time predictions display
- [ ] Model comparison shows differences

#### About Page
- [ ] Content loads
- [ ] Model descriptions display
- [ ] Links work

### 4. Browser Console Check

Open browser DevTools (F12) and check:
- [ ] No JavaScript errors in Console
- [ ] Network tab shows successful API calls (200 status)
- [ ] No CORS errors

### 5. Responsive Design Testing

Test on different screen sizes:
- [ ] Desktop (1920x1080)
- [ ] Laptop (1366x768)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667)

Use browser DevTools Device Toolbar (Ctrl+Shift+M) to test.

### 6. Performance Testing

Check Performance tab in DevTools:
- [ ] First Contentful Paint < 2s
- [ ] Time to Interactive < 4s
- [ ] Page load time < 3s
- [ ] API response times < 500ms

---

## Integration Testing

### Test Full Stack Workflow

**Scenario 1: New Patient Prediction**

1. Start both backend and frontend
2. Navigate to Single Prediction page
3. Enter patient data:
   ```
   Pregnancies: 6
   Glucose: 148
   BloodPressure: 72
   SkinThickness: 35
   Insulin: 0
   BMI: 33.6
   DiabetesPedigreeFunction: 0.627
   Age: 50
   ```
4. Click "Predict Risk"
5. Verify results display correctly
6. Check browser Network tab: API call to `/api/predict-explain`
7. Verify response time < 2 seconds

**Scenario 2: Batch Analysis**

1. Create a test CSV file (`test_patients.csv`):
   ```csv
   Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
   6,148,72,35,0,33.6,0.627,50
   1,85,66,29,0,26.6,0.351,31
   8,183,64,0,0,23.3,0.672,32
   2,89,90,30,0,33.5,0.292,42
   ```
2. Go to Batch Analysis page
3. Upload CSV file
4. Verify preview shows data
5. Click "Process Batch"
6. Wait for completion
7. Verify statistics display
8. Check results table
9. Export results

**Scenario 3: Model Comparison**

1. Go to Model Comparison page
2. Enter same patient data
3. Click "Compare Models"
4. Verify all 3 models return predictions
5. Check consensus calculation
6. Verify agreement percentage

**Scenario 4: A/B Testing (if configured)**

1. Navigate to A/B Testing Admin
2. Create new experiment
3. Assign variants to different models
4. Make predictions
5. Track metrics
6. View analytics dashboard

---

## API Testing Tools

### Using Postman

1. **Import API**
   - Open Postman
   - Import > Link > `http://localhost:8000/openapi.json`

2. **Test Endpoints**
   - All endpoints auto-imported
   - Use collection runner for automated tests

3. **Create Test Collection**

   Save these requests:
   
   **Collection: Diabetes API Tests**
   
   ```
   ├── Health Check (GET /health)
   ├── Single Prediction - High Risk (POST /api/predict)
   ├── Single Prediction - Low Risk (POST /api/predict)
   ├── Batch Prediction (POST /api/predict-batch)
   ├── Comprehensive Prediction (POST /api/predict-explain)
   ├── List Models (GET /api/models)
   ├── Model Metrics - XGBoost (GET /api/model/xgboost/metrics)
   ├── Feature Importance (GET /api/model/xgboost/feature-importance)
   ├── Compare Models (POST /api/compare-models)
   └── Data Stats (GET /api/data-stats)
   ```

### Using Thunder Client (VS Code Extension)

1. Install Thunder Client extension
2. Import OpenAPI spec
3. Test endpoints directly in VS Code

---

## Production Testing

### Backend Production Testing

#### 1. Environment Setup

```bash
# Create .env file
cp .env.example .env

# Configure production settings
ENVIRONMENT=production
DEBUG=False
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

#### 2. Production Server Test

```bash
# With Gunicorn
gunicorn -c gunicorn.conf.py app:app

# Test
curl http://localhost:8000/health
```

#### 3. Load Testing

Using Apache Bench:

```bash
# Install Apache Bench
# Test health endpoint
ab -n 1000 -c 10 http://localhost:8000/health

# Test prediction endpoint
ab -n 100 -c 5 -p patient_data.json -T application/json \
  "http://localhost:8000/api/predict?model_name=xgboost"
```

#### 4. Stress Testing

Create `locustfile.py`:

```python
from locust import HttpUser, task, between

class DiabetesPredictionUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def health_check(self):
        self.client.get("/health")
    
    @task(3)
    def predict(self):
        self.client.post("/api/predict?model_name=xgboost", json={
            "Pregnancies": 6,
            "Glucose": 148.0,
            "BloodPressure": 72.0,
            "SkinThickness": 35.0,
            "Insulin": 0.0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        })
```

Run:
```bash
pip install locust
locust -f locustfile.py
```

Visit: http://localhost:8089

### Frontend Production Testing

#### 1. Build Production Bundle

```bash
cd frontend
npm run build
```

#### 2. Preview Production Build

```bash
npm run preview
```

#### 3. Performance Audit

```bash
# Using Lighthouse
npm install -g lighthouse

lighthouse http://localhost:4173 --view
```

#### 4. Bundle Size Analysis

```bash
npm run build:analyze
```

---

## Troubleshooting

### Backend Issues

#### Issue: Models not loading

**Symptoms:**
```
✗ XGBoost model not found at models/xgboost_model.pkl
```

**Solution:**
```bash
# Train models
python src/preprocessing.py
python src/xgboost_model.py
```

#### Issue: Scaler not found

**Symptoms:**
```
✗ Scaler not found at data/processed/scaler.pkl
```

**Solution:**
```bash
python src/preprocessing.py
```

#### Issue: Port already in use

**Symptoms:**
```
ERROR: [Errno 10048] error while attempting to bind on address
```

**Solution:**
```bash
# Windows - Find process on port 8000
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app:app --port 8001
```

#### Issue: CORS errors

**Symptoms:**
```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:5173' 
has been blocked by CORS policy
```

**Solution:**
Check `app.py` CORS configuration:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Frontend Issues

#### Issue: Cannot connect to API

**Symptoms:**
```
Network Error / Failed to fetch
```

**Solution:**
1. Verify backend is running: http://localhost:8000/health
2. Check API URL in frontend (should be http://localhost:8000)
3. Check browser console for errors

#### Issue: npm install fails

**Symptoms:**
```
npm ERR! code ERESOLVE
```

**Solution:**
```bash
# Clear cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install --legacy-peer-deps
```

#### Issue: Build fails

**Symptoms:**
```
TypeScript error...
```

**Solution:**
```bash
# Check TypeScript errors
npm run type-check

# Fix errors or temporarily disable strict mode in tsconfig.json
```

#### Issue: Vite port already in use

**Solution:**
```bash
# Use different port
npm run dev -- --port 3001
```

### Common Issues

#### Issue: Python version mismatch

**Solution:**
```bash
# Verify Python version
python --version  # Should be 3.8+

# Use specific Python version
py -3.10 -m venv venv
```

#### Issue: Module not found

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Issue: Dataset not found

**Solution:**
Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Place in: `backend/data/raw/diabetes.csv`

---

## Test Results Checklist

Use this checklist to verify complete testing:

### Backend
- [ ] Health check endpoint responds
- [ ] All 3 models load successfully
- [ ] Single predictions work (all models)
- [ ] Batch predictions work
- [ ] Comprehensive predictions with SHAP work
- [ ] Model comparison works
- [ ] Feature importance returns data
- [ ] Dataset statistics available
- [ ] Error handling works correctly
- [ ] API documentation accessible
- [ ] Response times < 500ms for single predictions
- [ ] Response times < 3s for batch (10 patients)

### Frontend
- [ ] All pages load without errors
- [ ] Single prediction form works
- [ ] Batch analysis uploads and processes CSV
- [ ] Model comparison displays results
- [ ] Visualizations render correctly
- [ ] Charts are interactive
- [ ] Export functionality works
- [ ] Responsive on mobile/tablet/desktop
- [ ] Theme toggle works
- [ ] No console errors
- [ ] Page load time < 3s

### Integration
- [ ] Frontend successfully calls backend APIs
- [ ] CORS configured correctly
- [ ] Data flows correctly from input to results
- [ ] Real-time predictions work
- [ ] Batch processing completes
- [ ] Error messages display properly

### Production Readiness
- [ ] Production build succeeds
- [ ] Environment variables configured
- [ ] Database connection works (if used)
- [ ] Redis caching works (if used)
- [ ] Rate limiting active
- [ ] Security headers present
- [ ] HTTPS configured
- [ ] Error logging works
- [ ] Performance metrics collected

---

## Automated Testing Script

Create `run_all_tests.ps1` (Windows PowerShell):

```powershell
# Diabetes Prediction - Complete Test Suite

Write-Host "================================" -ForegroundColor Green
Write-Host "Starting Complete Test Suite" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# 1. Check Backend
Write-Host "`n[1/4] Testing Backend..." -ForegroundColor Yellow
$backend = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction SilentlyContinue
if ($backend.status -eq "healthy") {
    Write-Host "✓ Backend is healthy" -ForegroundColor Green
} else {
    Write-Host "✗ Backend is not responding" -ForegroundColor Red
    exit 1
}

# 2. Check Frontend
Write-Host "`n[2/4] Testing Frontend..." -ForegroundColor Yellow
$frontend = Invoke-WebRequest -Uri "http://localhost:5173" -ErrorAction SilentlyContinue
if ($frontend.StatusCode -eq 200) {
    Write-Host "✓ Frontend is accessible" -ForegroundColor Green
} else {
    Write-Host "✗ Frontend is not responding" -ForegroundColor Red
    exit 1
}

# 3. Run API Tests
Write-Host "`n[3/4] Running API Tests..." -ForegroundColor Yellow
cd backend
python test_api.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ API tests passed" -ForegroundColor Green
} else {
    Write-Host "✗ API tests failed" -ForegroundColor Red
    exit 1
}

# 4. Summary
Write-Host "`n================================" -ForegroundColor Green
Write-Host "All Tests Passed!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "`nApplication URLs:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "  Backend API: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health Check: http://localhost:8000/health" -ForegroundColor White
```

Run:
```powershell
.\run_all_tests.ps1
```

---

## Continuous Testing

### Watch Mode (Development)

**Backend:**
```bash
# Auto-reload on file changes
uvicorn app:app --reload
```

**Frontend:**
```bash
# Hot module replacement
npm run dev
```

### Git Hooks (Pre-commit Testing)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running pre-commit tests..."

# Backend linting
cd backend
python -m pylint src/*.py || exit 1

# Frontend type checking
cd ../frontend
npm run type-check || exit 1

echo "Pre-commit tests passed!"
```

---

## Support

If you encounter issues:

1. Check the logs:
   - Backend: Console output
   - Frontend: Browser DevTools Console
   
2. Review the documentation:
   - Backend: `backend/README.md`
   - Frontend: `frontend/README.md`
   - Main: `README.md`

3. Common solutions in Troubleshooting section above

---

**Happy Testing! 🧪**




