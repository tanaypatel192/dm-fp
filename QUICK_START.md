# Quick Start Guide - Diabetes Prediction System

⚡ **Get up and running in 5 minutes!**

---

## 🎯 TL;DR - Fastest Way to Test

```bash
# Windows (PowerShell)
.\start_all.ps1

# Windows (Command Prompt)
start_all.bat

# Linux/Mac
chmod +x start_all.sh
./start_all.sh
```

That's it! The script will:
- ✅ Start the backend server
- ✅ Start the frontend dev server
- ✅ Run automated tests
- ✅ Open your browser

---

## 📋 Prerequisites

Make sure you have these installed:

| Software | Version | Check Command |
|----------|---------|---------------|
| **Python** | 3.8+ | `python --version` |
| **Node.js** | 16+ | `node --version` |
| **npm** | 8+ | `npm --version` |

---

## 🚀 Manual Setup (If Scripts Don't Work)

### Step 1: Backend (Terminal 1)

```bash
cd backend

# Create & activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models (if needed)
python src/preprocessing.py
python src/decision_tree_model.py
python src/random_forest_model.py
python src/xgboost_model.py

# Start server
python app.py
```

✅ **Success:** You should see `Uvicorn running on http://0.0.0.0:8000`

### Step 2: Frontend (Terminal 2)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

✅ **Success:** You should see `Local: http://localhost:5173/`

### Step 3: Test (Terminal 3)

```bash
# Quick automated test
python quick_test.py
```

✅ **Success:** All tests should pass with ✓ marks

---

## 🌐 Access the Application

Once everything is running:

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:5173 | Main web interface |
| **API Docs** | http://localhost:8000/docs | Interactive API testing |
| **Health Check** | http://localhost:8000/health | Verify backend status |

---

## 🧪 Testing Options

### Option 1: Automated Tests (Recommended)

```bash
python quick_test.py
```

Runs comprehensive tests on all endpoints.

### Option 2: Backend Test Script

```bash
cd backend
python test_api.py
```

Detailed API endpoint testing with examples.

### Option 3: Interactive API Testing

1. Open http://localhost:8000/docs
2. Click any endpoint
3. Click "Try it out"
4. Enter parameters
5. Click "Execute"

### Option 4: Manual Testing via UI

1. Open http://localhost:5173
2. Navigate through pages:
   - **Dashboard** - System overview
   - **Single Prediction** - Predict for one patient
   - **Batch Analysis** - Upload CSV for multiple patients
   - **Model Comparison** - Compare all models
   - **Visualizations** - Explore data
   - **Model Explainability** - Understand how models work

---

## 📊 Test a Sample Prediction

### Via API (curl)

```bash
curl -X POST "http://localhost:8000/api/predict?model_name=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### Via Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    params={"model_name": "xgboost"},
    json={
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
)

print(response.json())
```

### Via UI

1. Go to http://localhost:5173
2. Click "Single Prediction"
3. Click "High Risk Example"
4. Click "Predict Risk"

---

## 🔍 What to Check

### ✅ Backend is Working

- Health check returns: `{"status": "healthy"}`
- Models loaded: `"models_loaded": 3`
- API docs accessible at `/docs`

### ✅ Frontend is Working

- Dashboard loads without errors
- Charts display correctly
- Forms accept input
- Predictions return results

### ✅ Integration is Working

- Frontend can call backend APIs
- No CORS errors in browser console
- Predictions complete successfully
- Data flows correctly

---

## ❌ Troubleshooting

### Backend Won't Start

**Problem:** `Port 8000 already in use`

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Problem:** `Models not found`

```bash
cd backend
python src/preprocessing.py
python src/decision_tree_model.py
python src/random_forest_model.py
python src/xgboost_model.py
```

### Frontend Won't Start

**Problem:** `Port 5173 already in use`

```bash
# Use different port
npm run dev -- --port 3001
```

**Problem:** `Module not found`

```bash
rm -rf node_modules package-lock.json
npm install
```

### Connection Errors

**Problem:** Frontend can't reach backend

1. Verify backend is running: http://localhost:8000/health
2. Check browser console for CORS errors
3. Verify API URL in frontend code
4. Check firewall settings

---

## 📚 Full Documentation

For comprehensive testing instructions, see:

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing documentation
- **[TEST_CHECKLIST.md](TEST_CHECKLIST.md)** - Detailed testing checklist
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend documentation
- **[README.md](README.md)** - Project overview

---

## 🎯 Quick Test Checklist

Use this minimal checklist for quick verification:

- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Health check returns "healthy"
- [ ] Can make a single prediction
- [ ] Dashboard loads correctly
- [ ] No console errors
- [ ] All 3 models loaded

**If all checked** ✅ **System is working!**

---

## 📞 Need Help?

1. Check the [TESTING_GUIDE.md](TESTING_GUIDE.md) troubleshooting section
2. Review logs:
   - Backend: Check terminal output
   - Frontend: Check browser console (F12)
3. Verify all prerequisites are installed
4. Try running `python quick_test.py` for diagnostic info

---

## 🎉 Success!

If you can see the dashboard and make predictions, you're all set!

### Next Steps:

1. **Explore the UI** - Try different pages and features
2. **Test with your data** - Upload a CSV for batch analysis
3. **Compare models** - See how different models perform
4. **Read the docs** - Learn about advanced features
5. **Check the code** - Understand the implementation

---

## 🚦 System Status Indicators

When everything is running correctly, you should see:

```
✓ Backend: http://localhost:8000/health returns {"status": "healthy"}
✓ Frontend: http://localhost:5173 displays dashboard
✓ API Docs: http://localhost:8000/docs is accessible
✓ Models: 3 models loaded (decision_tree, random_forest, xgboost)
✓ Tests: quick_test.py shows all tests passing
```

---

**Happy Testing! 🧪🚀**

For detailed testing procedures, see [TESTING_GUIDE.md](TESTING_GUIDE.md)




