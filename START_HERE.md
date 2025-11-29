# 🚀 START HERE - Complete A-Z Guide

## 🎯 Current Status

✅ **GPU Detected:** NVIDIA GeForce RTX 4070  
✅ **Project Structure:** Ready  
✅ **Scripts Created:** Multiple start options  
⚠️ **Models:** Need to be trained  
⚠️ **Dependencies:** Need proper installation  

---

## 🚦 Choose Your Path

###  **PATH 1: Quick Demo (5 minutes)** - RECOMMENDED FOR TESTING

Just want to see the UI and test the system?

```bash
.\SIMPLE_START.bat
```

**What this does:**
- ✅ Sets up basic environment
- ✅ Starts backend & frontend servers
- ✅ Opens browser automatically
- ⚠️ Models won't be loaded (API will show model errors)
- ✅ UI will work and you can test the interface

**Best for:** Quick testing, UI demo, checking if everything connects

---

### **PATH 2: Full Setup with Pre-trained Models** - BEST OPTION

**Step 1:** Download the dataset
1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place it in: `backend\data\raw\diabetes.csv`

**Step 2:** Install dependencies properly
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install packages (use pre-built wheels)
pip install fastapi==0.109.0
pip install uvicorn[standard]==0.27.0
pip install python-multipart==0.0.6
pip install pandas==2.1.4
pip install numpy==1.26.2
pip install scipy==1.11.4
pip install xgboost==2.0.3
pip install joblib==1.3.2

# Install scikit-learn (pre-built wheel)
pip install scikit-learn==1.3.2 --only-binary :all:

# Install shap
pip install shap==0.44.0
```

**Step 3:** Run preprocessing & training
```bash
# Preprocess data
python src\preprocessing.py

# Train models with GPU
python train_models_gpu.py
```

**Step 4:** Start servers
```bash
# Backend (Terminal 1)
cd backend
venv\Scripts\activate
python app.py

# Frontend (Terminal 2)  
cd frontend
npm install
npm run dev
```

**Best for:** Full experience with actual predictions

---

### **PATH 3: Use Docker** - EASIEST (if you have Docker)

Coming soon - will handle all dependencies automatically!

---

## 🔧 Dependency Installation Issues?

### Issue: "Microsoft Visual C++ 14.0 required"

**Solution 1:** Install pre-built wheels
```bash
pip install scikit-learn --only-binary :all:
```

**Solution 2:** Install Visual C++ Build Tools
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart terminal
4. Run: `pip install -r requirements.txt`

**Solution 3:** Use Anaconda (easiest)
```bash
# Download Anaconda from: https://www.anaconda.com/download
conda create -n diabetes python=3.10
conda activate diabetes
conda install scikit-learn pandas numpy xgboost fastapi uvicorn -c conda-forge
```

---

## 📊 What Each Script Does

| Script | What It Does | Time | Requires |
|--------|--------------|------|----------|
| `SIMPLE_START.bat` | Start servers only | 2 min | Nothing |
| `AUTORUN.bat` | Auto setup + start | 5 min | Internet |
| `RUN_ALL.bat` | Complete setup | 15 min | Dataset |
| `complete_setup.ps1` | Full PowerShell version | 20 min | Everything |
| `run_simple.ps1` | Just start servers | 1 min | Trained models |

---

## 🎮 GPU Acceleration

Your **RTX 4070** will automatically accelerate XGBoost when you:
1. Train models with: `python train_models_gpu.py`
2. The script auto-detects GPU
3. Uses `tree_method='gpu_hist'`
4. **10x faster** training!

---

## ✅ Quick Verification Checklist

Before starting, verify:

- [ ] Python 3.8+ installed: `python --version`
- [ ] Node.js 16+ installed: `node --version`
- [ ] GPU working: `nvidia-smi`
- [ ] In project root: `cd C:\Users\tanay\OneDrive\Desktop\dmfp\dm-fp`

---

## 🚀 Quickest Way to Get Running

**Option A: Just see the UI (30 seconds)**
```bash
cd frontend
npm install  # first time only
npm run dev
# Visit: http://localhost:5173
```

**Option B: Full system (5 minutes if dependencies installed)**
```bash
# Terminal 1
cd backend
venv\Scripts\activate
python app.py

# Terminal 2
cd frontend
npm run dev

# Browser
http://localhost:5173
```

---

## 📚 Available Documentation

| File | Purpose |
|------|---------|
| `START_HERE.md` | **This file - start here!** |
| `QUICK_START.md` | 5-minute quick start |
| `GPU_SETUP_GUIDE.md` | GPU acceleration guide |
| `TESTING_GUIDE.md` | Complete testing procedures |
| `TEST_CHECKLIST.md` | Testing checklist |
| `TESTING_SUMMARY.md` | Testing overview |

---

## 🆘 Troubleshooting

### Backend won't start
```bash
# Check if port is in use
netstat -ano | findstr :8000
# Kill process if needed
taskkill /PID <process_id> /F
```

### Frontend won't start
```bash
# Clear and reinstall
cd frontend
rmdir /s /q node_modules
npm install
```

### Models not loading
```bash
# Train them
cd backend
python train_models_gpu.py
```

### GPU not detected
```bash
# Check GPU
nvidia-smi
# Update drivers if needed
```

---

## 🎯 Recommended: Complete Setup Steps

1. **Install dependencies properly** (see PATH 2 above)
2. **Download dataset** or use sample data
3. **Train models** with GPU: `python train_models_gpu.py`
4. **Start backend**: `python app.py`
5. **Start frontend**: `npm run dev`
6. **Test**: `python quick_test.py`
7. **Use**: Visit http://localhost:5173

---

## 💡 Pro Tips

1. **First time?** Use `SIMPLE_START.bat` to test the UI
2. **Have dataset?** Follow PATH 2 for full experience
3. **Training slow?** Your GPU makes it 10x faster!
4. **Need help?** Check `TESTING_GUIDE.md`
5. **Want to test?** Run `python quick_test.py`

---

## 📞 Next Steps

1. **Choose a path above** (PATH 1 recommended for first try)
2. **Follow the instructions**
3. **Visit http://localhost:5173** when servers start
4. **Explore the UI:**
   - Dashboard
   - Single Prediction
   - Batch Analysis
   - Model Comparison
   - Visualizations
   - Model Explainability

---

## 🎉 You're Ready!

**Simplest way to start right now:**

```bash
.\SIMPLE_START.bat
```

Wait 30 seconds, then visit: **http://localhost:5173**

---

**Need more help? Check the other documentation files!**

- Questions about GPU? → `GPU_SETUP_GUIDE.md`
- Want to test? → `TESTING_GUIDE.md`
- Need quick start? → `QUICK_START.md`

**Happy predicting! 🚀**


