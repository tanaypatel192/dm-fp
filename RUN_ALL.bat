@echo off
SETLOCAL EnableDelayedExpansion

echo.
echo ================================================================
echo   DIABETES PREDICTION SYSTEM - COMPLETE SETUP
echo ================================================================
echo.

REM Step 1: Check Prerequisites
echo [Step 1/8] Checking Prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo   [OK] Python found

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found!
    pause
    exit /b 1
)
echo   [OK] Node.js found

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [WARNING] No GPU detected - using CPU only
    set USE_GPU=false
) else (
    echo   [OK] GPU detected
    set USE_GPU=true
)

REM Step 2: Setup Backend
echo.
echo [Step 2/8] Setting up Backend...
cd backend

if not exist venv (
    echo   Creating virtual environment...
    python -m venv venv
)

echo   Activating virtual environment...
call venv\Scripts\activate.bat

echo   Installing dependencies...
pip install -r requirements.txt --quiet >nul 2>&1

REM Step 3: Create Directories
echo.
echo [Step 3/8] Creating directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist results\decision_tree mkdir results\decision_tree
if not exist results\random_forest mkdir results\random_forest
if not exist results\xgboost mkdir results\xgboost
echo   [OK] Directories ready

REM Step 4: Check Dataset
echo.
echo [Step 4/8] Checking dataset...
if not exist data\raw\diabetes.csv (
    echo   Dataset not found!
    echo   Please download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    echo   Place it in: backend\data\raw\diabetes.csv
    echo.
    set /p CONTINUE="Press ENTER to continue once downloaded, or type SKIP to create sample data: "
    
    if /i "!CONTINUE!"=="SKIP" (
        echo   Creating sample data...
        (
            echo Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
            echo 6,148,72,35,0,33.6,0.627,50,1
            echo 1,85,66,29,0,26.6,0.351,31,0
            echo 8,183,64,0,0,23.3,0.672,32,1
            echo 1,89,66,23,94,28.1,0.167,21,0
            echo 0,137,40,35,168,43.1,2.288,33,1
            echo 5,116,74,0,0,25.6,0.201,30,0
            echo 3,78,50,32,88,31.0,0.248,26,1
            echo 10,115,0,0,0,35.3,0.134,29,0
            echo 2,197,70,45,543,30.5,0.158,53,1
            echo 8,125,96,0,0,0.0,0.232,54,1
        ) > data\raw\diabetes.csv
        echo   [OK] Sample data created
    )
) else (
    echo   [OK] Dataset found
)

REM Step 5: Preprocessing
echo.
echo [Step 5/8] Data preprocessing...
if not exist data\processed\X_train.csv (
    echo   Running preprocessing...
    python src\preprocessing.py
    if errorlevel 1 (
        echo   ERROR: Preprocessing failed!
        pause
        exit /b 1
    )
    echo   [OK] Preprocessing complete
) else (
    echo   [OK] Data already preprocessed
)

REM Step 6: Train Models
echo.
echo [Step 6/8] Training models...
if not exist models\xgboost_model.pkl (
    echo   This will take 5-10 minutes...
    
    if "%USE_GPU%"=="true" (
        echo   Training with GPU acceleration...
        python train_models_gpu.py
    ) else (
        echo   Training with CPU...
        echo   Training Decision Tree...
        python src\decision_tree_model.py
        echo   Training Random Forest...
        python src\random_forest_model.py
        echo   Training XGBoost...
        python src\xgboost_model.py
    )
    
    if errorlevel 1 (
        echo   ERROR: Model training failed!
        pause
        exit /b 1
    )
    echo   [OK] Models trained
) else (
    echo   [OK] Models already trained
)

REM Step 7: Start Backend
echo.
echo [Step 7/8] Starting backend server...
start "Backend Server" cmd /k "cd /d %CD% && venv\Scripts\activate && python app.py"
echo   [OK] Backend starting...

timeout /t 10 /nobreak >nul

REM Step 8: Start Frontend
echo.
echo [Step 8/8] Starting frontend server...
cd ..\frontend

if not exist node_modules (
    echo   Installing frontend dependencies...
    call npm install
)

start "Frontend Server" cmd /k "cd /d %CD% && npm run dev"
echo   [OK] Frontend starting...

timeout /t 8 /nobreak >nul

REM Final Summary
cd ..
echo.
echo ================================================================
echo   SETUP COMPLETE!
echo ================================================================
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000/docs
echo   Health:    http://localhost:8000/health
echo.
echo   Two windows have opened for backend and frontend.
echo   Close those windows to stop the servers.
echo.

set /p OPEN="Open browser? (Y/N): "
if /i "%OPEN%"=="Y" (
    start http://localhost:5173
)

echo.
echo Press any key to exit this window...
pause >nul


