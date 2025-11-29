@echo off
echo.
echo ========================================
echo   AUTO SETUP - NO INPUT REQUIRED
echo ========================================
echo.

cd backend

REM Create sample dataset automatically
if not exist data\raw\diabetes.csv (
    echo Creating sample dataset...
    if not exist data\raw mkdir data\raw
    (
        echo Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
        echo 6,148,72,35,0,33.6,0.627,50,1
        echo 1,85,66,29,0,26.6,0.351,31,0
        echo 8,183,64,0,0,23.3,0.672,32,1
        echo 1,89,66,23,94,28.1,0.167,21,0
        echo 0,137,40,35,168,43.1,2.288,33,1
    ) > data\raw\diabetes.csv
)

REM Setup venv and install
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Start servers in separate windows
start "Backend" cmd /k "cd /d %CD% && call venv\Scripts\activate.bat && python app.py"

cd ..\frontend
start "Frontend" cmd /k "cd /d %CD% && npm run dev"

cd ..
echo.
echo Servers starting...
echo Frontend: http://localhost:5173
echo Backend: http://localhost:8000/docs
echo.
timeout /t 15
start http://localhost:5173


