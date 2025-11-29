@echo off
echo.
echo ========================================
echo   SIMPLE START (Skip Training)
echo ========================================
echo.
echo This will start the servers without training models.
echo You can still test the API and UI!
echo.

cd backend

REM Quick setup
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install fastapi uvicorn pydantic numpy pandas --no-cache-dir
) else (
    call venv\Scripts\activate.bat
)

REM Create minimal structure
if not exist data\processed mkdir data\processed
if not exist models mkdir models

echo Starting backend...
start "Backend" cmd /k "cd /d %CD% && call venv\Scripts\activate.bat && python app.py"

cd ..\frontend

if not exist node_modules (
    echo Installing frontend...
    call npm install
)

echo Starting frontend...
start "Frontend" cmd /k "cd /d %CD% && npm run dev"

cd ..

echo.
echo ========================================
echo   SERVERS STARTING!
echo ========================================
echo.
echo   Wait 30 seconds, then visit:
echo   http://localhost:5173
echo.
echo   Note: Models may not be loaded yet.
echo   The app will still work for UI testing!
echo.

timeout /t 20
start http://localhost:5173


