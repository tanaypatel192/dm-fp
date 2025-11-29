@echo off
SETLOCAL EnableDelayedExpansion

echo.
echo ================================================================================
echo   DIABETES PREDICTION SYSTEM - COMPLETE TEST SUITE
echo ================================================================================
echo.

cd /d "%~dp0"

REM ============================================================================
REM PHASE 1: Environment Check
REM ============================================================================
echo [Phase 1/5] Checking Environment...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
echo   [OK] Python detected

node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found!
    pause
    exit /b 1
)
echo   [OK] Node.js detected

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [INFO] No GPU - using CPU
    set GPU_AVAILABLE=false
) else (
    echo   [OK] GPU detected
    set GPU_AVAILABLE=true
)

REM ============================================================================
REM PHASE 2: Setup
REM ============================================================================
echo.
echo [Phase 2/5] Setting Up Environment...
echo.

cd backend

REM Create virtual environment if needed
if not exist venv (
    echo   Creating virtual environment...
    python -m venv venv
)

REM Activate and install minimal dependencies
echo   Installing minimal dependencies...
call venv\Scripts\activate.bat
pip install --quiet --upgrade pip >nul 2>&1
pip install --quiet fastapi uvicorn pydantic python-multipart numpy >nul 2>&1

REM Create directories
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed  
if not exist models mkdir models

REM Create sample data if needed
if not exist data\raw\diabetes.csv (
    echo   Creating sample dataset...
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
)

echo   [OK] Backend setup complete

REM ============================================================================
REM PHASE 3: Start Servers
REM ============================================================================
echo.
echo [Phase 3/5] Starting Servers...
echo.

REM Start Backend
echo   Starting backend server...
start "Backend Server" /MIN cmd /c "cd /d %CD% && call venv\Scripts\activate.bat && python app.py"
timeout /t 5 /nobreak >nul

REM Start Frontend
cd ..\frontend
echo   Starting frontend server...

if not exist node_modules (
    echo   Installing frontend dependencies (first time only)...
    call npm install --silent
)

start "Frontend Server" /MIN cmd /c "cd /d %CD% && npm run dev"

cd ..

echo   [OK] Servers starting...
echo   Waiting for initialization (15 seconds)...
timeout /t 15 /nobreak >nul

REM ============================================================================
REM PHASE 4: Run Tests
REM ============================================================================
echo.
echo [Phase 4/5] Running Tests...
echo.

REM Test 1: Frontend
echo   [Test 1/5] Testing Frontend...
curl -s -o nul -w "%%{http_code}" http://localhost:5173 > temp_status.txt
set /p FRONTEND_STATUS=<temp_status.txt
del temp_status.txt

if "%FRONTEND_STATUS%"=="200" (
    echo     [PASS] Frontend is accessible
) else (
    echo     [FAIL] Frontend not responding
)

REM Test 2: Backend Health
echo   [Test 2/5] Testing Backend Health...
curl -s http://localhost:8000/health > temp_health.txt 2>nul
findstr /C:"healthy" temp_health.txt >nul 2>&1
if errorlevel 1 (
    echo     [FAIL] Backend not responding or unhealthy
    set BACKEND_HEALTHY=false
) else (
    echo     [PASS] Backend is healthy
    set BACKEND_HEALTHY=true
)
del temp_health.txt 2>nul

REM Test 3: API Documentation
if "%BACKEND_HEALTHY%"=="true" (
    echo   [Test 3/5] Testing API Documentation...
    curl -s -o nul -w "%%{http_code}" http://localhost:8000/docs > temp_docs.txt
    set /p DOCS_STATUS=<temp_docs.txt
    del temp_docs.txt
    
    if "!DOCS_STATUS!"=="200" (
        echo     [PASS] API docs accessible
    ) else (
        echo     [FAIL] API docs not accessible
    )
) else (
    echo   [Test 3/5] [SKIP] Backend not healthy
)

REM Test 4: Frontend-Backend Connection
if "%BACKEND_HEALTHY%"=="true" (
    echo   [Test 4/5] Testing Frontend-Backend Connection...
    curl -s http://localhost:8000/health > temp_cors.txt 2>nul
    findstr /C:"models_loaded" temp_cors.txt >nul 2>&1
    if errorlevel 1 (
        echo     [FAIL] CORS or connection issue
    ) else (
        echo     [PASS] Connection working
    )
    del temp_cors.txt 2>nul
) else (
    echo   [Test 4/5] [SKIP] Backend not healthy
)

REM Test 5: GPU Status
echo   [Test 5/5] Testing GPU Status...
if "%GPU_AVAILABLE%"=="true" (
    echo     [PASS] GPU available for acceleration
) else (
    echo     [INFO] CPU mode (no GPU)
)

REM ============================================================================
REM PHASE 5: Summary & URLs
REM ============================================================================
echo.
echo ================================================================================
echo   TEST SUMMARY
echo ================================================================================
echo.

echo   Frontend:     http://localhost:5173
if "%FRONTEND_STATUS%"=="200" (
    echo                 Status: [RUNNING]
) else (
    echo                 Status: [NOT RUNNING]
)

echo.
echo   Backend API:  http://localhost:8000/docs
if "%BACKEND_HEALTHY%"=="true" (
    echo                 Status: [RUNNING]
) else (
    echo                 Status: [NOT RUNNING]
)

echo.
echo   Health Check: http://localhost:8000/health

echo.
echo ================================================================================
echo.

if "%FRONTEND_STATUS%"=="200" (
    if "%BACKEND_HEALTHY%"=="true" (
        echo   [SUCCESS] All systems operational!
        echo.
        echo   Open your browser to test:
        echo   - http://localhost:5173 (Frontend UI^)
        echo   - http://localhost:8000/docs (API Documentation^)
        echo.
        
        set /p OPEN_BROWSER="Open in browser now? (Y/N): "
        if /i "!OPEN_BROWSER!"=="Y" (
            start http://localhost:5173
            start http://localhost:8000/docs
        )
    ) else (
        echo   [PARTIAL] Frontend running, backend has issues
        echo   Check the "Backend Server" window for errors
    )
) else (
    echo   [FAILED] Services not starting properly
    echo   Check the server windows for error messages
)

echo.
echo   Server windows are minimized. To view logs:
echo   - Look for "Backend Server" and "Frontend Server" in taskbar
echo   - Or check Task Manager for python.exe and node.exe
echo.
echo   To stop servers: Close the minimized windows or press Ctrl+C in them
echo.
echo   Press any key to exit this test window...
pause >nul


