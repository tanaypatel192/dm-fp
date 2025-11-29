@echo off
REM Batch script to start both backend and frontend (Windows)
REM Usage: start_all.bat

echo.
echo ========================================
echo   Diabetes Prediction System Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo [OK] Python found
echo [OK] Node.js found
echo.

REM Start backend in new window
echo [1/2] Starting Backend Server...
start "Backend - Diabetes Prediction API" cmd /k "cd backend && (if exist venv\Scripts\activate.bat venv\Scripts\activate.bat) && python app.py"

REM Wait for backend to start
echo       Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

REM Start frontend in new window
echo.
echo [2/2] Starting Frontend Server...
start "Frontend - Diabetes Prediction UI" cmd /k "cd frontend && npm run dev"

REM Wait for frontend to start
echo       Waiting for frontend to initialize...
timeout /t 8 /nobreak >nul

echo.
echo ========================================
echo   SYSTEM STARTED SUCCESSFULLY!
echo ========================================
echo.

echo Application URLs:
echo   Frontend:     http://localhost:5173
echo   Backend API:  http://localhost:8000/docs
echo   Health Check: http://localhost:8000/health
echo.

echo Check the opened windows for server logs.
echo Close the windows to stop the servers.
echo.

REM Ask to run tests
set /p run_tests="Do you want to run quick tests? (Y/N): "
if /i "%run_tests%"=="Y" (
    echo.
    echo Running quick tests...
    echo.
    timeout /t 3 /nobreak >nul
    python quick_test.py
)

REM Ask to open browser
echo.
set /p open_browser="Open frontend in browser? (Y/N): "
if /i "%open_browser%"=="Y" (
    start http://localhost:5173
)

echo.
echo Setup complete! Happy testing!
echo.
pause




