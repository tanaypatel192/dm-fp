# PowerShell script to start both backend and frontend
# Usage: .\start_all.ps1

param(
    [switch]$SkipTest = $false
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Diabetes Prediction System Launcher" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Python is installed
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    Write-Host "ERROR: Node.js is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python found: $($pythonCmd.Version)" -ForegroundColor Green
Write-Host "✓ Node.js found: $(node --version)" -ForegroundColor Green
Write-Host ""

# Function to start backend
$startBackend = {
    Write-Host "Starting Backend..." -ForegroundColor Yellow
    Set-Location backend
    
    # Activate virtual environment if it exists
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "  Activating virtual environment..." -ForegroundColor Gray
        & .\venv\Scripts\Activate.ps1
    }
    
    Write-Host "  Starting FastAPI server..." -ForegroundColor Gray
    python app.py
}

# Function to start frontend
$startFrontend = {
    Start-Sleep -Seconds 3  # Wait for backend to start
    Write-Host "Starting Frontend..." -ForegroundColor Yellow
    Set-Location frontend
    
    Write-Host "  Starting Vite dev server..." -ForegroundColor Gray
    npm run dev
}

# Start backend in new window
Write-Host "[1/2] Launching Backend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$startBackend}"

# Wait a bit for backend to start
Write-Host "      Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "      ✓ Backend is healthy!" -ForegroundColor Green
    Write-Host "      Models loaded: $($response.models_loaded)" -ForegroundColor Gray
}
catch {
    Write-Host "      ⚠ Backend health check failed, but continuing..." -ForegroundColor Yellow
}

# Start frontend in new window
Write-Host "`n[2/2] Launching Frontend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$startFrontend}"

Write-Host "      Waiting for frontend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Check if frontend is running
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "      ✓ Frontend is accessible!" -ForegroundColor Green
}
catch {
    Write-Host "      ⚠ Frontend check failed, but it may still be starting..." -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SYSTEM STARTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Application URLs:" -ForegroundColor White
Write-Host "  Frontend:     " -NoNewline
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend API:  " -NoNewline
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Health Check: " -NoNewline
Write-Host "http://localhost:8000/health" -ForegroundColor Cyan

Write-Host "`nPress " -NoNewline -ForegroundColor White
Write-Host "Ctrl+C" -NoNewline -ForegroundColor Yellow
Write-Host " in each window to stop the servers." -ForegroundColor White

# Run tests if not skipped
if (-not $SkipTest) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Running Quick Tests..." -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Start-Sleep -Seconds 3
    python quick_test.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ All tests passed!" -ForegroundColor Green
    }
    else {
        Write-Host "`n⚠ Some tests failed. Check the output above." -ForegroundColor Yellow
    }
}

Write-Host "`nPress any key to open the frontend in your browser..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Start-Process "http://localhost:5173"

Write-Host "`nSetup complete! Happy testing! 🚀" -ForegroundColor Green




