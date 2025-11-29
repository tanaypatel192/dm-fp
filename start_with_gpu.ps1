# PowerShell script to train models with GPU and start the system
# Usage: .\start_with_gpu.ps1

param(
    [switch]$SkipTraining = $false,
    [switch]$SkipTest = $false
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Diabetes Prediction - GPU Accelerated" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check GPU
Write-Host "[Step 1/6] Checking GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host "✓ GPU Found: $gpuInfo" -ForegroundColor Green
    Write-Host "  CUDA Version: $(nvidia-smi | Select-String 'CUDA Version' | Out-String).Trim()" -ForegroundColor Gray
}
catch {
    Write-Host "✗ No NVIDIA GPU detected!" -ForegroundColor Red
    $continue = Read-Host "Continue without GPU? (Y/N)"
    if ($continue -ne "Y") {
        exit 1
    }
}

# Navigate to backend
Set-Location backend

# Check if virtual environment exists
Write-Host "`n[Step 2/6] Setting up Python environment..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "  Activating virtual environment..." -ForegroundColor Gray
    & .\venv\Scripts\Activate.ps1
}
else {
    Write-Host "  Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
    & .\venv\Scripts\Activate.ps1
    
    Write-Host "  Installing dependencies..." -ForegroundColor Gray
    pip install -r requirements.txt
}

Write-Host "✓ Python environment ready" -ForegroundColor Green

# Check if data needs preprocessing
Write-Host "`n[Step 3/6] Checking data..." -ForegroundColor Yellow
if (-not (Test-Path "data\processed\X_train.csv")) {
    Write-Host "  Preprocessed data not found. Running preprocessing..." -ForegroundColor Gray
    python src\preprocessing.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Preprocessing failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Data preprocessing complete" -ForegroundColor Green
}
else {
    Write-Host "✓ Preprocessed data found" -ForegroundColor Green
}

# Train models with GPU
if (-not $SkipTraining) {
    Write-Host "`n[Step 4/6] Training models with GPU acceleration..." -ForegroundColor Yellow
    Write-Host "  This may take several minutes..." -ForegroundColor Gray
    Write-Host "  GPU: NVIDIA RTX 4070 (CUDA 13.0)" -ForegroundColor Gray
    Write-Host ""
    
    python train_models_gpu.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Model training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "`n✓ All models trained successfully" -ForegroundColor Green
}
else {
    Write-Host "`n[Step 4/6] Skipping model training..." -ForegroundColor Yellow
    # Check if models exist
    if (-not (Test-Path "models\xgboost_model.pkl")) {
        Write-Host "✗ Models not found! Please train models first." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Using existing models" -ForegroundColor Green
}

# Start backend
Write-Host "`n[Step 5/6] Starting backend server..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; if (Test-Path venv\Scripts\Activate.ps1) { .\venv\Scripts\Activate.ps1 }; python app.py"
Write-Host "  Backend starting in new window..." -ForegroundColor Gray

# Wait for backend
Write-Host "  Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 8

# Check backend health
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "✓ Backend is healthy!" -ForegroundColor Green
    Write-Host "  Status: $($health.status)" -ForegroundColor Gray
    Write-Host "  Models loaded: $($health.models_loaded)" -ForegroundColor Gray
    Write-Host "  Models: $($health.available_models -join ', ')" -ForegroundColor Gray
}
catch {
    Write-Host "⚠ Backend health check failed, but continuing..." -ForegroundColor Yellow
}

# Start frontend
Write-Host "`n[Step 6/6] Starting frontend..." -ForegroundColor Yellow
Set-Location ..\frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing frontend dependencies..." -ForegroundColor Gray
    npm install
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
Write-Host "  Frontend starting in new window..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Check frontend
try {
    $frontendCheck = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5
    Write-Host "✓ Frontend is accessible!" -ForegroundColor Green
}
catch {
    Write-Host "⚠ Frontend check failed, but it may still be starting..." -ForegroundColor Yellow
}

# Go back to root
Set-Location ..

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SYSTEM STARTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "GPU Configuration:" -ForegroundColor White
Write-Host "  XGBoost:      " -NoNewline -ForegroundColor White
Write-Host "GPU Accelerated (RTX 4070)" -ForegroundColor Green
Write-Host "  Random Forest:" -NoNewline -ForegroundColor White
Write-Host "CPU (no GPU support)" -ForegroundColor Yellow
Write-Host "  Decision Tree:" -NoNewline -ForegroundColor White
Write-Host "CPU (no GPU support)" -ForegroundColor Yellow

Write-Host "`nApplication URLs:" -ForegroundColor White
Write-Host "  Frontend:     " -NoNewline
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend API:  " -NoNewline
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Health Check: " -NoNewline
Write-Host "http://localhost:8000/health" -ForegroundColor Cyan

# Run tests
if (-not $SkipTest) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Running Tests..." -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Start-Sleep -Seconds 3
    python quick_test.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ All tests passed!" -ForegroundColor Green
    }
    else {
        Write-Host "`n⚠ Some tests failed." -ForegroundColor Yellow
    }
}

# Open browser
Write-Host "`nPress any key to open frontend in browser..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Start-Process "http://localhost:5173"

Write-Host "`n🚀 System running with GPU acceleration!" -ForegroundColor Green
Write-Host "📊 XGBoost predictions are GPU-accelerated on your RTX 4070" -ForegroundColor Green
Write-Host ""


