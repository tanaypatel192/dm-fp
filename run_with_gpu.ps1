# PowerShell script to setup and run everything with GPU acceleration
# Usage: .\run_with_gpu.ps1

$ErrorActionPreference = "Continue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Diabetes Prediction - GPU Accelerated" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check GPU
Write-Host "[GPU CHECK]" -ForegroundColor Yellow
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  No NVIDIA GPU detected!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ GPU detected and ready!`n" -ForegroundColor Green

# Navigate to backend
Set-Location backend

# Step 1: Setup data
Write-Host "[1/4] Setting up dataset..." -ForegroundColor Cyan
python quick_setup.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Setup failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Train models with GPU
Write-Host "`n[2/4] Training models with GPU acceleration..." -ForegroundColor Cyan
python train_models_gpu.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Model training had issues, but continuing..." -ForegroundColor Yellow
}

# Step 3: Start backend
Write-Host "`n[3/4] Starting backend server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🚀 Backend Server (GPU-Enabled)' -ForegroundColor Green; python app.py"
Start-Sleep -Seconds 5

# Check backend health
Write-Host "Checking backend health..." -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
    Write-Host "✓ Backend is running!" -ForegroundColor Green
    Write-Host "  Models loaded: $($response.models_loaded)" -ForegroundColor Gray
}
catch {
    Write-Host "⚠️  Backend may still be starting..." -ForegroundColor Yellow
}

# Step 4: Start frontend
Write-Host "`n[4/4] Starting frontend..." -ForegroundColor Cyan
Set-Location ..\frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Gray
    npm install --silent
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🎨 Frontend Server' -ForegroundColor Green; npm run dev"
Start-Sleep -Seconds 5

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  🎉 SYSTEM STARTED WITH GPU! 🚀" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Application URLs:" -ForegroundColor White
Write-Host "  Frontend:     " -NoNewline
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend API:  " -NoNewline
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Health Check: " -NoNewline
Write-Host "http://localhost:8000/health" -ForegroundColor Cyan

Write-Host "`n🔥 GPU Acceleration Status:" -ForegroundColor Yellow
Write-Host "  XGBoost: GPU-enabled (tree_method=gpu_hist)" -ForegroundColor Green
Write-Host "  Random Forest: CPU multi-threaded" -ForegroundColor White
Write-Host "  Decision Tree: CPU" -ForegroundColor White

Write-Host "`nPress any key to run tests..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Run tests
Set-Location ..
Write-Host "`n[TESTING]" -ForegroundColor Cyan
python quick_test.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ All tests passed!" -ForegroundColor Green
}
else {
    Write-Host "`n⚠️  Some tests failed. Check output above." -ForegroundColor Yellow
}

# Open browser
Write-Host "`nOpening browser..." -ForegroundColor Gray
Start-Process "http://localhost:5173"

Write-Host "`n🎉 Setup complete! GPU is accelerating your predictions!" -ForegroundColor Green
Write-Host "Close the terminal windows to stop the servers.`n" -ForegroundColor White

