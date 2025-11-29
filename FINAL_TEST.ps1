# Final Complete Test Script
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   COMPLETE SYSTEM TEST" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$project = "C:\Users\tanay\OneDrive\Desktop\dmfp\dm-fp"

# Phase 1: Environment Check
Write-Host "[1/5] Environment Check..." -ForegroundColor Yellow
$pythonOk = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
$nodeOk = $null -ne (Get-Command node -ErrorAction SilentlyContinue)
$gpuOk = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

Write-Host "  Python: $(if($pythonOk){'✓'}else{'✗'})" -ForegroundColor $(if($pythonOk){"Green"}else{"Red"})
Write-Host "  Node.js: $(if($nodeOk){'✓'}else{'✗'})" -ForegroundColor $(if($nodeOk){"Green"}else{"Red"})
Write-Host "  GPU: $(if($gpuOk){'✓'}else{'○'})" -ForegroundColor $(if($gpuOk){"Green"}else{"Yellow"})

# Phase 2: Kill existing processes
Write-Host "`n[2/5] Cleaning up old processes..." -ForegroundColor Yellow
Get-Process -Name python,node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "  Cleanup complete" -ForegroundColor Gray

# Phase 3: Start Backend
Write-Host "`n[3/5] Starting Backend..." -ForegroundColor Yellow
Set-Location "$project\backend"

if (Test-Path "venv\Scripts\python.exe") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$project\backend'; & '.\venv\Scripts\python.exe' app.py" -WindowStyle Normal
    Write-Host "  Backend starting..." -ForegroundColor Gray
} else {
    Write-Host "  Virtual environment not found!" -ForegroundColor Red
}

Start-Sleep -Seconds 8

# Phase 4: Start Frontend
Write-Host "`n[4/5] Starting Frontend..." -ForegroundColor Yellow
Set-Location "$project\frontend"

if (Test-Path "node_modules") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$project\frontend'; npm run dev" -WindowStyle Normal
    Write-Host "  Frontend starting..." -ForegroundColor Gray
} else {
    Write-Host "  Installing dependencies first..." -ForegroundColor Yellow
    npm install --silent
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$project\frontend'; npm run dev" -WindowStyle Normal
}

Start-Sleep -Seconds 10

# Phase 5: Test Everything
Write-Host "`n[5/5] Running Tests..." -ForegroundColor Yellow
Set-Location $project

$results = @()

# Test 1: Frontend
Write-Host "`n  Test 1: Frontend (port 5173)" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173" -Method Head -TimeoutSec 3 -UseBasicParsing
    Write-Host "    [PASS] Frontend is running" -ForegroundColor Green
    $results += "Frontend: PASS"
} catch {
    Write-Host "    [FAIL] Frontend not accessible" -ForegroundColor Red
    $results += "Frontend: FAIL"
}

# Test 2: Backend Health
Write-Host "`n  Test 2: Backend Health (port 8000)" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
    Write-Host "    [PASS] Backend is running" -ForegroundColor Green
    Write-Host "      Status: $($health.status)" -ForegroundColor Gray
    Write-Host "      Models: $($health.models_loaded)" -ForegroundColor Gray
    $results += "Backend: PASS"
} catch {
    Write-Host "    [FAIL] Backend not accessible" -ForegroundColor Red
    $results += "Backend: FAIL"
}

# Test 3: API Docs
Write-Host "`n  Test 3: API Documentation" -ForegroundColor Cyan
try {
    $docs = Invoke-WebRequest -Uri "http://localhost:8000/docs" -Method Head -TimeoutSec 3 -UseBasicParsing
    Write-Host "    [PASS] API docs accessible" -ForegroundColor Green
    $results += "API Docs: PASS"
} catch {
    Write-Host "    [FAIL] API docs not accessible" -ForegroundColor Red
    $results += "API Docs: FAIL"
}

# Test 4: Sample Prediction (if backend is up)
Write-Host "`n  Test 4: Sample API Call" -ForegroundColor Cyan
try {
    $testData = @{
        Pregnancies = 6
        Glucose = 148
        BloodPressure = 72
        SkinThickness = 35
        Insulin = 0
        BMI = 33.6
        DiabetesPedigreeFunction = 0.627
        Age = 50
    }
    
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/api/predict?model_name=xgboost" -Method Post -Body ($testData | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 5
    Write-Host "    [PASS] Prediction endpoint working" -ForegroundColor Green
    Write-Host "      Result: $($prediction.prediction_label)" -ForegroundColor Gray
    $results += "Prediction: PASS"
} catch {
    Write-Host "    [FAIL] Prediction failed (models may not be trained)" -ForegroundColor Yellow
    $results += "Prediction: FAIL (models not trained)"
}

# Test 5: GPU
Write-Host "`n  Test 5: GPU Acceleration" -ForegroundColor Cyan
if ($gpuOk) {
    Write-Host "    [PASS] GPU available for training" -ForegroundColor Green
    $results += "GPU: Available"
} else {
    Write-Host "    [INFO] CPU mode only" -ForegroundColor Yellow
    $results += "GPU: Not available"
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   TEST RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($result in $results) {
    if ($result -like "*PASS*") {
        Write-Host "  ✓ $result" -ForegroundColor Green
    } elseif ($result -like "*FAIL*") {
        Write-Host "  ✗ $result" -ForegroundColor Red
    } else {
        Write-Host "  ○ $result" -ForegroundColor Yellow
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   ACCESS URLS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "  Backend:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health:    http://localhost:8000/health" -ForegroundColor White

Write-Host "`n  Two PowerShell windows opened with server logs." -ForegroundColor Gray
Write-Host "  Close those windows to stop the servers.`n" -ForegroundColor Gray

# Open browser
$open = Read-Host "`nOpen in browser? (Y/N)"
if ($open -eq "Y" -or $open -eq "y") {
    Start-Process "http://localhost:5173"
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8000/docs"
}

Write-Host "`nTest complete! Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


