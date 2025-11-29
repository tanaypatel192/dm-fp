# Simple script to start both servers
Write-Host "Starting Diabetes Prediction System..." -ForegroundColor Cyan
Write-Host ""

# Start Backend
Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @"
-NoExit
-Command
Set-Location 'C:\Users\tanay\OneDrive\Desktop\dmfp\dm-fp\backend';
if (Test-Path 'venv\Scripts\Activate.ps1') { & '.\venv\Scripts\Activate.ps1' };
Write-Host ''; Write-Host '================================' -ForegroundColor Green;
Write-Host 'BACKEND SERVER' -ForegroundColor Green;
Write-Host '================================' -ForegroundColor Green; Write-Host '';
python app.py
"@

Start-Sleep -Seconds 5

# Start Frontend  
Write-Host "[2/2] Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @"
-NoExit
-Command
Set-Location 'C:\Users\tanay\OneDrive\Desktop\dmfp\dm-fp\frontend';
Write-Host ''; Write-Host '================================' -ForegroundColor Green;
Write-Host 'FRONTEND SERVER' -ForegroundColor Green;
Write-Host '================================' -ForegroundColor Green; Write-Host '';
npm run dev
"@

Write-Host ""
Write-Host "Servers starting..." -ForegroundColor Green
Write-Host ""
Write-Host "URLs:" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Wait 10 seconds then press any key to open browser..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Start-Process "http://localhost:5173"



