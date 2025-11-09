# M2.3 Production Monitoring Dashboard - Local Development Script
#
# This script starts the FastAPI application with proper PYTHONPATH configuration
#
# Usage:
#   .\scripts\run_local.ps1
#
# The application will be available at:
#   - API: http://localhost:8001
#   - API Docs: http://localhost:8001/docs
#   - Prometheus Metrics: http://localhost:8000/metrics

Write-Host "="*60 -ForegroundColor Cyan
Write-Host "M2.3 Production Monitoring Dashboard - Starting" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan
Write-Host ""

# Set PYTHONPATH to current directory
$env:PYTHONPATH = "$PWD"
Write-Host "✓ PYTHONPATH set to: $PWD" -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
}
elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
    & ".venv\Scripts\Activate.ps1"
}
else {
    Write-Host "⚠ No virtual environment found. Using global Python." -ForegroundColor Yellow
    Write-Host "  Recommend creating one: python -m venv venv" -ForegroundColor Yellow
}

# Check if required packages are installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Cyan
python -c "import fastapi, uvicorn, prometheus_client" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Missing required packages. Installing..." -ForegroundColor Red
    pip install -r requirements.txt
}
else {
    Write-Host "✓ All dependencies installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting FastAPI application..." -ForegroundColor Cyan
Write-Host "  API: http://localhost:8001" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8001/docs" -ForegroundColor White
Write-Host "  Prometheus Metrics: http://localhost:8000/metrics" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8001
