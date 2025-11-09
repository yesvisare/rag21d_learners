# PowerShell script to run the M2.2 FastAPI application locally
# Usage: .\scripts\run_local.ps1

Write-Host "Starting M2.2 Prompt Optimization API..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = "$PWD"

Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Yellow

# Run uvicorn with reload
Write-Host "Launching uvicorn on http://0.0.0.0:8000..." -ForegroundColor Green
Write-Host "API docs available at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
