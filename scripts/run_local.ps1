# PowerShell script to run M1.4 Query Pipeline API locally
# Usage: .\scripts\run_local.ps1

Write-Host "Starting M1.4 Query Pipeline API..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = "$PWD"

Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Cyan

# Run uvicorn with auto-reload
Write-Host "Running uvicorn on http://0.0.0.0:8000" -ForegroundColor Cyan
Write-Host "API docs available at http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop..." -ForegroundColor Yellow

uvicorn app:app --reload --host 0.0.0.0 --port 8000
