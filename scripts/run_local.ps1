# PowerShell script to run M1.3 Document Processing API locally
# Usage: .\scripts\run_local.ps1

Write-Host "Starting M1.3 Document Processing Pipeline API..." -ForegroundColor Green

# Set PYTHONPATH to include the project root
$env:PYTHONPATH = "$PWD"

Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Cyan

# Check if .env file exists
if (Test-Path ".env") {
    Write-Host "✓ .env file found" -ForegroundColor Green
} else {
    Write-Host "⚠️  No .env file found. API will run without external services." -ForegroundColor Yellow
    Write-Host "   Copy .env.example to .env and add your API keys to enable full functionality." -ForegroundColor Yellow
}

# Run uvicorn with reload
Write-Host "`nStarting server on http://localhost:8000" -ForegroundColor Green
Write-Host "API docs: http://localhost:8000/docs`n" -ForegroundColor Cyan

uvicorn app:app --reload --host 0.0.0.0 --port 8000
