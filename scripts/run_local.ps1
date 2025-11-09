# M2.1 Caching Strategies - Local Development Server
#
# This script starts the FastAPI development server with proper PYTHONPATH configuration.
#
# Usage:
#   .\scripts\run_local.ps1

Write-Host "Starting M2.1 Caching Strategies API..." -ForegroundColor Green
Write-Host "Setting PYTHONPATH to project root..." -ForegroundColor Yellow

# Set PYTHONPATH to current directory
$env:PYTHONPATH = "$PWD"

Write-Host "PYTHONPATH = $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting uvicorn server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
