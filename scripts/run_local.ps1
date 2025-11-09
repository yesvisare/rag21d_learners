# PowerShell script to run M1.2 Pinecone Hybrid Search API locally
#
# Usage:
#   .\scripts\run_local.ps1
#
# Or from project root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1

Write-Host "Starting M1.2 Pinecone Hybrid Search API..." -ForegroundColor Green
Write-Host "Setting PYTHONPATH to current directory..." -ForegroundColor Yellow

# Set PYTHONPATH to current directory
$env:PYTHONPATH = $PWD

Write-Host "PYTHONPATH = $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting uvicorn server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API docs at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run uvicorn with reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
