# ============================================================================
# M1.1 Vector Databases - Local Development Server Launcher (Windows)
# ============================================================================
# This script starts the FastAPI development server with auto-reload enabled.
#
# Usage:
#   .\scripts\run_local.ps1
#
# Or from PowerShell in repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_local.ps1
# ============================================================================

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "M1.1 Vector Databases - Starting Development Server" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Set PYTHONPATH to include repo root
$env:PYTHONPATH = $PWD

Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Green
Write-Host ""

# Check if uvicorn is installed
try {
    $uvicornVersion = & python -m uvicorn --version 2>&1
    Write-Host "✓ Uvicorn found: $uvicornVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Uvicorn not found. Installing dependencies..." -ForegroundColor Red
    & python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "  Host: 0.0.0.0"
Write-Host "  Port: 8000"
Write-Host "  Reload: Enabled"
Write-Host ""
Write-Host "API Documentation will be available at:" -ForegroundColor Cyan
Write-Host "  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Start uvicorn server
& python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
