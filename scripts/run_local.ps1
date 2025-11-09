# M2.4 — Error Handling & Reliability
# Local development server script for Windows PowerShell

Write-Host "Starting M2.4 Error Handling & Reliability server..." -ForegroundColor Green

# Set PYTHONPATH to project root
$env:PYTHONPATH = "$PWD"

# Start uvicorn with hot reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Usage:
#   PS> .\scripts\run_local.ps1
#
# Or one-liner:
#   PS> powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
