# M3.1 Docker Containerization - Local Development Script
# PowerShell script to run the application locally without Docker

Write-Host "Starting M3.1 Docker Containerization API locally..." -ForegroundColor Green

# Set PYTHONPATH to include the project root
$env:PYTHONPATH = "$PWD"

# Run uvicorn with hot reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
