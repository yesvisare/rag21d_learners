param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

Write-Host "Starting Locust Web UI on http://localhost:8089"
Write-Host "Target: $HOST"
Write-Host ""
Write-Host "Configure test parameters in the web interface."
Write-Host "Press Ctrl+C to stop."

locust -f locustfile.py --host=$HOST
