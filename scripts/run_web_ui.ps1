Write-Host "Starting Locust Web UI on http://localhost:8089"
locust -f locustfile.py --host=http://localhost:8000
