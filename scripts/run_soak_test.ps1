Write-Host "Starting SOAK test (50 users, 4 hours)..."
locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 4h --headless
