Write-Host "Starting SPIKE test (500 users, 5 min)..."
locust -f locustfile.py --host=http://localhost:8000 --users 500 --spawn-rate 500 --run-time 5m --headless
