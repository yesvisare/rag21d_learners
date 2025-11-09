Write-Host "Starting LOAD test (100 users, 10 min)..."
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 10m --headless
