Write-Host "Starting SMOKE test (10 users, 2 min)..."
locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 2m --headless
