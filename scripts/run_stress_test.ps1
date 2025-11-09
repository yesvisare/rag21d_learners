Write-Host "Starting STRESS test (1000 users, 15 min)..."
locust -f locustfile.py --host=http://localhost:8000 --users 1000 --spawn-rate 50 --run-time 15m --headless
