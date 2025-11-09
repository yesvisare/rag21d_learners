param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

# Load settings from environment with fallbacks
$USERS = if ($env:STRESS_USERS) { $env:STRESS_USERS } else { "1000" }
$SPAWN_RATE = if ($env:STRESS_SPAWN_RATE) { $env:STRESS_SPAWN_RATE } else { "50" }
$RUNTIME = if ($env:STRESS_RUNTIME) { $env:STRESS_RUNTIME } else { "15m" }

# Ensure results directory exists
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting STRESS test ($USERS users, $RUNTIME)..."
Write-Host "Target: $HOST"
Write-Host "Artifacts: results/stress_stats.csv, results/stress.html"

locust -f locustfile.py `
    --host=$HOST `
    --users $USERS `
    --spawn-rate $SPAWN_RATE `
    --run-time $RUNTIME `
    --headless `
    --csv results/stress `
    --html results/stress.html
