param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

# Load settings from environment with fallbacks
$USERS = if ($env:LOAD_USERS) { $env:LOAD_USERS } else { "100" }
$SPAWN_RATE = if ($env:LOAD_SPAWN_RATE) { $env:LOAD_SPAWN_RATE } else { "10" }
$RUNTIME = if ($env:LOAD_RUNTIME) { $env:LOAD_RUNTIME } else { "10m" }

# Ensure results directory exists
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting LOAD test ($USERS users, $RUNTIME)..."
Write-Host "Target: $HOST"
Write-Host "Artifacts: results/load_stats.csv, results/load.html"

locust -f locustfile.py `
    --host=$HOST `
    --users $USERS `
    --spawn-rate $SPAWN_RATE `
    --run-time $RUNTIME `
    --headless `
    --csv results/load `
    --html results/load.html
