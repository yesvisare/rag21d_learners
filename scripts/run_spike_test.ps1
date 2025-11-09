param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

# Load settings from environment with fallbacks
$USERS = if ($env:SPIKE_USERS) { $env:SPIKE_USERS } else { "500" }
$SPAWN_RATE = if ($env:SPIKE_SPAWN_RATE) { $env:SPIKE_SPAWN_RATE } else { "500" }
$RUNTIME = if ($env:SPIKE_RUNTIME) { $env:SPIKE_RUNTIME } else { "5m" }

# Ensure results directory exists
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting SPIKE test ($USERS users, $RUNTIME)..."
Write-Host "Target: $HOST"
Write-Host "Artifacts: results/spike_stats.csv, results/spike.html"

locust -f locustfile.py `
    --host=$HOST `
    --users $USERS `
    --spawn-rate $SPAWN_RATE `
    --run-time $RUNTIME `
    --headless `
    --csv results/spike `
    --html results/spike.html
