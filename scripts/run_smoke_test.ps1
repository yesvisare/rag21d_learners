param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

# Load settings from environment with fallbacks
$USERS = if ($env:SMOKE_USERS) { $env:SMOKE_USERS } else { "10" }
$SPAWN_RATE = if ($env:SMOKE_SPAWN_RATE) { $env:SMOKE_SPAWN_RATE } else { "2" }
$RUNTIME = if ($env:SMOKE_RUNTIME) { $env:SMOKE_RUNTIME } else { "2m" }

# Ensure results directory exists
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting SMOKE test ($USERS users, $RUNTIME)..."
Write-Host "Target: $HOST"
Write-Host "Artifacts: results/smoke_stats.csv, results/smoke.html"

locust -f locustfile.py `
    --host=$HOST `
    --users $USERS `
    --spawn-rate $SPAWN_RATE `
    --run-time $RUNTIME `
    --headless `
    --csv results/smoke `
    --html results/smoke.html
