param([string]$HOST = $env:LOADTEST_HOST)

# Fallback to default if not set
if (-not $HOST) { $HOST = "http://localhost:8000" }

# Load settings from environment with fallbacks
$USERS = if ($env:SOAK_USERS) { $env:SOAK_USERS } else { "50" }
$SPAWN_RATE = if ($env:SOAK_SPAWN_RATE) { $env:SOAK_SPAWN_RATE } else { "5" }
$RUNTIME = if ($env:SOAK_RUNTIME) { $env:SOAK_RUNTIME } else { "4h" }

# Ensure results directory exists
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting SOAK test ($USERS users, $RUNTIME)..."
Write-Host "Target: $HOST"
Write-Host "Artifacts: results/soak_stats.csv, results/soak.html"

locust -f locustfile.py `
    --host=$HOST `
    --users $USERS `
    --spawn-rate $SPAWN_RATE `
    --run-time $RUNTIME `
    --headless `
    --csv results/soak `
    --html results/soak.html
