<#
.SYNOPSIS
    PowerShell wrapper for Test Structure Scaffolder

.DESCRIPTION
    Creates a complete test structure with pytest configuration.
    This is a Windows-first wrapper around tests_scaffold.py.

.PARAMETER Path
    Base path for test structure creation (default: current directory)

.EXAMPLE
    .\run_tests_scaffold.ps1

.EXAMPLE
    .\run_tests_scaffold.ps1 -Path .\my-project
#>

param(
    [Parameter(Position=0)]
    [string]$Path = "."
)

# Get the script directory (scripts folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Get the parent directory (workspace root)
$WorkspaceRoot = Split-Path -Parent $ScriptDir

# Path to the tests scaffolder script
$TestsScaffolderScript = Join-Path $WorkspaceRoot "tests_scaffold.py"

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+ first."
    exit 1
}

# Check if scaffolder script exists
if (-not (Test-Path $TestsScaffolderScript)) {
    Write-Error "Tests scaffolder script not found at: $TestsScaffolderScript"
    exit 1
}

# Build command arguments
$Args = @($TestsScaffolderScript, "--path", $Path)

# Display command being run
Write-Host "Running: python $($Args -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Execute the scaffolder
& python $Args

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Test scaffolder completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Test scaffolder failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
