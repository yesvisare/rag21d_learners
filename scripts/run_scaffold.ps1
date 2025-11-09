<#
.SYNOPSIS
    PowerShell wrapper for Portfolio Project Scaffolder

.DESCRIPTION
    Creates a complete portfolio project structure for RAG applications.
    This is a Windows-first wrapper around m4_3_portfolio_scaffold.py.

.PARAMETER Name
    Name of the project to create (default: DocuMentor)

.PARAMETER Path
    Base path for project creation (default: current directory)

.PARAMETER DryRun
    Preview what would be created without actually creating files

.PARAMETER Force
    Allow overwriting existing directory

.PARAMETER NoFrontend
    Skip frontend (React) structure generation

.PARAMETER NoCI
    Skip CI/CD (GitHub Actions) workflow generation

.EXAMPLE
    .\run_scaffold.ps1 -Name DocuMentor -Path .\output

.EXAMPLE
    .\run_scaffold.ps1 -Name MyProject -DryRun

.EXAMPLE
    .\run_scaffold.ps1 -Name BackendOnly -NoFrontend -NoCI
#>

param(
    [Parameter(Position=0)]
    [string]$Name = "DocuMentor",

    [Parameter()]
    [string]$Path = ".",

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [switch]$NoFrontend,

    [Parameter()]
    [switch]$NoCI
)

# Get the script directory (scripts folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Get the parent directory (workspace root)
$WorkspaceRoot = Split-Path -Parent $ScriptDir

# Path to the scaffolder script
$ScaffolderScript = Join-Path $WorkspaceRoot "m4_3_portfolio_scaffold.py"

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+ first."
    exit 1
}

# Check if scaffolder script exists
if (-not (Test-Path $ScaffolderScript)) {
    Write-Error "Scaffolder script not found at: $ScaffolderScript"
    exit 1
}

# Build command arguments
$Args = @($ScaffolderScript, $Name, "--path", $Path)

if ($DryRun) {
    $Args += "--dry-run"
}

if ($Force) {
    $Args += "--force"
}

if ($NoFrontend) {
    $Args += "--no-frontend"
}

if ($NoCI) {
    $Args += "--no-ci"
}

# Display command being run
Write-Host "Running: python $($Args -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Execute the scaffolder
& python $Args

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Scaffolder completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Scaffolder failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
