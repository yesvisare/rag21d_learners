# M4.4 Test Runner (PowerShell)
# Runs pytest with proper PYTHONPATH setup

$env:PYTHONPATH = "$PWD"
pytest -q
