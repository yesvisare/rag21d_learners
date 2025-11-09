# Run M3.3 API Security locally
# Sets PYTHONPATH to allow app.py to find src/ directory

$env:PYTHONPATH = "$PWD"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
