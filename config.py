"""
Configuration for M2.3 Production Monitoring Dashboard
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Metrics configuration
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "rag-service")
ENVIRONMENT = os.getenv("ENV", "development")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json or text

# Cost tracking (example values - adjust based on your models)
COST_PER_1K_INPUT_TOKENS = float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0.003"))
COST_PER_1K_OUTPUT_TOKENS = float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0.015"))

# Alert thresholds
LATENCY_P95_THRESHOLD_MS = float(os.getenv("LATENCY_P95_THRESHOLD_MS", "2000"))
ERROR_RATE_THRESHOLD = float(os.getenv("ERROR_RATE_THRESHOLD", "0.05"))  # 5%
CACHE_HIT_MIN_THRESHOLD = float(os.getenv("CACHE_HIT_MIN_THRESHOLD", "0.30"))  # 30%
