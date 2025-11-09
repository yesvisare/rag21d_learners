"""
Configuration for M2.4 Error Handling & Reliability
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== API CONFIGURATION ====================

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30.0"))

# ==================== RETRY CONFIGURATION ====================

# Retry strategy defaults
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_INITIAL_DELAY = float(os.getenv("RETRY_INITIAL_DELAY", "1.0"))
RETRY_EXPONENTIAL_BASE = float(os.getenv("RETRY_EXPONENTIAL_BASE", "2.0"))
RETRY_ENABLE_JITTER = os.getenv("RETRY_ENABLE_JITTER", "true").lower() == "true"

# Operation-specific retry configs
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_INITIAL_DELAY = float(os.getenv("EMBEDDING_INITIAL_DELAY", "1.0"))

COMPLETION_MAX_RETRIES = int(os.getenv("COMPLETION_MAX_RETRIES", "2"))
COMPLETION_INITIAL_DELAY = float(os.getenv("COMPLETION_INITIAL_DELAY", "2.0"))

# ==================== CIRCUIT BREAKER CONFIGURATION ====================

# Circuit breaker thresholds
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RECOVERY_TIMEOUT = float(os.getenv("CB_RECOVERY_TIMEOUT", "60.0"))
CB_HALF_OPEN_MAX_CALLS = int(os.getenv("CB_HALF_OPEN_MAX_CALLS", "1"))

# ==================== QUEUE CONFIGURATION ====================

# Request queue settings
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "1000"))
QUEUE_WORKER_SLEEP = float(os.getenv("QUEUE_WORKER_SLEEP", "0.1"))

# Backpressure thresholds
QUEUE_WARNING_THRESHOLD = int(os.getenv("QUEUE_WARNING_THRESHOLD", "750"))  # 75% capacity
QUEUE_CRITICAL_THRESHOLD = int(os.getenv("QUEUE_CRITICAL_THRESHOLD", "950"))  # 95% capacity

# ==================== TIMEOUT CONFIGURATION ====================

# Service timeouts (in seconds)
VECTOR_DB_TIMEOUT = float(os.getenv("VECTOR_DB_TIMEOUT", "10.0"))
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "15.0"))
COMPLETION_TIMEOUT = float(os.getenv("COMPLETION_TIMEOUT", "30.0"))

# ==================== CACHE CONFIGURATION ====================

# Fallback cache settings
CACHE_MAX_AGE = int(os.getenv("CACHE_MAX_AGE", "3600"))  # 1 hour in seconds
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))  # Max cached items

# ==================== MONITORING & LOGGING ====================

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Metrics collection
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"
METRICS_INTERVAL = int(os.getenv("METRICS_INTERVAL", "60"))  # seconds

# ==================== FEATURE FLAGS ====================

# Enable/disable specific resilience features
ENABLE_RETRIES = os.getenv("ENABLE_RETRIES", "true").lower() == "true"
ENABLE_CIRCUIT_BREAKER = os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"
ENABLE_FALLBACKS = os.getenv("ENABLE_FALLBACKS", "true").lower() == "true"
ENABLE_QUEUEING = os.getenv("ENABLE_QUEUEING", "true").lower() == "true"

# ==================== HELPER FUNCTIONS ====================

def get_retry_config(operation: str = "default") -> dict:
    """Get retry configuration for specific operation."""
    configs = {
        "embedding": {
            "max_retries": EMBEDDING_MAX_RETRIES,
            "initial_delay": EMBEDDING_INITIAL_DELAY,
            "exponential_base": RETRY_EXPONENTIAL_BASE,
            "jitter": RETRY_ENABLE_JITTER
        },
        "completion": {
            "max_retries": COMPLETION_MAX_RETRIES,
            "initial_delay": COMPLETION_INITIAL_DELAY,
            "exponential_base": RETRY_EXPONENTIAL_BASE,
            "jitter": RETRY_ENABLE_JITTER
        },
        "default": {
            "max_retries": RETRY_MAX_ATTEMPTS,
            "initial_delay": RETRY_INITIAL_DELAY,
            "exponential_base": RETRY_EXPONENTIAL_BASE,
            "jitter": RETRY_ENABLE_JITTER
        }
    }
    return configs.get(operation, configs["default"])


def get_circuit_breaker_config() -> dict:
    """Get circuit breaker configuration."""
    return {
        "failure_threshold": CB_FAILURE_THRESHOLD,
        "recovery_timeout": CB_RECOVERY_TIMEOUT,
        "half_open_max_calls": CB_HALF_OPEN_MAX_CALLS
    }


def get_queue_config() -> dict:
    """Get queue configuration."""
    return {
        "max_size": QUEUE_MAX_SIZE,
        "worker_sleep": QUEUE_WORKER_SLEEP,
        "warning_threshold": QUEUE_WARNING_THRESHOLD,
        "critical_threshold": QUEUE_CRITICAL_THRESHOLD
    }


def validate_config(require_api_key: bool = False) -> bool:
    """
    Validate configuration values.

    Args:
        require_api_key: If True, require OPENAI_API_KEY to be set. Default False for demos.

    Returns:
        True if configuration is valid, False otherwise.
    """
    errors = []

    # API key is optional for demos/simulation mode
    if require_api_key and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")

    if RETRY_MAX_ATTEMPTS < 0:
        errors.append("RETRY_MAX_ATTEMPTS must be >= 0")

    if CB_FAILURE_THRESHOLD < 1:
        errors.append("CB_FAILURE_THRESHOLD must be >= 1")

    if QUEUE_MAX_SIZE < 1:
        errors.append("QUEUE_MAX_SIZE must be >= 1")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


if __name__ == "__main__":
    print("Current Configuration:")
    print(f"  Retry Max Attempts: {RETRY_MAX_ATTEMPTS}")
    print(f"  Circuit Breaker Threshold: {CB_FAILURE_THRESHOLD}")
    print(f"  Queue Max Size: {QUEUE_MAX_SIZE}")
    print(f"  OpenAI API Key: {'Set' if OPENAI_API_KEY else 'Not set'}")
    print(f"\nConfiguration valid: {validate_config()}")
