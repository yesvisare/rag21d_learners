"""
Configuration module for M2.1 Caching Strategies.

Handles environment variables, Redis and OpenAI client setup with graceful
fallbacks when services are unavailable.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Cache layer toggles
ENABLE_EXACT_CACHE = os.getenv("ENABLE_EXACT_CACHE", "true").lower() == "true"
ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
ENABLE_CONTEXT_CACHE = os.getenv("ENABLE_CONTEXT_CACHE", "true").lower() == "true"

# TTL settings (in seconds)
TTL_EXACT_CACHE = int(os.getenv("TTL_EXACT_CACHE", "3600"))  # 1 hour
TTL_SEMANTIC_CACHE = int(os.getenv("TTL_SEMANTIC_CACHE", "1800"))  # 30 minutes
TTL_EMBEDDING_CACHE = int(os.getenv("TTL_EMBEDDING_CACHE", "7200"))  # 2 hours
TTL_CONTEXT_CACHE = int(os.getenv("TTL_CONTEXT_CACHE", "1800"))  # 30 minutes

# Semantic similarity threshold
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))

# Cache key prefixes
PREFIX_EXACT = "exact:"
PREFIX_SEMANTIC = "semantic:"
PREFIX_EMBEDDING = "embed:"
PREFIX_CONTEXT = "context:"

# Redis connection pool settings
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))


def get_redis():
    """
    Get Redis client with connection pooling.

    Returns:
        redis.Redis | None: Configured Redis client or None if unavailable.

    Rationale:
        Returns None instead of raising to allow graceful degradation in
        environments without Redis (e.g., CI, local development).
    """
    try:
        import redis
        client = redis.from_url(
            REDIS_URL,
            max_connections=REDIS_MAX_CONNECTIONS,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            decode_responses=True
        )
        # Test connection
        client.ping()
        return client
    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        return None


def get_openai():
    """
    Get OpenAI client.

    Returns:
        OpenAI | None: Configured OpenAI client or None if unavailable.

    Rationale:
        Returns None instead of raising to enable demo mode without API keys.
        Checks for placeholder key pattern (sk-...) from .env.example.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
        print("⚠️ OpenAI API key not configured")
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ OpenAI client initialization failed: {e}")
        return None


def has_services() -> bool:
    """
    Check if both Redis and OpenAI are available.

    Returns:
        bool: True if both services are available, False otherwise.
    """
    return get_redis() is not None and get_openai() is not None
