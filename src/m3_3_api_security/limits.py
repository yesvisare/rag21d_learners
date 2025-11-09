"""
Rate Limiting using Token Bucket Algorithm
Supports in-memory limiter with optional Redis backend stub.
"""
import time
from typing import Dict, Tuple
from fastapi import HTTPException, status


class TokenBucketLimiter:
    """
    Token bucket rate limiter with burst protection.

    Tracks both per-minute and per-hour limits:
    - Burst: 10 tokens
    - Per-minute: 60 requests (configurable)
    - Per-hour: 1000 requests (configurable)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.burst_size = burst_size

        # Refill rate: tokens per second
        self.refill_rate = requests_per_minute / 60.0

        # Storage: {key_hash: {"tokens": float, "last_refill": timestamp, "hour_count": int, "hour_start": timestamp}}
        self.buckets: Dict[str, Dict] = {}

    def check_rate_limit(self, key_identifier: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limits.

        Returns:
            (allowed: bool, info: dict)
        """
        now = time.time()

        # Initialize bucket if needed
        if key_identifier not in self.buckets:
            self.buckets[key_identifier] = {
                "tokens": self.burst_size,
                "last_refill": now,
                "hour_count": 0,
                "hour_start": now
            }

        bucket = self.buckets[key_identifier]

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + (elapsed * self.refill_rate)
        )
        bucket["last_refill"] = now

        # Reset hourly counter if needed
        if now - bucket["hour_start"] >= 3600:
            bucket["hour_count"] = 0
            bucket["hour_start"] = now

        # Check limits
        if bucket["tokens"] < 1:
            return False, {
                "limit_type": "burst",
                "retry_after": int((1 - bucket["tokens"]) / self.refill_rate),
                "tokens_remaining": 0
            }

        if bucket["hour_count"] >= self.rph:
            seconds_until_reset = 3600 - (now - bucket["hour_start"])
            return False, {
                "limit_type": "hourly",
                "retry_after": int(seconds_until_reset),
                "tokens_remaining": 0
            }

        # Consume token
        bucket["tokens"] -= 1
        bucket["hour_count"] += 1

        return True, {
            "limit_type": None,
            "retry_after": 0,
            "tokens_remaining": int(bucket["tokens"]),
            "hour_remaining": self.rph - bucket["hour_count"]
        }


# Global limiter instance
rate_limiter = TokenBucketLimiter()


async def check_rate_limit_dependency(x_api_key: str):
    """FastAPI dependency to enforce rate limits."""
    allowed, info = rate_limiter.check_rate_limit(x_api_key)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({info['limit_type']}). Retry after {info['retry_after']} seconds.",
            headers={
                "Retry-After": str(info["retry_after"]),
                "X-RateLimit-Limit": str(rate_limiter.rpm),
                "X-RateLimit-Remaining": "0"
            }
        )

    return info


# Redis-based rate limiter (stub for production use)
class RedisRateLimiter:
    """
    Redis-backed rate limiter for distributed systems.

    NOTE: This is a stub. In production:
    1. pip install redis
    2. Connect to Redis: redis.Redis(host='localhost', port=6379, db=0)
    3. Use INCR with EXPIRE for atomic counters
    4. Use Lua scripts for token bucket atomicity
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        print(f"[STUB] RedisRateLimiter would connect to {redis_url}")
        print("[STUB] Falling back to in-memory limiter")
        self.fallback = TokenBucketLimiter()

    def check_rate_limit(self, key_identifier: str) -> Tuple[bool, Dict]:
        """Stub: delegates to in-memory limiter."""
        return self.fallback.check_rate_limit(key_identifier)
