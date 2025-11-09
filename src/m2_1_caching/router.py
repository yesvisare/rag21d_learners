"""
FastAPI router for M2.1 Caching Strategies API.

Provides HTTP endpoints for cache health checks, metrics inspection,
and manual invalidation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from . import config
from .module import MultiLayerCache

# Initialize router
router = APIRouter(
    prefix="/m2_1_caching",
    tags=["caching"],
    responses={404: {"description": "Not found"}},
)

# Global cache instance (initialized on first request)
_cache_instance: Optional[MultiLayerCache] = None


def get_cache() -> MultiLayerCache:
    """
    Get or create global cache instance.

    Returns:
        MultiLayerCache instance.
    """
    global _cache_instance
    if _cache_instance is None:
        redis_client = config.get_redis()
        openai_client = config.get_openai()
        _cache_instance = MultiLayerCache(redis_client, openai_client)
    return _cache_instance


# Request/Response models
class InvalidateRequest(BaseModel):
    """Request body for cache invalidation."""
    prefix: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    module: str
    redis_available: bool
    openai_available: bool


class MetricsResponse(BaseModel):
    """Cache metrics response."""
    hits: int
    misses: int
    hit_rate: float
    stampede_prevented: int
    invalidations: int
    errors: int
    summary: str


class InvalidateResponse(BaseModel):
    """Invalidation response."""
    prefix: str
    keys_invalidated: int
    message: str


# Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status and module identification.
    """
    redis_client = config.get_redis()
    openai_client = config.get_openai()

    return HealthResponse(
        status="ok",
        module="m2_1_caching",
        redis_available=redis_client is not None,
        openai_available=openai_client is not None
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get cache performance metrics.

    Returns:
        Current cache metrics including hits, misses, and hit rate.
    """
    cache = get_cache()
    metrics = cache.metrics

    return MetricsResponse(
        hits=metrics.hits,
        misses=metrics.misses,
        hit_rate=metrics.get_hit_rate(),
        stampede_prevented=metrics.stampede_prevented,
        invalidations=metrics.invalidations,
        errors=metrics.errors,
        summary=metrics.summary()
    )


@router.post("/invalidate", response_model=InvalidateResponse)
async def invalidate_cache(request: InvalidateRequest):
    """
    Invalidate cache entries by prefix.

    Args:
        request: Invalidation request with prefix.

    Returns:
        Number of keys invalidated.

    Raises:
        HTTPException: If Redis is not available.
    """
    cache = get_cache()

    if not cache.redis:
        raise HTTPException(
            status_code=503,
            detail="Redis not available - cannot invalidate cache"
        )

    # Validate prefix
    valid_prefixes = [
        config.PREFIX_EXACT,
        config.PREFIX_SEMANTIC,
        config.PREFIX_EMBEDDING,
        config.PREFIX_CONTEXT
    ]

    if request.prefix not in valid_prefixes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prefix. Must be one of: {', '.join(valid_prefixes)}"
        )

    # Invalidate
    count = cache.invalidate_by_prefix(request.prefix)

    return InvalidateResponse(
        prefix=request.prefix,
        keys_invalidated=count,
        message=f"Invalidated {count} keys with prefix '{request.prefix}'"
    )


@router.get("/")
async def root():
    """
    API root endpoint.

    Returns:
        Module information and available endpoints.
    """
    return {
        "module": "M2.1 Caching Strategies",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /metrics": "Cache performance metrics",
            "POST /invalidate": "Invalidate cache by prefix"
        },
        "documentation": "/docs"
    }
