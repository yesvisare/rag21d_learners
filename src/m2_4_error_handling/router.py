"""
FastAPI router for M2.4 Error Handling & Reliability demo endpoints.

Provides simulation endpoints that don't require external API keys.
Demonstrates resilience patterns in action without real service dependencies.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import random

from .module import (
    RetryStrategy,
    CircuitBreaker,
    CircuitState,
    GracefulFallbacks,
    RequestQueue,
    CircuitBreakerOpenError,
)
from .config import get_circuit_breaker_config, get_queue_config

# Initialize router
router = APIRouter(prefix="/m2_4", tags=["M2.4 Error Handling"])

# Module-level instances for stateful demos
_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0
)
_request_queue = RequestQueue(max_size=1000)
_fallbacks = GracefulFallbacks()


# ==================== MODELS ====================

class SimulateRequest(BaseModel):
    """Request model for /simulate endpoint."""
    n: Optional[int] = Field(default=10, ge=1, le=100, description="Number of requests to simulate")
    failure_rate: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="Simulated failure rate (0.0-1.0)")
    operation: Optional[str] = Field(default="all", description="Pattern to test: 'retry', 'circuit_breaker', 'fallback', 'queue', or 'all'")


class SimulateResponse(BaseModel):
    """Response model for /simulate endpoint."""
    operation: str
    total_requests: int
    successful: int
    failed: int
    retried: int
    circuit_state: str
    queue_stats: dict
    fallback_hits: int
    summary: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    module: str
    version: str


class CircuitStateResponse(BaseModel):
    """Circuit breaker state response."""
    state: str
    failure_count: int
    description: str


class QueueStatsResponse(BaseModel):
    """Queue statistics response."""
    current_size: int
    processed: int
    rejected: int
    capacity: int
    utilization_pct: float


# ==================== ENDPOINTS ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns basic module information and status.
    """
    return {
        "status": "ok",
        "module": "m2_4_error_handling",
        "version": "1.0.0"
    }


@router.get("/state", response_model=CircuitStateResponse)
async def get_circuit_state():
    """
    Get current circuit breaker state.

    Returns:
        Circuit breaker state (CLOSED, OPEN, or HALF_OPEN) with explanation.
    """
    state = _circuit_breaker.get_state()

    descriptions = {
        CircuitState.CLOSED: "Normal operation - accepting requests and tracking failures",
        CircuitState.OPEN: "Circuit open - rejecting requests due to failure threshold exceeded",
        CircuitState.HALF_OPEN: "Testing recovery - allowing one request to check service health"
    }

    return {
        "state": state.value,
        "failure_count": _circuit_breaker.failure_count,
        "description": descriptions[state]
    }


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """
    Get request queue statistics.

    Returns:
        Current queue state including size, processed count, rejected count, and utilization.
    """
    stats = _request_queue.stats()

    # Calculate utilization percentage
    utilization = (stats["current_size"] / stats["capacity"]) * 100 if stats["capacity"] > 0 else 0

    return {
        "current_size": stats["current_size"],
        "processed": stats["processed"],
        "rejected": stats["rejected"],
        "capacity": stats["capacity"],
        "utilization_pct": round(utilization, 2)
    }


@router.post("/simulate", response_model=SimulateResponse)
async def simulate_resilience_patterns(request: SimulateRequest):
    """
    Simulate resilience patterns without external dependencies.

    Demonstrates retry logic, circuit breaker, fallbacks, and queueing in action.
    No API keys required - uses simulated failures for demonstration.

    Args:
        request: Simulation parameters (n requests, failure rate, operation type)

    Returns:
        Summary of simulation results including success/failure counts and pattern behavior.
    """
    def simulate_flaky_service(failure_rate: float):
        """Simulate a service that fails randomly."""
        if random.random() < failure_rate:
            raise ConnectionError("Simulated service failure")
        return "Success"

    # Initialize counters
    total = request.n
    successful = 0
    failed = 0
    retried = 0
    fallback_hits = 0

    # Create instances for this simulation
    retry_strategy = RetryStrategy(max_retries=3, initial_delay=0.1)
    local_fallbacks = GracefulFallbacks()

    # Simulate requests based on operation type
    if request.operation in ["retry", "all"]:
        # Test retry pattern
        for i in range(total):
            attempt_count = [0]

            def tracked_call():
                attempt_count[0] += 1
                return simulate_flaky_service(request.failure_rate)

            try:
                retry_strategy.execute(tracked_call)
                successful += 1
                if attempt_count[0] > 1:
                    retried += 1
            except Exception:
                failed += 1

    elif request.operation == "circuit_breaker":
        # Test circuit breaker pattern
        for i in range(total):
            try:
                _circuit_breaker.call(simulate_flaky_service, request.failure_rate)
                successful += 1
            except (ConnectionError, CircuitBreakerOpenError):
                failed += 1

    elif request.operation == "fallback":
        # Test fallback pattern
        for i in range(total):
            query = f"query_{i}"
            try:
                result = simulate_flaky_service(request.failure_rate)
                local_fallbacks.update_cache(query, result)
                successful += 1
            except Exception:
                # Use fallback
                cached = local_fallbacks.get_cached_or_fallback(
                    query,
                    "Generic fallback response"
                )
                if cached != "Generic fallback response":
                    fallback_hits += 1
                successful += 1  # Fallback counts as success (degraded mode)

    elif request.operation == "queue":
        # Test queue pattern
        for i in range(total):
            if _request_queue.enqueue(f"request_{i}"):
                successful += 1
            else:
                failed += 1  # Rejected due to backpressure

    # Get current states
    circuit_state = _circuit_breaker.get_state().value
    queue_stats = _request_queue.stats()

    # Generate summary
    success_rate = (successful / total * 100) if total > 0 else 0
    summary = (
        f"Simulated {total} requests with {request.failure_rate*100:.0f}% failure rate. "
        f"Success rate: {success_rate:.1f}%. "
        f"Circuit: {circuit_state}. "
    )

    if retried > 0:
        summary += f"Retried {retried} requests successfully. "
    if fallback_hits > 0:
        summary += f"Used fallbacks {fallback_hits} times. "

    return {
        "operation": request.operation,
        "total_requests": total,
        "successful": successful,
        "failed": failed,
        "retried": retried,
        "circuit_state": circuit_state,
        "queue_stats": queue_stats,
        "fallback_hits": fallback_hits,
        "summary": summary
    }


@router.post("/reset")
async def reset_state():
    """
    Reset circuit breaker and queue state for fresh demos.

    Useful for running multiple simulation scenarios.
    """
    global _circuit_breaker, _request_queue, _fallbacks

    _circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    _request_queue = RequestQueue(max_size=1000)
    _fallbacks = GracefulFallbacks()

    return {
        "message": "State reset successfully",
        "circuit_state": _circuit_breaker.get_state().value,
        "queue_size": _request_queue.size()
    }
