"""
M2.4 — Error Handling & Reliability for RAG Systems

PURPOSE
-------
This module teaches production-ready resilience patterns for RAG systems that depend on
external APIs (OpenAI, vector databases, etc.). You'll learn to handle transient failures
automatically, prevent cascading outages, and provide graceful degradation—reducing
user-facing errors by 80-95% in production.

Real-world impact: The difference between "service down" and "slightly slower but functional."


CONCEPTS COVERED
----------------
1. **Retry Strategy with Exponential Backoff**
   - Smart error classification: Retryable (5xx, 429) vs non-retryable (4xx)
   - Exponential delays prevent thundering herd
   - Jitter spreads retry timing to reduce server load collisions

2. **Circuit Breaker Pattern**
   - Three-state machine: CLOSED → OPEN → HALF_OPEN
   - Prevents retry storms during outages (fail fast vs retry forever)
   - Automatic recovery testing with configurable thresholds

3. **Graceful Degradation with Fallbacks**
   - Last-known-good (LKG) caching with age annotations
   - Generic helpful messages > stack traces
   - Stale data vs no data trade-offs

4. **Request Queueing & Backpressure**
   - Bounded queues prevent memory exhaustion
   - FIFO processing with rejection strategy
   - Traffic spike protection (reject some vs crash all)

5. **Honest Trade-offs**
   - When NOT to use each pattern
   - Complexity vs reliability costs
   - Latency overhead analysis
   - Production tuning guidance


AFTER COMPLETING THIS MODULE
----------------------------
You will be able to:
- ✓ Implement retry logic that distinguishes retryable vs non-retryable errors
- ✓ Add circuit breakers to prevent cascading failures in multi-service systems
- ✓ Design graceful fallback strategies for degraded-mode operation
- ✓ Handle traffic spikes with bounded queues and backpressure
- ✓ Tune resilience thresholds based on monitoring data
- ✓ Make informed decisions about when NOT to add complexity
- ✓ Reduce production error rates by 80-95% in RAG applications

Production ready: Copy patterns into your RAG system, tune thresholds, deploy.


CONTEXT IN TRACK
----------------
**Prerequisites (M1.x - M2.3):**
- M1.1-M1.4: Basic RAG architecture, embeddings, retrieval, generation
- M2.1-M2.3: Chunking strategies, vector search optimization, evaluation metrics

**This Module (M2.4):**
Making RAG systems **production-ready** with error handling and resilience patterns.
Focuses on external dependency failures (API outages, rate limits, network issues).

**Next Steps:**
- M3.x: Advanced RAG techniques (hybrid search, re-ranking, etc.)
- Production deployment with monitoring and alerting

**Why This Matters:**
Without resilience patterns:
- 2-5% base failure rate → 2-5% of users see errors
- API outages cascade into total system failures
- Traffic spikes crash your system
- User experience: unpredictable and frustrating

With resilience patterns:
- <0.1% error rate even with flaky dependencies
- Graceful degradation during outages
- Traffic spikes handled smoothly
- User experience: reliable and professional


QUICK START
-----------
>>> from src.m2_4_error_handling import RetryStrategy, CircuitBreaker, with_retry
>>>
>>> # Simple retry decorator
>>> @with_retry(max_retries=3, initial_delay=1.0)
>>> def fetch_embedding(text):
>>>     # Your OpenAI API call here
>>>     pass
>>>
>>> # Circuit breaker for protection
>>> cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
>>> result = cb.call(fetch_embedding, "query text")


MODULE STRUCTURE
----------------
- config.py: Tunable thresholds and environment configuration
- module.py: Core implementations (RetryStrategy, CircuitBreaker, etc.)
- router.py: FastAPI endpoints for demos and testing
- ../notebooks/: Interactive Jupyter notebook with examples
- ../tests/: Comprehensive test suite


TRADE-OFFS (HONEST REALITY CHECK)
----------------------------------
✅ Benefits:
- 80-95% error reduction in production
- Prevents cascading failures
- Better user experience during outages

⚠️ Costs:
- +20-30% code complexity
- +50-200ms latency per retry
- +10-20% infrastructure cost (retry traffic, queue memory)
- 8-12 hours implementation time

🎯 Use when:
- User-facing apps (10+ users)
- External API dependencies
- Uptime > cost

🚫 Don't use when:
- Internal tools (<10 users)
- Real-time systems (<50ms SLA)
- Batch processing (investigate failures, don't hide them)


For detailed documentation, examples, and production configs, see:
- README.md: Comprehensive guide with decision matrices
- notebooks/M2_4_Error_Handling_and_Reliability.ipynb: Interactive examples
- module.py: Inline docstrings with Google-style Args/Returns/Raises
"""

# Public API exports
from .module import (
    RetryStrategy,
    with_retry,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    GracefulFallbacks,
    RequestQueue,
    QueueWorker,
    ResilientOpenAIClient,
    CircuitProtectedRAG,
)

from .config import (
    get_retry_config,
    get_circuit_breaker_config,
    get_queue_config,
    validate_config,
)

__all__ = [
    # Retry patterns
    "RetryStrategy",
    "with_retry",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    # Graceful degradation
    "GracefulFallbacks",
    # Queueing
    "RequestQueue",
    "QueueWorker",
    # Integrated clients
    "ResilientOpenAIClient",
    "CircuitProtectedRAG",
    # Config helpers
    "get_retry_config",
    "get_circuit_breaker_config",
    "get_queue_config",
    "validate_config",
]

__version__ = "1.0.0"
__module_name__ = "M2.4 — Error Handling & Reliability"
