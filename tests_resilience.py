"""
Tests for M2.4 Error Handling & Reliability

Smoke tests to verify all resilience patterns work correctly.
Run with: pytest tests_resilience.py -v
"""

import pytest
import time
import random
from m2_4_resilience import (
    RetryStrategy, with_retry,
    CircuitBreaker, CircuitState, CircuitBreakerOpenError,
    GracefulFallbacks,
    RequestQueue, QueueWorker
)


# ==================== RETRY STRATEGY TESTS ====================

def test_retry_succeeds_on_flaky_function():
    """Test that retry strategy eventually succeeds on flaky function."""
    call_count = [0]

    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("Simulated failure")
        return "Success"

    strategy = RetryStrategy(max_retries=3, initial_delay=0.1)
    result = strategy.execute(flaky_function)

    assert result == "Success"
    assert call_count[0] == 3  # Failed twice, succeeded on 3rd attempt


def test_retry_fails_after_max_retries():
    """Test that retry strategy fails after max retries."""
    def always_fails():
        raise ConnectionError("Always fails")

    strategy = RetryStrategy(max_retries=2, initial_delay=0.1)

    with pytest.raises(ConnectionError):
        strategy.execute(always_fails)


def test_retry_decorator():
    """Test @with_retry decorator."""
    call_count = [0]

    @with_retry(max_retries=3, initial_delay=0.1)
    def flaky_decorated():
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionError("Fail once")
        return "Decorated success"

    result = flaky_decorated()
    assert result == "Decorated success"


def test_retry_non_retryable_error():
    """Test that non-retryable errors fail immediately."""
    class APIError(Exception):
        def __init__(self):
            super().__init__("Not found")
            self.status_code = 404

    call_count = [0]

    def non_retryable():
        call_count[0] += 1
        raise APIError()

    strategy = RetryStrategy(max_retries=3, initial_delay=0.1)

    with pytest.raises(APIError):
        strategy.execute(non_retryable)

    assert call_count[0] == 1  # Should not retry 404


def test_retry_retryable_error():
    """Test that retryable errors are retried."""
    class APIError(Exception):
        def __init__(self):
            super().__init__("Server error")
            self.status_code = 500

    call_count = [0]

    def retryable():
        call_count[0] += 1
        if call_count[0] < 2:
            raise APIError()
        return "Recovered"

    strategy = RetryStrategy(max_retries=3, initial_delay=0.1)
    result = strategy.execute(retryable)

    assert result == "Recovered"
    assert call_count[0] == 2  # Failed once, succeeded on 2nd


# ==================== CIRCUIT BREAKER TESTS ====================

def test_circuit_breaker_opens_after_failures():
    """Test that circuit breaker opens after failure threshold."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    def failing_service():
        raise ConnectionError("Service down")

    # Cause failures to open circuit
    for _ in range(3):
        with pytest.raises(ConnectionError):
            cb.call(failing_service)

    # Circuit should now be open
    assert cb.get_state() == CircuitState.OPEN


def test_circuit_breaker_rejects_when_open():
    """Test that circuit breaker rejects requests when open."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)

    def failing():
        raise ConnectionError("Fail")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(ConnectionError):
            cb.call(failing)

    # Should reject next call immediately
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(lambda: "Should not execute")


def test_circuit_breaker_half_open_recovery():
    """Test circuit breaker transitions to half-open and recovers."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

    service_is_down = [True]

    def service():
        if service_is_down[0]:
            raise ConnectionError("Down")
        return "OK"

    # Open circuit
    for _ in range(2):
        with pytest.raises(ConnectionError):
            cb.call(service)

    assert cb.get_state() == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.6)

    # Service recovers
    service_is_down[0] = False

    # Should transition to HALF_OPEN and succeed
    result = cb.call(service)
    assert result == "OK"
    assert cb.get_state() == CircuitState.CLOSED


def test_circuit_breaker_resets_on_success():
    """Test circuit breaker resets failure count on success."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    call_count = [0]

    def intermittent():
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return "Success"
        raise ConnectionError("Fail")

    # Fail, succeed, fail, succeed - should never open
    for _ in range(4):
        try:
            cb.call(intermittent)
        except:
            pass

    assert cb.get_state() == CircuitState.CLOSED  # Should still be closed


# ==================== GRACEFUL FALLBACKS TESTS ====================

def test_fallback_cache():
    """Test fallback cache stores and retrieves values."""
    fallbacks = GracefulFallbacks()

    key = "test_query"
    value = "test_answer"

    fallbacks.update_cache(key, value)
    result = fallbacks.get_cached_or_fallback(key, "default")

    assert result == value


def test_fallback_default_message():
    """Test fallback returns default when no cache."""
    fallbacks = GracefulFallbacks()

    default = "Service unavailable"
    result = fallbacks.get_cached_or_fallback("unknown_key", default)

    assert result == default


def test_fallback_last_known_good():
    """Test last-known-good pattern."""
    fallbacks = GracefulFallbacks()

    key = "query"
    value = "answer"

    fallbacks.update_cache(key, value)
    time.sleep(0.1)

    result = fallbacks.get_last_known_good(key)
    assert result is not None

    cached_value, age = result
    assert cached_value == value
    assert age >= 0.1


def test_fallback_generic_answer():
    """Test generic fallback message generation."""
    fallbacks = GracefulFallbacks()

    query = "What is machine learning?"
    answer = fallbacks.get_generic_answer(query)

    assert "technical difficulties" in answer.lower()
    assert len(answer) > 0


# ==================== REQUEST QUEUE TESTS ====================

def test_queue_enqueue_dequeue():
    """Test basic queue operations."""
    queue = RequestQueue(max_size=10)

    assert queue.enqueue("item1")
    assert queue.enqueue("item2")
    assert queue.size() == 2

    item = queue.dequeue()
    assert item == "item1"
    assert queue.size() == 1


def test_queue_backpressure():
    """Test queue rejects when full."""
    queue = RequestQueue(max_size=3)

    # Fill queue
    assert queue.enqueue("1")
    assert queue.enqueue("2")
    assert queue.enqueue("3")

    # Should reject when full
    assert not queue.enqueue("4")
    assert queue.size() == 3


def test_queue_stats():
    """Test queue statistics."""
    queue = RequestQueue(max_size=5)

    queue.enqueue("1")
    queue.enqueue("2")
    queue.dequeue()

    stats = queue.stats()

    assert stats["current_size"] == 1
    assert stats["processed"] == 1
    assert stats["capacity"] == 5


def test_queue_worker():
    """Test queue worker processes items."""
    queue = RequestQueue(max_size=10)
    processed = []

    def process(item):
        processed.append(item)

    worker = QueueWorker(queue, process)

    # Add items
    queue.enqueue("task1")
    queue.enqueue("task2")
    queue.enqueue("task3")

    # Start worker
    worker.start()
    time.sleep(0.5)  # Let it process
    worker.stop()

    assert len(processed) == 3
    assert "task1" in processed
    assert "task2" in processed
    assert "task3" in processed


# ==================== INTEGRATION TESTS ====================

def test_retry_with_circuit_breaker():
    """Test retry strategy combined with circuit breaker."""
    retry = RetryStrategy(max_retries=2, initial_delay=0.1)
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    call_count = [0]

    def flaky():
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionError("Fail")
        return "Success"

    # Should succeed with retry
    result = retry.execute(lambda: cb.call(flaky))
    assert result == "Success"


def test_fallback_with_circuit_breaker():
    """Test fallback combined with circuit breaker."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    fallbacks = GracefulFallbacks()

    def service(should_fail=False):
        if should_fail:
            raise ConnectionError("Down")
        return "Live answer"

    # Normal operation - cache answer
    answer = cb.call(lambda: service(False))
    fallbacks.update_cache("query", answer)

    # Service goes down
    for _ in range(2):
        try:
            cb.call(lambda: service(True))
        except:
            pass

    # Use fallback
    cached = fallbacks.get_cached_or_fallback("query", "default")
    assert cached == "Live answer"


# ==================== SMOKE TESTS ====================

def test_smoke_all_patterns():
    """Smoke test: Verify all patterns can be instantiated."""
    retry = RetryStrategy()
    cb = CircuitBreaker()
    fallbacks = GracefulFallbacks()
    queue = RequestQueue()

    assert retry is not None
    assert cb is not None
    assert fallbacks is not None
    assert queue is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
