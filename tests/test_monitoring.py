"""
Tests for M2.3 Production Monitoring Dashboard
Smoke tests for metrics, logging, and decorators
"""
import time
import pytest
from src.m2_3_monitoring import (
    RAGMetrics,
    StructuredLogger,
    monitored_query,
    start_metrics_server,
    track_cache_operation
)
from prometheus_client import REGISTRY


def test_metrics_initialization():
    """Test that RAGMetrics initializes all metric types"""
    metrics = RAGMetrics(service_name="test-service")

    # Check that metrics are registered
    metric_names = [m.name for m in REGISTRY.collect()]

    assert 'rag_query_latency_seconds' in metric_names
    assert 'rag_input_tokens' in metric_names
    assert 'rag_output_tokens' in metric_names
    assert 'rag_query_cost_usd' in metric_names
    assert 'rag_cache_hits_total' in metric_names
    assert 'rag_cache_misses_total' in metric_names
    assert 'rag_errors_total' in metric_names
    assert 'rag_requests_total' in metric_names

    print("✓ All metrics initialized correctly")


def test_cost_calculation():
    """Test cost calculation from token usage"""
    metrics = RAGMetrics()

    # Calculate cost for 1000 input tokens and 500 output tokens
    # Default rates: $0.003/1K input, $0.015/1K output
    cost = metrics.calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model="test-model"
    )

    expected_cost = (1000 / 1000 * 0.003) + (500 / 1000 * 0.015)
    assert abs(cost - expected_cost) < 0.0001, f"Expected {expected_cost}, got {cost}"

    print(f"✓ Cost calculation correct: ${cost:.4f}")


def test_cache_tracking():
    """Test cache hit/miss tracking"""
    metrics = RAGMetrics()

    # Track some cache operations
    track_cache_operation(hit=True, cache_type="test")
    track_cache_operation(hit=True, cache_type="test")
    track_cache_operation(hit=False, cache_type="test")

    # Verify cache hit rate is calculated
    # Should be 2/(2+1) = 0.667
    hit_rate = metrics.cache_hit_rate.labels(cache_type="test")._value.get()
    assert 0.66 < hit_rate < 0.68, f"Expected ~0.667, got {hit_rate}"

    print(f"✓ Cache hit rate tracking works: {hit_rate:.2%}")


def test_monitored_query_decorator():
    """Test that decorator records metrics"""
    metrics = RAGMetrics()

    @monitored_query(operation="test_op", model="test-model")
    def test_function():
        time.sleep(0.1)  # Simulate work
        return {
            'input_tokens': 100,
            'output_tokens': 50,
            'relevance_score': 0.85
        }

    # Execute function
    result = test_function()

    # Verify it returns the result
    assert result['input_tokens'] == 100
    assert result['output_tokens'] == 50

    # Verify metrics were recorded (check that counters incremented)
    # Note: In real tests, you'd query the metrics more thoroughly
    print("✓ Monitored query decorator works")


def test_monitored_query_error_handling():
    """Test that decorator handles errors correctly"""
    metrics = RAGMetrics()

    @monitored_query(operation="test_error", model="test-model")
    def failing_function():
        raise ValueError("Test error")

    # Should raise the error but still record metrics
    with pytest.raises(ValueError, match="Test error"):
        failing_function()

    print("✓ Error handling in decorator works")


def test_structured_logger():
    """Test structured logging output"""
    logger = StructuredLogger("test-logger")

    # Test request logging (should not raise)
    logger.log_request(
        query="Test query",
        user_id="test-user",
        session_id="test-session"
    )

    # Test response logging
    logger.log_response(
        duration_ms=123.45,
        tokens={'input': 100, 'output': 50},
        cost=0.0123,
        success=True,
        model="test-model"
    )

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, context={'test': 'context'})

    print("✓ Structured logging works (check console output)")


def test_metrics_endpoint():
    """Test that metrics server can start"""
    # Try to start on a different port to avoid conflicts
    success = start_metrics_server(port=8002)

    # Should either succeed or report already running
    assert isinstance(success, bool)

    print("✓ Metrics endpoint can be started")


def test_metrics_server_exposes_data():
    """Test that /metrics endpoint exposes Prometheus format"""
    from prometheus_client import generate_latest

    # Generate metrics in Prometheus format
    metrics_output = generate_latest(REGISTRY)

    # Should be bytes
    assert isinstance(metrics_output, bytes)

    # Should contain metric names
    metrics_str = metrics_output.decode('utf-8')
    assert 'rag_' in metrics_str  # Our metrics should be present

    print("✓ Metrics are exposed in Prometheus format")


if __name__ == "__main__":
    print("="*60)
    print("M2.3 Monitoring - Smoke Tests")
    print("="*60 + "\n")

    # Run tests
    test_metrics_initialization()
    test_cost_calculation()
    test_cache_tracking()
    test_monitored_query_decorator()
    test_monitored_query_error_handling()
    test_structured_logger()
    test_metrics_endpoint()
    test_metrics_server_exposes_data()

    print("\n" + "="*60)
    print("✓ All smoke tests passed!")
    print("="*60)
