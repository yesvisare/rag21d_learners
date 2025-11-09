"""
M2.3 Production Monitoring Dashboard
Prometheus metrics, structured logging, and monitoring decorators for RAG systems
"""
import time
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
from . import config


class RAGMetrics:
    """
    Comprehensive metrics collection for RAG pipeline monitoring.
    Tracks performance, costs, quality, and system health.
    """

    def __init__(self, service_name: str = None):
        self.service_name = service_name or config.SERVICE_NAME

        # === PERFORMANCE METRICS ===

        # Query latency histogram (in seconds) with percentile buckets
        self.query_latency = Histogram(
            'rag_query_latency_seconds',
            'End-to-end RAG query latency',
            ['operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        # Individual component latencies
        self.retrieval_latency = Histogram(
            'rag_retrieval_latency_seconds',
            'Document retrieval latency',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )

        self.llm_latency = Histogram(
            'rag_llm_latency_seconds',
            'LLM generation latency',
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        )

        # === TOKEN USAGE METRICS ===

        self.input_tokens = Histogram(
            'rag_input_tokens',
            'Input tokens per request',
            ['model'],
            buckets=[100, 500, 1000, 2000, 4000, 8000, 16000]
        )

        self.output_tokens = Histogram(
            'rag_output_tokens',
            'Output tokens per request',
            ['model'],
            buckets=[50, 100, 500, 1000, 2000, 4000]
        )

        # === COST METRICS ===

        self.query_cost = Histogram(
            'rag_query_cost_usd',
            'Cost per query in USD',
            ['model'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self.total_cost = Counter(
            'rag_total_cost_usd',
            'Cumulative cost in USD',
            ['model']
        )

        # === CACHE METRICS ===

        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Cache hit count',
            ['cache_type']
        )

        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Cache miss count',
            ['cache_type']
        )

        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate (0-1)',
            ['cache_type']
        )

        # === QUALITY METRICS ===

        self.relevance_score = Histogram(
            'rag_relevance_score',
            'Response relevance score (0-1)',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        )

        self.context_precision = Histogram(
            'rag_context_precision',
            'Context retrieval precision',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        )

        # === ERROR & REQUEST METRICS ===

        self.requests_total = Counter(
            'rag_requests_total',
            'Total requests',
            ['endpoint', 'status']
        )

        self.errors_total = Counter(
            'rag_errors_total',
            'Total errors',
            ['error_type', 'endpoint']
        )

        # === RATE LIMIT METRICS ===

        self.rate_limit_remaining = Gauge(
            'rag_rate_limit_remaining',
            'Remaining API calls before rate limit',
            ['provider']
        )

        self.rate_limit_hits = Counter(
            'rag_rate_limit_hits_total',
            'Number of times rate limit was hit',
            ['provider']
        )

        # === SYSTEM METRICS ===

        self.active_requests = Gauge(
            'rag_active_requests',
            'Currently active requests'
        )

        self.queue_depth = Gauge(
            'rag_queue_depth',
            'Request queue depth'
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str = "default") -> float:
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1000) * config.COST_PER_1K_INPUT_TOKENS
        output_cost = (output_tokens / 1000) * config.COST_PER_1K_OUTPUT_TOKENS
        total = input_cost + output_cost

        # Record metrics
        self.query_cost.labels(model=model).observe(total)
        self.total_cost.labels(model=model).inc(total)

        return total

    def update_cache_hit_rate(self, cache_type: str = "default"):
        """Calculate and update cache hit rate"""
        hits = self.cache_hits.labels(cache_type=cache_type)._value.get()
        misses = self.cache_misses.labels(cache_type=cache_type)._value.get()

        total = hits + misses
        if total > 0:
            hit_rate = hits / total
            self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)


# Global metrics instance
metrics = RAGMetrics()


class StructuredLogger:
    """
    JSON structured logging for production environments.
    Emits logs compatible with cloud logging services (CloudWatch, Stackdriver, etc.)
    """

    def __init__(self, name: str, level: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level or config.LOG_LEVEL)

        # Clear existing handlers
        self.logger.handlers = []

        # Add console handler with JSON formatting
        handler = logging.StreamHandler()
        handler.setFormatter(self._json_formatter())
        self.logger.addHandler(handler)

    def _json_formatter(self):
        """Create a JSON formatter"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'service': config.SERVICE_NAME,
                    'environment': config.ENVIRONMENT,
                }

                # Add extra fields if present
                if hasattr(record, 'extra_fields'):
                    log_obj.update(record.extra_fields)

                # Add exception info if present
                if record.exc_info:
                    log_obj['exception'] = self.formatException(record.exc_info)

                return json.dumps(log_obj)

        return JSONFormatter()

    def log_request(self, query: str, user_id: Optional[str] = None, **kwargs):
        """Log incoming request"""
        extra = {
            'event_type': 'request',
            'query_length': len(query),
            'user_id': user_id,
            **kwargs
        }
        self.logger.info(f"Request received", extra={'extra_fields': extra})

    def log_response(self, duration_ms: float, tokens: Dict[str, int],
                    cost: float, success: bool = True, **kwargs):
        """Log query response"""
        extra = {
            'event_type': 'response',
            'duration_ms': duration_ms,
            'input_tokens': tokens.get('input', 0),
            'output_tokens': tokens.get('output', 0),
            'cost_usd': cost,
            'success': success,
            **kwargs
        }

        if success:
            self.logger.info(f"Response completed", extra={'extra_fields': extra})
        else:
            self.logger.error(f"Response failed", extra={'extra_fields': extra})

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context"""
        extra = {
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            **context
        }
        self.logger.error(f"Error occurred: {error}",
                         extra={'extra_fields': extra},
                         exc_info=True)


# Global logger instance
logger = StructuredLogger(__name__)


def monitored_query(operation: str = "query", model: str = "default"):
    """
    Decorator to automatically instrument RAG queries with monitoring.

    Usage:
        @monitored_query(operation="search", model="gpt-4")
        def my_rag_function(query: str):
            # Your RAG logic here
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Increment active requests
            metrics.active_requests.inc()

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Record success metrics
                duration = time.time() - start_time
                metrics.query_latency.labels(operation=operation).observe(duration)
                metrics.requests_total.labels(endpoint=operation, status="success").inc()

                # If result contains metrics, record them
                if isinstance(result, dict):
                    if 'input_tokens' in result and 'output_tokens' in result:
                        metrics.input_tokens.labels(model=model).observe(result['input_tokens'])
                        metrics.output_tokens.labels(model=model).observe(result['output_tokens'])

                        # Calculate cost
                        cost = metrics.calculate_cost(
                            result['input_tokens'],
                            result['output_tokens'],
                            model
                        )

                    if 'relevance_score' in result:
                        metrics.relevance_score.observe(result['relevance_score'])

                return result

            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                metrics.query_latency.labels(operation=operation).observe(duration)
                metrics.requests_total.labels(endpoint=operation, status="error").inc()
                metrics.errors_total.labels(
                    error_type=type(e).__name__,
                    endpoint=operation
                ).inc()

                # Log error
                logger.log_error(e, {'operation': operation, 'duration': duration})

                raise

            finally:
                # Decrement active requests
                metrics.active_requests.dec()

        return wrapper
    return decorator


def start_metrics_server(port: int = None):
    """
    Start Prometheus metrics HTTP server.
    Exposes metrics at http://localhost:{port}/metrics

    Args:
        port: Port to bind to (default from config.METRICS_PORT)
    """
    port = port or config.METRICS_PORT
    try:
        start_http_server(port)
        print(f"✓ Metrics server started on port {port}")
        print(f"  View metrics at: http://localhost:{port}/metrics")
        return True
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"✓ Metrics server already running on port {port}")
            return True
        else:
            print(f"✗ Failed to start metrics server: {e}")
            return False


# Example integration hooks
def track_cache_operation(hit: bool, cache_type: str = "default"):
    """Helper to track cache hits/misses"""
    if hit:
        metrics.cache_hits.labels(cache_type=cache_type).inc()
    else:
        metrics.cache_misses.labels(cache_type=cache_type).inc()

    metrics.update_cache_hit_rate(cache_type)


def track_rate_limit(remaining: int, provider: str = "openai"):
    """Helper to track API rate limits"""
    metrics.rate_limit_remaining.labels(provider=provider).set(remaining)


def track_rate_limit_hit(provider: str = "openai"):
    """Helper to track when rate limit is hit"""
    metrics.rate_limit_hits.labels(provider=provider).inc()


if __name__ == "__main__":
    # Demo: Start metrics server
    start_metrics_server()

    print("\n" + "="*60)
    print("M2.3 Production Monitoring Dashboard - Demo")
    print("="*60)

    # Simulate some metrics
    @monitored_query(operation="demo_query", model="gpt-4")
    def demo_query(query: str):
        """Simulated RAG query"""
        time.sleep(0.1)  # Simulate processing
        return {
            'input_tokens': 500,
            'output_tokens': 150,
            'relevance_score': 0.85,
            'answer': 'Demo answer'
        }

    # Run demo queries
    print("\nSimulating queries...")
    for i in range(5):
        result = demo_query(f"Demo query {i+1}")
        track_cache_operation(hit=(i % 2 == 0), cache_type="semantic")

    print("\n✓ Demo metrics recorded")
    print(f"  Check metrics at: http://localhost:{config.METRICS_PORT}/metrics")
    print("\nPress Ctrl+C to stop...")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown complete")
