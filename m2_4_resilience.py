"""
M2.4 — Error Handling & Reliability
Production-ready resilience patterns for RAG systems with external dependencies.
"""

import time
import random
import threading
from enum import Enum
from functools import wraps
from typing import Callable, Any, Optional, List
from datetime import datetime, timedelta
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== RETRY STRATEGY ====================

class RetryStrategy:
    """
    Implements exponential backoff with jitter for handling transient failures.

    Key Features:
    - Distinguishes retryable (5xx, 429) vs non-retryable errors (4xx)
    - Exponential backoff prevents thundering herd
    - Configurable jitter adds randomness to retry timing
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (ConnectionError, TimeoutError)
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        if self.jitter:
            # Add ±25% jitter to prevent synchronized retries
            delay *= (0.75 + 0.5 * random.random())
        return delay

    def is_retryable(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Check for retryable HTTP status codes if available
        if hasattr(exception, 'status_code'):
            status = exception.status_code
            # Retry on 5xx (server errors) and 429 (rate limit)
            if status >= 500 or status == 429:
                return True
            # Don't retry 4xx (client errors, except 429)
            if 400 <= status < 500:
                return False

        # Check exception type
        return isinstance(exception, self.retryable_exceptions)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✓ Retry succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.is_retryable(e):
                    logger.error(f"✗ Non-retryable error: {type(e).__name__}: {e}")
                    raise

                # Check if we have retries left
                if attempt >= self.max_retries:
                    logger.error(f"✗ Max retries ({self.max_retries}) exceeded")
                    raise

                # Calculate delay and retry
                delay = self.calculate_delay(attempt)
                logger.warning(f"⚠ Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)

        # Should never reach here, but just in case
        raise last_exception


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        @with_retry(max_retries=3, initial_delay=1.0)
        def flaky_api_call():
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            strategy = RetryStrategy(
                max_retries=max_retries,
                initial_delay=initial_delay,
                exponential_base=exponential_base,
                jitter=jitter
            )
            return strategy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# ==================== CIRCUIT BREAKER ====================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Implements circuit breaker pattern to prevent cascading failures.

    State Machine:
    - CLOSED: Normal operation, tracks failures
    - OPEN: After failure_threshold reached, rejects all requests
    - HALF_OPEN: After recovery_timeout, allows test request

    Thread-safe implementation for concurrent environments.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

        # Thread safety
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info("🔄 Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                    self.state = CircuitState.HALF_OPEN
                else:
                    time_remaining = self.recovery_timeout - (datetime.now() - self.last_failure_time).total_seconds()
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Retry in {time_remaining:.1f}s"
                    )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("✓ Circuit breaker CLOSED (recovery successful)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery test - go back to OPEN
                logger.error("✗ Circuit breaker reopened (recovery test failed)")
                self.state = CircuitState.OPEN

            elif self.failure_count >= self.failure_threshold:
                # Threshold exceeded - open the circuit
                logger.error(f"✗ Circuit breaker OPEN (threshold {self.failure_threshold} exceeded)")
                self.state = CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state (thread-safe)."""
        with self._lock:
            return self.state


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""
    pass


# ==================== GRACEFUL DEGRADATION ====================

class GracefulFallbacks:
    """
    Provides fallback responses when services are unavailable.

    Patterns:
    - Cached responses (last-known-good)
    - Generic helpful messages
    - Degraded functionality indicators
    """

    def __init__(self):
        self.cache: dict = {}
        self.last_successful: dict = {}

    def get_cached_or_fallback(self, key: str, default_message: str) -> str:
        """Return cached value or fallback message."""
        if key in self.cache:
            logger.info(f"📦 Using cached response for: {key}")
            return self.cache[key]

        logger.warning(f"⚠ No cache available, using fallback: {default_message}")
        return default_message

    def update_cache(self, key: str, value: str):
        """Update cache with successful response."""
        self.cache[key] = value
        self.last_successful[key] = datetime.now()

    def get_generic_answer(self, query: str) -> str:
        """Provide generic but helpful response when RAG fails."""
        return (
            f"I apologize, but I'm experiencing technical difficulties accessing my knowledge base. "
            f"Your query about '{query[:50]}...' has been logged. "
            f"Please try again in a few moments or contact support if this persists."
        )

    def get_last_known_good(self, key: str) -> Optional[tuple]:
        """Get last successful response and its timestamp."""
        if key in self.cache and key in self.last_successful:
            age = (datetime.now() - self.last_successful[key]).total_seconds()
            return self.cache[key], age
        return None


# ==================== REQUEST QUEUE & BACKPRESSURE ====================

class RequestQueue:
    """
    In-memory queue with bounded size to prevent memory exhaustion.

    Features:
    - FIFO processing
    - Configurable max size (backpressure)
    - Worker pattern for async processing
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self.processed_count = 0
        self.rejected_count = 0

    def enqueue(self, item: Any) -> bool:
        """
        Add item to queue. Returns False if queue is full.

        Note: deque with maxlen automatically drops oldest items,
        but we track rejections for monitoring.
        """
        with self._lock:
            current_size = len(self.queue)
            if current_size >= self.max_size:
                self.rejected_count += 1
                logger.warning(f"⚠ Queue full ({self.max_size}), request rejected")
                return False

            self.queue.append(item)
            logger.debug(f"+ Queued item (queue size: {len(self.queue)})")
            return True

    def dequeue(self) -> Optional[Any]:
        """Remove and return oldest item from queue."""
        with self._lock:
            if not self.queue:
                return None
            item = self.queue.popleft()
            self.processed_count += 1
            return item

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self.queue)

    def stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                "current_size": len(self.queue),
                "processed": self.processed_count,
                "rejected": self.rejected_count,
                "capacity": self.max_size
            }


class QueueWorker:
    """Worker that processes items from a RequestQueue."""

    def __init__(self, queue: RequestQueue, process_func: Callable):
        self.queue = queue
        self.process_func = process_func
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start processing queue in background thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("🚀 Queue worker started")

    def stop(self):
        """Stop processing queue."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🛑 Queue worker stopped")

    def _process_loop(self):
        """Main processing loop."""
        while self.running:
            item = self.queue.dequeue()
            if item is None:
                time.sleep(0.1)  # No work, sleep briefly
                continue

            try:
                self.process_func(item)
            except Exception as e:
                logger.error(f"✗ Worker failed to process item: {e}")


# ==================== RESILIENT OPENAI CLIENT ====================

class ResilientOpenAIClient:
    """
    Wrapper for OpenAI API calls with built-in resilience.

    Features:
    - Retry strategy (different configs for embeddings vs completions)
    - Timeouts to prevent hanging
    - Circuit breaker for cascading failure prevention
    """

    def __init__(self, api_key: str, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout

        # Different retry strategies for different operations
        self.embedding_strategy = RetryStrategy(
            max_retries=3,
            initial_delay=1.0,
            exponential_base=2.0
        )

        self.completion_strategy = RetryStrategy(
            max_retries=2,  # Fewer retries for expensive completions
            initial_delay=2.0,
            exponential_base=2.0
        )

        # Circuit breaker for severe outages
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get text embedding with retry logic."""
        def _call():
            # Simulated API call - replace with actual OpenAI SDK call
            logger.debug(f"Calling OpenAI embeddings API: {text[:50]}...")
            # Simulate occasional failure
            if random.random() < 0.1:  # 10% failure rate for demo
                raise ConnectionError("Simulated API failure")
            return [random.random() for _ in range(1536)]  # Mock embedding

        return self.circuit_breaker.call(
            self.embedding_strategy.execute,
            _call
        )

    def get_completion(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Get completion with retry logic."""
        def _call():
            # Simulated API call - replace with actual OpenAI SDK call
            logger.debug(f"Calling OpenAI completions API: {prompt[:50]}...")
            # Simulate occasional failure
            if random.random() < 0.1:  # 10% failure rate for demo
                raise ConnectionError("Simulated API failure")
            return f"Mock completion for: {prompt[:30]}..."

        return self.circuit_breaker.call(
            self.completion_strategy.execute,
            _call
        )

    def get_circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.circuit_breaker.get_state()


# ==================== EXAMPLE: CIRCUIT-PROTECTED RAG ====================

class CircuitProtectedRAG:
    """
    Example RAG implementation with full resilience stack.

    Demonstrates:
    - Retry logic for transient failures
    - Circuit breaker for cascading failure prevention
    - Graceful degradation with fallbacks
    """

    def __init__(self, openai_client: ResilientOpenAIClient):
        self.client = openai_client
        self.fallbacks = GracefulFallbacks()
        self.vector_db_available = True  # Simulate vector DB state

    def query(self, question: str) -> str:
        """
        Query RAG system with full error handling.

        Flow:
        1. Try to get embedding and search vector DB
        2. If that fails, use circuit breaker
        3. If circuit is open, use fallback
        """
        try:
            # Try normal RAG flow
            logger.info(f"Processing query: {question}")

            # Get embedding (with retry)
            embedding = self.client.get_embedding(question)

            # Search vector DB (simulated)
            if not self.vector_db_available:
                raise ConnectionError("Vector DB unavailable")

            context = self._search_vector_db(embedding)

            # Generate answer (with retry)
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = self.client.get_completion(prompt)

            # Cache successful response
            self.fallbacks.update_cache(question, answer)

            return answer

        except CircuitBreakerOpenError as e:
            # Circuit is open - use fallback immediately
            logger.error(f"Circuit breaker open: {e}")
            return self.fallbacks.get_cached_or_fallback(
                question,
                self.fallbacks.get_generic_answer(question)
            )

        except Exception as e:
            # Other errors - try fallback
            logger.error(f"RAG query failed: {e}")
            cached = self.fallbacks.get_last_known_good(question)
            if cached:
                answer, age = cached
                return f"{answer}\n\n[Note: Using cached response from {age:.0f}s ago due to technical issues]"
            return self.fallbacks.get_generic_answer(question)

    def _search_vector_db(self, embedding: List[float]) -> str:
        """Simulate vector DB search."""
        # In real implementation, this would query Pinecone/Weaviate/etc.
        return "Mock context from vector database"


# ==================== UTILITY FUNCTIONS ====================

def simulate_flaky_service(failure_rate: float = 0.5) -> str:
    """Simulate a service that fails randomly."""
    if random.random() < failure_rate:
        raise ConnectionError(f"Simulated failure (rate={failure_rate})")
    return "Success!"


def simulate_rate_limit() -> str:
    """Simulate a 429 rate limit error."""
    class RateLimitError(Exception):
        def __init__(self):
            super().__init__("Rate limit exceeded")
            self.status_code = 429

    raise RateLimitError()
