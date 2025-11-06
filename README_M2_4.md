# M2.4 — Error Handling & Reliability

Production-ready resilience patterns for RAG systems with external API dependencies.

## Overview

This module implements battle-tested error handling patterns that reduce user-facing errors by 80-95% in production RAG systems. You'll learn when to use each pattern, how to tune thresholds, and—critically—when NOT to add complexity.

## What's Included

### Core Implementation (`m2_4_resilience.py`)
- **RetryStrategy**: Exponential backoff with jitter for transient failures
- **CircuitBreaker**: Three-state machine (CLOSED → OPEN → HALF_OPEN) to prevent cascading failures
- **GracefulFallbacks**: Cache-based degraded mode for service outages
- **RequestQueue**: Bounded FIFO queue with backpressure for traffic spikes
- **ResilientOpenAIClient**: Production-ready wrapper with all patterns integrated

### Interactive Notebook (`M2_4_Error_Handling_and_Reliability.ipynb`)
7 sections built incrementally:
1. **Reality Check** - What resilience solves (and doesn't)
2. **Smart Retries** - Exponential backoff, jitter, retryable vs non-retryable
3. **Circuit Breaker** - State transitions, recovery, false positives
4. **Graceful Degradation** - Fallback strategies, cache patterns
5. **Queueing & Backpressure** - Traffic spike handling, bounded queues
6. **Putting It Together** - Full resilience stack integration
7. **Common Failures & Decision Card** - 5 failure modes + production configs

### Configuration (`config.py`, `.env.example`)
- Tunable thresholds for all patterns
- Operation-specific retry configs (embeddings vs completions)
- Production profiles (Conservative, Balanced, Aggressive, Cost-Optimized)

### Tests (`tests_resilience.py`)
Comprehensive test suite covering:
- Retry success/failure scenarios
- Circuit breaker state transitions
- Fallback cache behavior
- Queue backpressure
- Integration tests

## Quick Start

### Installation

```bash
# Clone or download this module
cd rag21d_learners

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Basic Usage

```python
from m2_4_resilience import RetryStrategy, CircuitBreaker, with_retry

# 1. Simple retry decorator
@with_retry(max_retries=3, initial_delay=1.0)
def fetch_embedding(text):
    # Your OpenAI API call here
    pass

# 2. Manual retry strategy
strategy = RetryStrategy(max_retries=3, initial_delay=1.0)
result = strategy.execute(fetch_embedding, "query text")

# 3. Circuit breaker for protection
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
result = cb.call(fetch_embedding, "query text")

# 4. Combined resilience
from m2_4_resilience import ResilientOpenAIClient
client = ResilientOpenAIClient(api_key="your-key")
embedding = client.get_embedding("What is AI?")  # Automatically resilient
```

### Running the Notebook

```bash
jupyter notebook M2_4_Error_Handling_and_Reliability.ipynb
```

The notebook is designed to be run cell-by-cell, with each section demonstrating a specific pattern. All cells print `SAVED_SECTION:N` markers to show incremental progress.

### Running Tests

```bash
# Run all tests
pytest tests_resilience.py -v

# Run specific test
pytest tests_resilience.py::test_retry_succeeds_on_flaky_function -v
```

## Key Concepts

### 1. Retry Strategy

**When to use**: External API calls, network operations, transient failures <5%
**When NOT to use**: Database writes (idempotency issues), real-time systems (<50ms SLA)

```python
RetryStrategy(
    max_retries=3,           # How many times to retry
    initial_delay=1.0,       # Starting delay in seconds
    exponential_base=2.0,    # Delay multiplier (1s → 2s → 4s)
    jitter=True              # Add randomness to prevent thundering herd
)
```

**Key insight**: Distinguishes between retryable (5xx, 429) and non-retryable (4xx) errors to avoid wasting API calls.

### 2. Circuit Breaker

**When to use**: Protecting downstream services, preventing cascading failures
**When NOT to use**: Single-service apps, when false positives unacceptable

```python
CircuitBreaker(
    failure_threshold=5,     # Open after N consecutive failures
    recovery_timeout=60.0,   # Wait N seconds before testing recovery
)
```

**States**:
- **CLOSED**: Normal operation, tracking failures
- **OPEN**: Service down, rejecting all requests (fast-fail)
- **HALF_OPEN**: Testing recovery with one request

**Trade-off**: Prevents cascading failures but can cause false positives if thresholds too aggressive.

### 3. Graceful Degradation

**When to use**: User-facing apps, when cached/partial data acceptable
**When NOT to use**: Financial transactions, real-time data requirements

```python
fallbacks = GracefulFallbacks()

# Cache successful responses
fallbacks.update_cache(query, answer)

# Use cache during outages
cached = fallbacks.get_cached_or_fallback(
    query,
    fallbacks.get_generic_answer(query)
)
```

**Patterns**:
- Last-known-good responses (with age indicator)
- Generic helpful messages (better than stack traces)
- Partial functionality (some features work, others degraded)

### 4. Request Queue

**When to use**: Traffic spikes, rate-limited APIs, background processing OK
**When NOT to use**: Latency-sensitive ops, low traffic, immediate response required

```python
queue = RequestQueue(max_size=1000)  # ALWAYS bounded!

# Producer
if queue.enqueue(request):
    print("Accepted")
else:
    print("Rejected - backpressure")

# Consumer
worker = QueueWorker(queue, process_function)
worker.start()
```

**Critical**: Always use bounded queues to prevent memory exhaustion.

## Production Configuration

### Recommended Profiles

**Balanced (Start Here)**:
```python
RetryStrategy(max_retries=3, initial_delay=1.0)
CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
RequestQueue(max_size=1000)
```

**Conservative (High Reliability)**:
```python
RetryStrategy(max_retries=5, initial_delay=2.0)
CircuitBreaker(failure_threshold=3, recovery_timeout=120.0)
RequestQueue(max_size=500)
```

**Aggressive (Fast Recovery)**:
```python
RetryStrategy(max_retries=2, initial_delay=0.5)
CircuitBreaker(failure_threshold=10, recovery_timeout=30.0)
RequestQueue(max_size=2000)
```

**Cost-Optimized (Minimize API Costs)**:
```python
RetryStrategy(max_retries=1, initial_delay=2.0)
CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
RequestQueue(max_size=500)
```

### Environment Variables

See `.env.example` for full configuration options:

```bash
# Core settings
RETRY_MAX_ATTEMPTS=3
CB_FAILURE_THRESHOLD=5
QUEUE_MAX_SIZE=1000

# Operation-specific
EMBEDDING_MAX_RETRIES=3
COMPLETION_MAX_RETRIES=2

# Timeouts
OPENAI_TIMEOUT=30.0
VECTOR_DB_TIMEOUT=10.0
```

## Common Pitfalls

### 1. Retry Storm
**Problem**: Aggressive retries amplify load during outages
**Solution**: Use exponential backoff + jitter + circuit breaker

### 2. Circuit Breaker False Positives
**Problem**: Over-sensitive thresholds reject valid requests
**Solution**: Tune `failure_threshold` higher (5-10), monitor state transitions

### 3. Queue Memory Exhaustion
**Problem**: Unbounded queues consume all memory
**Solution**: ALWAYS use `max_size`, reject when full (backpressure)

### 4. Graceful Degradation Stuck
**Problem**: Fallbacks remain active after service recovers
**Solution**: Circuit breaker automatically tests recovery (HALF_OPEN state)

### 5. Retrying Non-Retryable Errors
**Problem**: Wasting API calls on 404s
**Solution**: `RetryStrategy.is_retryable()` checks status codes (5xx = retry, 4xx = don't)

## When NOT to Use

Don't add resilience patterns if:

- **Simple internal tools** (<10 users, non-critical)
- **Real-time systems** (<50ms SLA requirements)
- **Batch processing** (failures should be investigated, not hidden)
- **Single-user applications** (complexity not justified)
- **Already using service mesh** (Istio/Linkerd handle this at infra level)

**Remember**: These patterns add 20-30% code complexity. Use them when the trade-off makes sense.

## Architecture

### Full Resilience Stack

```
User Query
    ↓
Request Queue (backpressure)
    ↓
Retry Strategy (exponential backoff)
    ↓
Circuit Breaker (cascading failure prevention)
    ↓
API Call (OpenAI, Vector DB, etc.)
    ↓
   [Success] → Cache response → Return to user
    ↓
   [Failure] → Fallback (cached or generic) → Return to user
```

### Integration Example

```python
class ProductionRAG:
    def __init__(self):
        self.queue = RequestQueue(max_size=1000)
        self.retry = RetryStrategy(max_retries=3, initial_delay=1.0)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5)
        self.fallbacks = GracefulFallbacks()

    def query(self, question: str) -> dict:
        # Step 1: Queue (backpressure)
        if not self.queue.enqueue(question):
            return {"answer": "System busy", "degraded": True}

        # Step 2: Retry + Circuit Breaker
        try:
            def call():
                return self.circuit_breaker.call(lambda: self._api_call(question))

            answer = self.retry.execute(call)
            self.fallbacks.update_cache(question, answer)
            return {"answer": answer, "degraded": False}

        except Exception:
            # Step 3: Fallback
            cached = self.fallbacks.get_cached_or_fallback(
                question,
                self.fallbacks.get_generic_answer(question)
            )
            return {"answer": cached, "degraded": True}
```

## Monitoring

Track these metrics in production:

- **Retry rate**: % of requests requiring retries (should be <5%)
- **Circuit breaker state**: Time spent in OPEN/HALF_OPEN (should be minimal)
- **Queue depth**: Current size vs capacity (alert at 75%)
- **Fallback usage**: % of requests using cache (should be <1% normally)
- **Error types**: Breakdown of retryable vs non-retryable errors

## Development Workflow

1. **Start simple**: Add retry decorator to critical API calls
2. **Add circuit breaker**: Once you have monitoring in place
3. **Implement fallbacks**: For user-facing features
4. **Add queueing**: If experiencing traffic spikes
5. **Tune thresholds**: Based on real production data

## Resources

- **Notebook**: `M2_4_Error_Handling_and_Reliability.ipynb` - Interactive examples
- **Source**: `m2_4_resilience.py` - Copy into your project
- **Tests**: `tests_resilience.py` - Verify behavior
- **Config**: `.env.example` - Production settings

## Gotchas

1. **Retries add latency**: 3 retries with 1s delay = 7s total worst-case
2. **Circuit breaker can reject valid requests**: During recovery testing
3. **Fallbacks can serve stale data**: Add age indicators to responses
4. **Queues add latency**: Typical 100-500ms during normal load
5. **Jitter is critical**: Without it, all retries happen simultaneously (thundering herd)

## Trade-Offs Summary

| Pattern | Benefit | Cost | When to Use |
|---------|---------|------|-------------|
| Retry | 80-95% error reduction | +50-200ms latency | Always for API calls |
| Circuit Breaker | Prevents cascades | False positives | Production systems |
| Fallbacks | Better UX | Stale data risk | User-facing apps |
| Queue | Handles spikes | Added latency | Traffic bursts expected |

## Next Steps

1. Run the notebook to see all patterns in action
2. Copy `m2_4_resilience.py` into your RAG project
3. Start with retry decorator (easiest, highest impact)
4. Add circuit breaker before production deployment
5. Implement fallbacks for user-facing features
6. Tune thresholds based on your monitoring data

## Support

For issues, questions, or contributions:
- Review the notebook examples first
- Check `tests_resilience.py` for usage patterns
- Tune config values in `.env` for your use case

**Remember**: The best error handling is the kind users never notice.

---

**Module**: M2.4 — Error Handling & Reliability
**Target**: Production RAG Systems
**Impact**: 80-95% error reduction
**Cost**: 8-12 hours implementation
**ROI**: High for user-facing applications
