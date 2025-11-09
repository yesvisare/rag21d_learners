# M2.4 — Error Handling & Reliability

Production-ready resilience patterns for RAG systems with external API dependencies.

---

## **Purpose**

**Make RAG systems production-ready with error handling that actually works.**

This module teaches you to handle the 2-5% base failure rate of external APIs (OpenAI, vector databases) automatically, reducing user-facing errors by 80-95%. You'll learn patterns used by companies like Netflix, AWS, and Google to keep services running even when dependencies fail.

**Real-world impact**: Transform "Service Unavailable 503" errors into slightly slower but functional responses.

## **Concepts Covered**

1. **Retry Strategy with Exponential Backoff**
   - Smart error classification: What to retry (5xx, 429) vs what not to (4xx)
   - Exponential delays prevent thundering herd
   - Jitter spreads retry timing to reduce server load collisions

2. **Circuit Breaker Pattern**
   - Three-state machine: CLOSED → OPEN → HALF_OPEN
   - Prevents retry storms during outages (fail fast vs retry forever)
   - Automatic recovery testing with tunable thresholds

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
   - Complexity vs reliability costs (20-30% code increase)
   - Latency overhead analysis (50-200ms per retry)
   - Production tuning guidance

## **After Completing This Module**

You will be able to:
- ✅ Distinguish retryable (5xx, 429) from non-retryable (4xx) errors
- ✅ Implement circuit breakers to prevent cascading failures
- ✅ Design graceful fallback strategies for degraded operation
- ✅ Handle traffic spikes with bounded queues and backpressure
- ✅ Tune resilience thresholds based on monitoring data
- ✅ Make informed decisions about when NOT to add complexity
- ✅ Deploy RAG systems with 80-95% error reduction

**Production ready**: Copy patterns into your code, tune thresholds, deploy.

## **Context in Track**

**Prerequisites** (M1.x - M2.3):
- M1.1-M1.4: Basic RAG architecture (embeddings, retrieval, generation)
- M2.1-M2.3: Chunking strategies, vector search optimization, evaluation metrics

**This Module** (M2.4):
Making RAG systems **production-ready** with error handling and resilience patterns.
Focuses on external dependency failures (API outages, rate limits, network issues).

**Next Steps**:
- M3.x: Advanced RAG techniques (hybrid search, re-ranking, etc.)
- Production deployment with monitoring and alerting

**Why This Matters**:
- **Without resilience**: 2-5% base failure rate, cascading outages, crashes
- **With resilience**: <0.1% error rate, graceful degradation, smooth traffic handling

---

## Project Structure

```
rag21d_learners/
├── README.md                          # This file
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore patterns
├── .env.example                        # Environment template
├── requirements.txt                    # Python dependencies
├── app.py                              # FastAPI application entry point
│
├── src/m2_4_error_handling/           # Core module package
│   ├── __init__.py                     # Purpose/Concepts/After/Context docs
│   ├── config.py                       # Configuration and environment
│   ├── module.py                       # Core resilience implementations
│   └── router.py                       # FastAPI demo endpoints
│
├── notebooks/                          # Interactive tutorials
│   └── M2_4_Error_Handling_and_Reliability.ipynb
│
├── tests/                              # Test suite
│   ├── test_resilience.py              # Core pattern tests
│   └── test_smoke.py                   # FastAPI endpoint tests
│
└── scripts/                            # Utility scripts
    └── run_local.ps1                   # Windows development server
```

## Quick Start

### Installation

```bash
# Clone and navigate to project
cd rag21d_learners

# Install dependencies
pip install -r requirements.txt

# Copy environment template (optional for demos)
cp .env.example .env
```

### Run FastAPI Demo Server

**Linux/Mac**:
```bash
# Set PYTHONPATH and run
export PYTHONPATH=$PWD
uvicorn app:app --reload
```

**Windows PowerShell**:
```powershell
# One-liner
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"

# Or use script
.\scripts\run_local.ps1
```

**Access**:
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Circuit state: http://localhost:8000/m2_4/state

### Python Usage

```python
from src.m2_4_error_handling import RetryStrategy, CircuitBreaker, with_retry

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

# 4. Full resilience client
from src.m2_4_error_handling import ResilientOpenAIClient
client = ResilientOpenAIClient(api_key="your-key")
embedding = client.get_embedding("What is AI?")
```

### CLI Usage

```bash
# Run module directly
PYTHONPATH=. python -m src.m2_4_error_handling.module

# Check configuration
PYTHONPATH=. python -m src.m2_4_error_handling.config
```

### Running the Notebook

```bash
# From project root
jupyter notebook notebooks/M2_4_Error_Handling_and_Reliability.ipynb
```

### Running Tests

```bash
# All tests
PYTHONPATH=. pytest tests/ -v

# Specific test file
PYTHONPATH=. pytest tests/test_resilience.py -v

# Smoke tests (FastAPI)
PYTHONPATH=. pytest tests/test_smoke.py -v
```

## API Demo Endpoints

Test resilience patterns without API keys:

### Health & State

```bash
# Module health
curl http://localhost:8000/m2_4/health

# Circuit breaker state
curl http://localhost:8000/m2_4/state

# Queue statistics
curl http://localhost:8000/m2_4/queue/stats
```

### Simulation

```bash
# Simulate retry pattern
curl -X POST http://localhost:8000/m2_4/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "n": 20,
    "failure_rate": 0.4,
    "operation": "retry"
  }'

# Simulate circuit breaker
curl -X POST http://localhost:8000/m2_4/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "n": 15,
    "failure_rate": 0.8,
    "operation": "circuit_breaker"
  }'

# Reset state for fresh demos
curl -X POST http://localhost:8000/m2_4/reset
```

## Core Patterns

### 1. Retry Strategy

**When to use**: External API calls, network operations, transient failures <5%
**When NOT to use**: Database writes (idempotency issues), real-time systems (<50ms SLA)

```python
from src.m2_4_error_handling import RetryStrategy

strategy = RetryStrategy(
    max_retries=3,           # How many times to retry
    initial_delay=1.0,       # Starting delay in seconds
    exponential_base=2.0,    # Delay multiplier (1s → 2s → 4s)
    jitter=True              # Add randomness to prevent thundering herd
)

result = strategy.execute(your_function, *args, **kwargs)
```

**Key insight**: Distinguishes between retryable (5xx, 429) and non-retryable (4xx) errors to avoid wasting API calls.

### 2. Circuit Breaker

**When to use**: Protecting downstream services, preventing cascading failures
**When NOT to use**: Single-service apps, when false positives unacceptable

```python
from src.m2_4_error_handling import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,     # Open after N consecutive failures
    recovery_timeout=60.0,   # Wait N seconds before testing recovery
)

result = cb.call(your_function, *args, **kwargs)
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
from src.m2_4_error_handling import GracefulFallbacks

fallbacks = GracefulFallbacks()

# Cache successful responses
fallbacks.update_cache(query, answer)

# Use cache during outages
cached = fallbacks.get_cached_or_fallback(
    query,
    fallbacks.get_generic_answer(query)
)

# Get last-known-good with age
answer, age = fallbacks.get_last_known_good(query)
```

**Patterns**:
- Last-known-good responses (with age indicator)
- Generic helpful messages (better than stack traces)
- Partial functionality (some features work, others degraded)

### 4. Request Queue

**When to use**: Traffic spikes, rate-limited APIs, background processing OK
**When NOT to use**: Latency-sensitive ops, low traffic, immediate response required

```python
from src.m2_4_error_handling import RequestQueue, QueueWorker

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

### Environment Variables

See `.env.example` for all options:

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

## Trade-Offs Summary

| Pattern | Benefit | Cost | When to Use |
|---------|---------|------|-------------|
| Retry | 80-95% error reduction | +50-200ms latency | Always for API calls |
| Circuit Breaker | Prevents cascades | False positives | Production systems |
| Fallbacks | Better UX | Stale data risk | User-facing apps |
| Queue | Handles spikes | Added latency | Traffic bursts expected |

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

- **Notebook**: `notebooks/M2_4_Error_Handling_and_Reliability.ipynb` - Interactive examples
- **Source**: `src/m2_4_error_handling/module.py` - Copy into your project
- **Tests**: `tests/test_resilience.py` - Verify behavior
- **Config**: `.env.example` - Production settings
- **API Docs**: http://localhost:8000/docs - Interactive demo

## Support

For issues, questions, or contributions:
- Review the notebook examples first
- Check `tests/` for usage patterns
- Tune config values in `.env` for your use case

**Remember**: The best error handling is the kind users never notice.

---

**Module**: M2.4 — Error Handling & Reliability
**Slug**: `m2_4_error_handling`
**Target**: Production RAG Systems
**Impact**: 80-95% error reduction
**Cost**: 8-12 hours implementation
**ROI**: High for user-facing applications
**License**: MIT
