# Video M2.4: Error Handling & Reliability (Enhanced) — 32 min

## [0:00] Introduction

[SLIDE: "M2.4: Error Handling & Reliability"]

Welcome to the final video in Module 2! We've optimized costs, we're monitoring everything, and now we need to make sure our system doesn't fall over when things go wrong. And trust me - things WILL go wrong.

APIs will timeout. Rate limits will hit. Vector databases will hiccup. Your embedding model will return 500 errors at 3 AM. The question isn't if it will happen, but how your system will handle it.

[SLIDE: "Common Failure Modes"]
```
1. API Failures
   - Rate limits (429)
   - Timeouts
   - Service degradation

2. Data Issues
   - Empty search results
   - Malformed documents
   - Encoding errors

3. Resource Exhaustion
   - Memory leaks
   - Connection pool exhaustion
   - Queue backups

4. Cascading Failures
   - One slow component affects everything
   - Thundering herd problem
```

Today we're implementing: retries with exponential backoff, circuit breakers, graceful degradation, and request queuing. Let's make this thing production-ready.

---

<!-- ============ NEW SECTION: REALITY CHECK ============ -->

## [1:30] Reality Check: What Error Handling Actually Solves

**[1:30] [SLIDE: Reality Check - The Honest Truth About Resilience]**

Before we write a single line of code, let's be completely honest about what these error handling patterns do and don't do. This matters because I see teams add complexity without understanding the trade-offs.

**What error handling DOES well:**

- ✅ **Handles transient failures automatically** - That random 503 from your vector DB at 2 AM? Retried and recovered without waking you up. Reduces user-facing errors by 80-95% in most systems.
- ✅ **Prevents cascading failures** - One slow service doesn't take down your entire system. Circuit breakers isolate failures so they don't spread.
- ✅ **Improves user experience during outages** - Graceful degradation means users get *something* useful instead of a blank error page. Your system degrades smoothly instead of falling off a cliff.

**What error handling DOESN'T do:**

- ❌ **Doesn't fix bugs or architectural issues** - If your code has a logic error, retrying it 10 times just fails 10 times. Error handling masks symptoms; it doesn't cure diseases.
- ❌ **Doesn't eliminate latency** - Every retry adds 50-200ms. Circuit breakers add 5-15ms overhead per request. If you have strict latency SLAs (<50ms), this might break them.
- ❌ **Doesn't handle data corruption** - If your embedding has bad data or your vector DB returns wrong results, retry logic happily returns wrong results faster. Garbage in, garbage out.

**[EMPHASIS]** Here's the critical limitation: circuit breakers can cause false positives. If your failure threshold is too aggressive, a brief spike opens the circuit and rejects valid requests. I've seen this take down healthy services.

**The trade-offs you're making:**

- You gain 95% error reduction but add 20-30% code complexity. Your codebase now has retry logic, circuit breaker state machines, and fallback paths to maintain.
- Works great for transient API failures but poorly for systematic bugs. If OpenAI changes their API format, retrying 10 times won't help.
- Cost structure: 10-20% increased infrastructure cost (queue memory, duplicate retry traffic, monitoring overhead). Plus 6-8 hours initial implementation time.

**[PAUSE]**

We'll see these trade-offs in action throughout this video. Keep them in mind as we build.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ NEW SECTION: ALTERNATIVE SOLUTIONS ============ -->

## [4:00] Alternative Solutions: Different Error Handling Philosophies

**[4:00] [SLIDE: Alternative Approaches to Reliability]**

Before we commit to building a full resilience stack, you should know there are fundamentally different approaches to error handling. Let me show you three options and when each makes sense.

**Option 1: Full Resilience Stack (what we're teaching today)**

- **Best for:** User-facing applications with external API dependencies; systems where uptime matters more than implementation time
- **Key trade-off:** High complexity (retries + circuit breakers + queuing + fallbacks) vs maximum reliability
- **Cost:** 8-12 hours implementation; 10-20% increased infrastructure cost; requires dedicated monitoring
- **Example use case:** Customer-facing chatbot with 1000+ daily users. A 5-minute OpenAI outage can't take down your entire service.

**Option 2: Fail Fast with Comprehensive Monitoring**

- **Best for:** Internal tools, admin dashboards, batch processing jobs; teams with strong ops/monitoring culture
- **Key trade-off:** Simple code (basic try/catch) vs manual intervention when things break
- **Cost:** 2-4 hours implementation; minimal infrastructure overhead; requires on-call rotation
- **Example use case:** Internal document processing pipeline that runs nightly. If it fails, Slack alerts ops team and they rerun manually. 10 users can tolerate brief outages.

**Option 3: External Orchestration (Service Mesh/API Gateway)**

- **Best for:** Microservices architectures where infrastructure team owns reliability; standardized retry/timeout policies across services
- **Key trade-off:** Zero application code for reliability vs dependency on infrastructure layer; vendor lock-in
- **Cost:** 0 hours application dev (infrastructure team handles it); requires Istio/Linkerd/Kong expertise
- **Example use case:** 20+ microservices where every team implementing their own retry logic creates chaos. Centralize it in Istio with standardized policies.

**[DIAGRAM: Decision Framework]**

```
START
│
├─ >10 users + user-facing? 
│  └─ YES → Full Resilience Stack (Option 1)
│  └─ NO → Continue
│
├─ Microservices architecture?
│  └─ YES → Service Mesh (Option 3)
│  └─ NO → Continue
│
└─ <10 users + batch processing?
   └─ YES → Fail Fast + Monitoring (Option 2)
```

**For this video, we're using the Full Resilience Stack because:**

We're building a customer-facing RAG system with external dependencies (OpenAI, Pinecone). Users expect sub-second responses and can't tolerate "try again later" errors. We don't have infrastructure team to manage a service mesh, so we're implementing reliability in application code. This matches the most common real-world scenario for RAG systems.

**[PAUSE]**

Now let's build it.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ EXISTING SECTION: Updated timestamp ============ -->

## [6:30] Retry Patterns with Exponential Backoff

**[6:30] [SLIDE: "Smart Retry Logic"]**

First rule: Don't retry blindly. Some errors should retry, others shouldn't.

[CODE: "retry_patterns.py"]
```python
import time
import random
from typing import Callable, Optional, Type, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RetryError(Exception):
    """Raised when all retries are exhausted."""
    pass

class RetryStrategy:
    """
    Configurable retry strategy with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions
    
    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Never retry these
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        
        # Check if it's in retryable list
        if isinstance(exception, self.retryable_exceptions):
            # For API errors, check status code
            if hasattr(exception, 'status_code'):
                # Don't retry 4xx errors (except 429 rate limit)
                if 400 <= exception.status_code < 500 and exception.status_code != 429:
                    return False
            return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        # Exponential backoff: delay = initial * (base ** attempt)
        delay = self.initial_delay * (self.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self.should_retry(e):
                    logger.error(f"Non-retryable error: {e}")
                    raise
                
                # Check if we have retries left
                if attempt >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exhausted")
                    break
                
                # Calculate delay
                delay = self.calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                time.sleep(delay)
        
        # All retries exhausted
        raise RetryError(
            f"Failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_exception}"
        ) from last_exception

def with_retry(strategy: Optional[RetryStrategy] = None):
    """
    Decorator to add retry logic to any function.
    """
    if strategy is None:
        strategy = RetryStrategy()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return strategy.execute(func, *args, **kwargs)
        return wrapper
    return decorator

# Example usage with OpenAI
class ResilientOpenAIClient:
    """
    OpenAI client wrapper with intelligent retry logic.
    """
    
    def __init__(self, client):
        self.client = client
        
        # Different strategies for different operations
        self.embedding_strategy = RetryStrategy(
            max_retries=3,
            initial_delay=0.5,
            max_delay=10.0,
            retryable_exceptions=(Exception,),
            non_retryable_exceptions=(ValueError, TypeError)
        )
        
        self.completion_strategy = RetryStrategy(
            max_retries=2,  # LLM calls are expensive, fewer retries
            initial_delay=1.0,
            max_delay=30.0
        )
    
    @with_retry()
    def embed(self, text: str, model: str = "text-embedding-3-small"):
        """Embed text with retry logic."""
        response = self.client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    
    def complete_with_retry(self, messages: list, model: str = "gpt-3.5-turbo"):
        """Generate completion with retry logic."""
        return self.completion_strategy.execute(
            self._complete,
            messages=messages,
            model=model
        )
    
    def _complete(self, messages: list, model: str):
        """Internal completion method."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30.0  # Always set timeouts!
        )
        return response.choices[0].message.content

# Test the retry logic
def demonstrate_retry():
    """Show retry logic in action."""
    
    call_count = {'count': 0}
    
    @with_retry(RetryStrategy(max_retries=3, initial_delay=0.1))
    def flaky_function():
        """Simulate a flaky API call."""
        call_count['count'] += 1
        print(f"Attempt {call_count['count']}")
        
        # Fail first 2 times, succeed on 3rd
        if call_count['count'] < 3:
            raise Exception("Simulated API error")
        
        return "Success!"
    
    print("Testing retry logic...")
    result = flaky_function()
    print(f"Result: {result}")
    print(f"Total attempts: {call_count['count']}")
```

[TERMINAL: Run retry demonstration]

<!-- ============ END EXISTING SECTION ============ -->

---

<!-- ============ EXISTING SECTION: Updated timestamp ============ -->

## [10:30] Circuit Breaker Pattern

**[10:30] [SLIDE: "Preventing Cascading Failures"]**

Circuit breakers protect your system from hammering a failing service.

[CODE: "circuit_breaker.py"]
```python
import time
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass
import threading

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Open after N failures
    success_threshold: int = 2  # Close after N successes in half-open
    timeout: float = 60.0  # Seconds before trying half-open
    expected_exception: Type[Exception] = Exception

class CircuitBreakerOpen(Exception):
    """Raised when circuit is open."""
    pass

class CircuitBreaker:
    """
    Implements the circuit breaker pattern.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker.
        """
        with self._lock:
            # Check if we should attempt to transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit {self.name}: Attempting reset (HALF_OPEN)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Failing fast to prevent cascading failures."
                    )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit {self.name}: CLOSED (recovered)")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed during testing, go back to OPEN
                logger.warning(f"Circuit {self.name}: Failed during HALF_OPEN, back to OPEN")
                self.state = CircuitState.OPEN
                self.success_count = 0
            
            elif self.failure_count >= self.config.failure_threshold:
                # Too many failures, open the circuit
                logger.error(
                    f"Circuit {self.name}: OPEN after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count
            }
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit {self.name}: Manually reset")

# Wrapper for RAG components
class CircuitProtectedRAG:
    """
    RAG system with circuit breakers on all external dependencies.
    """
    
    def __init__(self, openai_client, vector_db, cache):
        self.openai = openai_client
        self.vector_db = vector_db
        self.cache = cache
        
        # Create circuit breakers for each service
        self.circuits = {
            'openai': CircuitBreaker('openai', CircuitBreakerConfig(
                failure_threshold=3,
                timeout=30.0
            )),
            'vector_db': CircuitBreaker('vector_db', CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60.0
            )),
            'cache': CircuitBreaker('cache', CircuitBreakerConfig(
                failure_threshold=10,  # Cache can fail more before opening
                timeout=10.0
            ))
        }
    
    def query(self, user_query: str) -> dict:
        """
        Process query with circuit breaker protection.
        """
        # Try cache (if circuit is closed)
        try:
            cached = self.circuits['cache'].call(
                self.cache.get_cached_response,
                user_query
            )
            if cached:
                return {'response': cached, 'source': 'cache'}
        except CircuitBreakerOpen:
            logger.warning("Cache circuit is open, skipping cache")
        except Exception as e:
            logger.error(f"Cache error: {e}")
        
        # Search vector DB (with circuit breaker)
        try:
            docs = self.circuits['vector_db'].call(
                self.vector_db.search,
                user_query,
                top_k=5
            )
        except CircuitBreakerOpen:
            # Graceful degradation: return generic response
            return {
                'response': "I'm having trouble accessing my knowledge base right now. Please try again shortly.",
                'source': 'fallback',
                'error': 'vector_db_unavailable'
            }
        
        # Generate response (with circuit breaker)
        try:
            context = self._format_context(docs)
            response = self.circuits['openai'].call(
                self._generate_response,
                user_query,
                context
            )
            
            # Try to cache (best effort)
            try:
                self.circuits['cache'].call(
                    self.cache.cache_response,
                    user_query,
                    response
                )
            except:
                pass  # Cache failure is not critical
            
            return {
                'response': response,
                'source': 'llm'
            }
        
        except CircuitBreakerOpen:
            # LLM is down - use fallback
            return {
                'response': "I'm experiencing high load right now. Please try again in a moment.",
                'source': 'fallback',
                'error': 'llm_unavailable'
            }
    
    def _format_context(self, docs):
        return "\n".join([d['content'] for d in docs])
    
    def _generate_response(self, query, context):
        # Your LLM call here
        return "Generated response"
    
    def get_circuit_states(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: circuit.get_state()
            for name, circuit in self.circuits.items()
        }
```

[SCREEN: Show circuit breaker state transitions]

<!-- ============ END EXISTING SECTION ============ -->

---

<!-- ============ EXISTING SECTION: Updated timestamp ============ -->

## [15:00] Graceful Degradation

**[15:00] [SLIDE: "Failing Gracefully"]**

When things break, give users something useful instead of errors.

[CODE: "graceful_degradation.py"]
```python
from typing import Optional, List, Dict
from enum import Enum

class ServiceTier(Enum):
    """Service quality tiers for graceful degradation."""
    FULL = "full"  # All features working
    DEGRADED = "degraded"  # Reduced functionality
    MINIMAL = "minimal"  # Basic functionality only
    OFFLINE = "offline"  # Service unavailable

class GracefulRAGSystem:
    """
    RAG system with multiple fallback levels.
    """
    
    def __init__(self, openai_client, vector_db, cache, fallback_responses):
        self.openai = openai_client
        self.vector_db = vector_db
        self.cache = cache
        self.fallback_responses = fallback_responses
        
        self.current_tier = ServiceTier.FULL
    
    def query(self, user_query: str) -> dict:
        """
        Process query with graceful degradation.
        """
        # Level 1: Full service (cache + vector + LLM)
        if self.current_tier == ServiceTier.FULL:
            try:
                return self._full_rag_query(user_query)
            except Exception as e:
                logger.warning(f"Full service failed: {e}, degrading...")
                self.current_tier = ServiceTier.DEGRADED
        
        # Level 2: Degraded (skip cache, use vector + LLM)
        if self.current_tier == ServiceTier.DEGRADED:
            try:
                return self._degraded_query(user_query)
            except Exception as e:
                logger.warning(f"Degraded service failed: {e}, going minimal...")
                self.current_tier = ServiceTier.MINIMAL
        
        # Level 3: Minimal (cached responses only or keyword matching)
        if self.current_tier == ServiceTier.MINIMAL:
            try:
                return self._minimal_query(user_query)
            except Exception as e:
                logger.error(f"Minimal service failed: {e}, offline...")
                self.current_tier = ServiceTier.OFFLINE
        
        # Level 4: Offline (static fallback)
        return self._offline_response(user_query)
    
    def _full_rag_query(self, query: str) -> dict:
        """Full RAG pipeline."""
        # Try cache
        cached = self.cache.get_cached_response(query)
        if cached:
            return {
                'response': cached,
                'tier': ServiceTier.FULL.value,
                'source': 'cache'
            }
        
        # Vector search
        docs = self.vector_db.search(query, top_k=5)
        
        # LLM generation
        context = "\n".join([d['content'] for d in docs])
        response = self._call_llm(query, context)
        
        # Cache result
        self.cache.cache_response(query, response)
        
        return {
            'response': response,
            'tier': ServiceTier.FULL.value,
            'source': 'llm'
        }
    
    def _degraded_query(self, query: str) -> dict:
        """Degraded service - no cache, but vector + LLM work."""
        docs = self.vector_db.search(query, top_k=3)  # Fewer docs
        context = "\n".join([d['content'] for d in docs])
        
        # Use faster/cheaper model
        response = self._call_llm(
            query,
            context,
            model="gpt-3.5-turbo",  # Fallback to faster model
            max_tokens=200  # Shorter responses
        )
        
        return {
            'response': response,
            'tier': ServiceTier.DEGRADED.value,
            'source': 'llm_degraded',
            'notice': 'Service is experiencing high load. Responses may be shorter than usual.'
        }
    
    def _minimal_query(self, query: str) -> dict:
        """Minimal service - keyword matching to cached responses."""
        # Try to find similar cached query
        query_lower = query.lower()
        
        for keyword, response in self.fallback_responses.items():
            if keyword in query_lower:
                return {
                    'response': response,
                    'tier': ServiceTier.MINIMAL.value,
                    'source': 'keyword_match',
                    'notice': 'Service is degraded. Using cached information.'
                }
        
        # No match found
        raise Exception("No keyword match found")
    
    def _offline_response(self, query: str) -> dict:
        """Offline - return static error message."""
        return {
            'response': (
                "I'm currently unavailable due to technical difficulties. "
                "Please try again in a few minutes, or contact support "
                "if the issue persists."
            ),
            'tier': ServiceTier.OFFLINE.value,
            'source': 'static_fallback',
            'error': 'service_unavailable'
        }
    
    def _call_llm(
        self,
        query: str,
        context: str,
        model: str = "gpt-4",
        max_tokens: int = 500
    ) -> str:
        """Call LLM with error handling."""
        response = self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer based on context."},
                {"role": "user", "content": f"Context: {context}\n\nQ: {query}"}
            ],
            max_tokens=max_tokens,
            timeout=10.0
        )
        return response.choices[0].message.content

# Setup fallback responses
fallback_responses = {
    'return': "Our return policy allows returns within 30 days with receipt.",
    'shipping': "Standard shipping takes 3-5 business days.",
    'refund': "Refunds are processed within 5-7 business days to original payment method.",
    'contact': "You can reach support at support@example.com or call 1-800-555-0123.",
}

# Usage
def demonstrate_graceful_degradation():
    """Show graceful degradation in action."""
    system = GracefulRAGSystem(None, None, None, fallback_responses)
    
    # Simulate full service
    print("=" * 60)
    print("FULL SERVICE")
    result = system.query("What's your return policy?")
    print(f"Tier: {result['tier']}")
    print(f"Response: {result['response']}")
    
    # Simulate degradation
    system.current_tier = ServiceTier.MINIMAL
    print("\n" + "=" * 60)
    print("MINIMAL SERVICE (Degraded)")
    result = system.query("How do I return something?")
    print(f"Tier: {result['tier']}")
    print(f"Response: {result['response']}")
    if 'notice' in result:
        print(f"Notice: {result['notice']}")
```

[SCREEN: Show degradation levels]

<!-- ============ END EXISTING SECTION ============ -->

---

<!-- ============ EXISTING SECTION: Updated timestamp ============ -->

## [18:30] Request Queue & Rate Limiting

**[18:30] [SLIDE: "Handling Traffic Spikes"]**

Protect your system from being overwhelmed.

[CODE: "request_queue.py"]
```python
import asyncio
from collections import deque
from typing import Callable, Any
import time

class RateLimitedQueue:
    """
    Request queue with rate limiting.
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        requests_per_minute: int = 60,
        queue_max_size: int = 1000
    ):
        self.max_concurrent = max_concurrent
        self.requests_per_minute = requests_per_minute
        self.queue_max_size = queue_max_size
        
        self.queue = deque()
        self.active_requests = 0
        self.request_timestamps = deque()
        
        self._lock = asyncio.Lock()
    
    async def enqueue(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> Any:
        """
        Add request to queue and wait for execution.
        """
        async with self._lock:
            if len(self.queue) >= self.queue_max_size:
                raise Exception("Queue is full")
            
            # Create request object
            request = {
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'priority': priority,
                'future': asyncio.Future()
            }
            
            # Add to queue (sorted by priority)
            self.queue.append(request)
            self.queue = deque(sorted(
                self.queue,
                key=lambda x: x['priority'],
                reverse=True
            ))
        
        # Wait for execution
        return await request['future']
    
    async def process_queue(self):
        """
        Background task to process queued requests.
        """
        while True:
            async with self._lock:
                # Check if we can process more requests
                if (self.active_requests < self.max_concurrent and
                    self.queue and
                    self._can_make_request()):
                    
                    request = self.queue.popleft()
                    self.active_requests += 1
                    self.request_timestamps.append(time.time())
                    
                    # Process request in background
                    asyncio.create_task(
                        self._execute_request(request)
                    )
            
            await asyncio.sleep(0.1)  # Check queue every 100ms
    
    async def _execute_request(self, request: dict):
        """Execute a single request."""
        try:
            result = await request['func'](
                *request['args'],
                **request['kwargs']
            )
            request['future'].set_result(result)
        except Exception as e:
            request['future'].set_exception(e)
        finally:
            async with self._lock:
                self.active_requests -= 1
    
    def _can_make_request(self) -> bool:
        """Check if we're within rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old timestamps
        while self.request_timestamps and self.request_timestamps[0] < minute_ago:
            self.request_timestamps.popleft()
        
        # Check if under limit
        return len(self.request_timestamps) < self.requests_per_minute
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'queue_size': len(self.queue),
            'active_requests': self.active_requests,
            'requests_last_minute': len(self.request_timestamps)
        }

# Example usage
class QueuedRAGSystem:
    """RAG system with request queuing."""
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.queue = RateLimitedQueue(
            max_concurrent=5,
            requests_per_minute=50
        )
        
        # Start queue processor
        asyncio.create_task(self.queue.process_queue())
    
    async def query(self, user_query: str, priority: int = 0) -> dict:
        """
        Process query through rate-limited queue.
        """
        result = await self.queue.enqueue(
            self._process_query,
            user_query,
            priority=priority
        )
        return result
    
    async def _process_query(self, query: str) -> dict:
        """Actual query processing."""
        # Your RAG logic here
        await asyncio.sleep(0.5)  # Simulate processing
        return {'response': f"Answer to: {query}"}
```

<!-- ============ END EXISTING SECTION ============ -->

---

<!-- ============ NEW SECTION: WHEN THIS BREAKS ============ -->

## [21:30] When This Breaks: Common Failures

**[21:30] [SLIDE: When Error Handling Goes Wrong]**

Now for the MOST important part: what happens when your error handling itself fails? Let me show you the 5 most common errors and how to debug them. These are the ones that will wake you up at 3 AM.

---

### Failure #1: Retry Storm Amplifying Problems (21:30-22:30)

**[TERMINAL] Let me reproduce this error:**

```bash
# Simulate 100 concurrent requests with 3 retries each
python reproduce_retry_storm.py
```

**Error message you'll see:**

```
OpenAIError: Rate limit exceeded (429)
Attempt 1/3 failed. Retrying in 1.2s...
Attempt 2/3 failed. Retrying in 2.8s...
Attempt 3/3 failed. Retrying in 5.1s...
[100 identical messages flood logs in 10 seconds]
Database connection pool exhausted: all 20 connections in use
```

**What this means:**

You hit OpenAI's rate limit (429). Each of your 100 requests retries 3 times. That's 400 total requests hammering the API in 10 seconds, making the problem worse. Meanwhile, your database connection pool is exhausted from all the hanging retries.

**How to fix it:**

[SCREEN] [CODE: retry_patterns.py]

```python
# Add exponential backoff WITH jitter
class RetryStrategy:
    def calculate_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
-       return delay
+       # Add jitter to prevent thundering herd
+       if self.jitter:
+           delay = delay * (0.5 + random.random() * 0.5)
+       return delay

# Also: Don't retry 429 errors more than once
    def should_retry(self, exception: Exception) -> bool:
        if hasattr(exception, 'status_code'):
+           # Rate limit? Back off aggressively
+           if exception.status_code == 429:
+               return self.current_attempt < 1  # Only 1 retry for rate limits
```

**How to verify:**

```bash
# Run again - retries should spread out over time
python reproduce_retry_storm.py
# Check logs: delays should vary (3.2s, 4.8s, 7.1s) not all identical
```

**How to prevent:**

Always enable jitter on retries. Set `max_retries=1` for rate limit errors specifically. Monitor "retry ratio" metric (retries / total requests). If >0.3, you have a problem.

---

### Failure #2: Circuit Breaker False Positives (22:30-23:30)

**[TERMINAL] Let me reproduce this error:**

```bash
# Simulate brief latency spike
python reproduce_circuit_false_positive.py
```

**Error message you'll see:**

```
2025-10-14 03:15:42 - Circuit 'openai': Failure 1/5
2025-10-14 03:15:43 - Circuit 'openai': Failure 2/5
2025-10-14 03:15:43 - Circuit 'openai': Failure 3/5
2025-10-14 03:15:44 - Circuit 'openai': OPEN after 3 failures
CircuitBreakerOpen: Circuit breaker 'openai' is OPEN.

[Next 60 seconds: All requests rejected even though service recovered]
User-facing error rate: 100% for 1 minute
```

**What this means:**

OpenAI had a 2-second latency spike (not uncommon). Your timeout is set to 1 second. Three requests timeout in a row, circuit opens. Now you're rejecting ALL requests for 60 seconds even though OpenAI recovered after 5 seconds. Your circuit breaker just caused a self-inflicted outage.

**How to fix it:**

[SCREEN] [CODE: circuit_breaker.py]

```python
# Increase failure threshold and reduce timeout
@dataclass
class CircuitBreakerConfig:
-   failure_threshold: int = 3  # Too aggressive!
+   failure_threshold: int = 5  # More forgiving
-   timeout: float = 60.0  # Too long!
+   timeout: float = 15.0  # Test recovery faster

# Also: Distinguish timeout from genuine failure
class CircuitBreaker:
    def _on_failure(self):
        with self._lock:
+           # Timeout? Only count as 0.5 failures
+           if isinstance(exception, TimeoutError):
+               self.failure_count += 0.5
+           else:
+               self.failure_count += 1
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
```

**How to verify:**

```bash
# Circuit should stay CLOSED through brief latency spikes
python test_circuit_resilience.py
```

**How to prevent:**

Set `failure_threshold >= 5` for external APIs. Use short timeout values (15-30s) so circuits test recovery faster. Monitor "circuit open duration" - if average >30s, threshold is too aggressive.

---

### Failure #3: Queue Memory Exhaustion (23:30-24:30)

**[TERMINAL] Let me reproduce this error:**

```bash
# Simulate traffic spike with slow processing
python reproduce_queue_overflow.py
```

**Error message you'll see:**

```
Queue size: 500... 750... 950... 1000
QueueFullError: Queue is full (1000/1000)
[Requests start failing immediately]

MemoryError: Cannot allocate memory
Process killed by OOM killer
Container restart loop detected
```

**What this means:**

Traffic spiked to 200 req/s but you can only process 50 req/s. Queue fills up in 5 seconds. New requests get rejected. Queue objects consume memory (each request object = ~5KB). 1000 requests = 5MB, but Python overhead means actual memory usage is 50-100MB. With multiple workers, this scales linearly.

**How to fix it:**

[SCREEN] [CODE: request_queue.py]

```python
class RateLimitedQueue:
    async def enqueue(self, func, *args, priority=0, **kwargs):
        async with self._lock:
            if len(self.queue) >= self.queue_max_size:
-               raise Exception("Queue is full")
+               # Implement load shedding based on priority
+               if priority < 5:  # Low priority requests get dropped
+                   raise QueueFullError("Queue full, low-priority request dropped")
+               
+               # For high priority: Drop oldest low-priority item
+               for i, req in enumerate(self.queue):
+                   if req['priority'] < 5:
+                       self.queue.remove(req)
+                       req['future'].set_exception(QueueFullError("Evicted"))
+                       break

# Also: Set memory-based limit, not just count
+   def _estimate_memory_usage(self) -> int:
+       """Estimate queue memory usage in MB."""
+       return len(self.queue) * 0.005  # ~5KB per request
+   
+   def _should_accept_request(self) -> bool:
+       return (len(self.queue) < self.queue_max_size and
+               self._estimate_memory_usage() < 50)  # Max 50MB
```

**How to verify:**

```bash
# Monitor memory during traffic spike
docker stats
# Queue should shed low-priority requests before OOM
```

**How to prevent:**

Set `queue_max_size` to (available_memory_MB * 0.8) / 0.005. Implement priority-based load shedding. Monitor "queue_depth" metric - alert if >70% capacity for >5 minutes.

---

### Failure #4: Graceful Degradation Gets Stuck (24:30-25:30)

**[TERMINAL] Let me reproduce this error:**

```bash
# Simulate transient failure that resolves
python reproduce_degradation_stuck.py
```

**Error message you'll see:**

```
2025-10-14 14:23:11 - Full service failed: TimeoutError. Degrading to DEGRADED.
2025-10-14 14:23:15 - Degraded service failed: TimeoutError. Going to MINIMAL.
[Service stays in MINIMAL for 6 hours even though upstream recovered after 2 minutes]

User complaints: "Responses are terrible quality since this afternoon"
Actual issue: Vector DB is fine, but we're still using keyword matching
```

**What this means:**

Your graceful degradation only moves DOWN service tiers (Full → Degraded → Minimal), never back UP. Vector DB had a 1-minute outage at 2pm. System degraded to MINIMAL. Now it's 8pm, vector DB is fine, but you're still serving crappy keyword-matched responses because nothing triggers recovery.

**How to fix it:**

[SCREEN] [CODE: graceful_degradation.py]

```python
class GracefulRAGSystem:
    def __init__(self, ...):
        self.current_tier = ServiceTier.FULL
+       self.last_degradation_time = None
+       self.recovery_check_interval = 60  # Check recovery every 60s
    
    def query(self, user_query: str) -> dict:
+       # Try to recover if we've been degraded for a while
+       if (self.current_tier != ServiceTier.FULL and 
+           self._should_attempt_recovery()):
+           self._attempt_recovery()
        
        # Existing logic...
+   
+   def _should_attempt_recovery(self) -> bool:
+       if self.last_degradation_time is None:
+           return False
+       elapsed = time.time() - self.last_degradation_time
+       return elapsed >= self.recovery_check_interval
+   
+   def _attempt_recovery(self):
+       """Try to move back up service tiers."""
+       try:
+           # Test if higher tier works
+           if self.current_tier == ServiceTier.MINIMAL:
+               self._test_degraded_tier()
+               self.current_tier = ServiceTier.DEGRADED
+           elif self.current_tier == ServiceTier.DEGRADED:
+               self._test_full_tier()
+               self.current_tier = ServiceTier.FULL
+           
+           logger.info(f"Recovered to {self.current_tier.value}")
+           self.last_degradation_time = None
+       except:
+           logger.warning("Recovery attempt failed, staying degraded")
```

**How to verify:**

```bash
# Simulate brief outage, verify system recovers
python test_degradation_recovery.py
# Check logs: should see "Recovered to FULL" within 60 seconds
```

**How to prevent:**

Always implement recovery logic, not just degradation. Set `recovery_check_interval` to 1-2 minutes. Monitor "current_service_tier" metric - alert if degraded for >10 minutes.

---

### Failure #5: Retry Logic Retrying Non-Retryable Errors (25:30-26:30)

**[TERMINAL] Let me reproduce this error:**

```bash
# Trigger authentication error
python reproduce_retrying_auth_error.py
```

**Error message you'll see:**

```
OpenAIError: Invalid API key (401 Unauthorized)
Attempt 1/3 failed. Retrying in 1.0s...
OpenAIError: Invalid API key (401 Unauthorized)
Attempt 2/3 failed. Retrying in 2.0s...
OpenAIError: Invalid API key (401 Unauthorized)
Attempt 3/3 failed. Retrying in 4.0s...

[Retries 3 times, each taking 1-4 seconds]
Total latency: 7 seconds to return an error that was immediate
User sees: Slow loading spinner, then "Invalid API key"
```

**What this means:**

Your API key expired or was revoked. This is a 401 error. Retrying won't help - if the key is invalid now, it'll be invalid in 2 seconds. But your retry logic retries ALL exceptions, wasting 7 seconds giving the user the same error.

**How to fix it:**

[SCREEN] [CODE: retry_patterns.py]

```python
class RetryStrategy:
    def should_retry(self, exception: Exception) -> bool:
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        
        if isinstance(exception, self.retryable_exceptions):
            if hasattr(exception, 'status_code'):
+               # NEVER retry auth errors (401, 403)
+               if exception.status_code in [401, 403]:
+                   logger.error(f"Auth error: {exception}")
+                   return False
                
-               # Don't retry 4xx errors (except 429)
+               # Don't retry client errors (4xx) except 429, 408
                if 400 <= exception.status_code < 500:
-                   return exception.status_code == 429
+                   return exception.status_code in [408, 429]  # Timeout, Rate limit
            return True
        
        return False

# Also: Make validation errors non-retryable
-   non_retryable_exceptions = ()
+   non_retryable_exceptions = (ValueError, TypeError, KeyError, AttributeError)
```

**How to verify:**

```bash
# 401 error should fail immediately without retries
python test_auth_error_handling.py
# Total time should be <100ms, not 7 seconds
```

**How to prevent:**

Never retry 4xx errors except 429 (rate limit) and 408 (request timeout). Add validation logic to catch bad inputs BEFORE calling external APIs. Monitor "non-retryable error rate" - if suddenly spikes, you likely have a config/auth issue.

---

**[26:00] [SLIDE: Error Prevention Checklist]**

To avoid these 5 errors:

- [ ] Enable jitter on all retries (prevents retry storms)
- [ ] Set circuit breaker thresholds ≥5 for external APIs (prevents false positives)
- [ ] Implement priority-based load shedding in queues (prevents OOM)
- [ ] Add recovery logic to graceful degradation (prevents getting stuck)
- [ ] Whitelist retryable errors, don't retry everything (prevents wasting time)

**[PAUSE]**

These are the errors that will actually happen in production. Bookmark this section.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ NEW SECTION: WHEN NOT TO USE ============ -->

## [26:30] When NOT to Use This

**[26:30] [SLIDE: When to AVOID This Approach]**

Let me be crystal clear about when you should NOT use the error handling patterns we just built. I see teams add this complexity when it's completely wrong for their use case.

**❌ Don't use this when:**

### 1. **Real-time or Ultra-Low Latency Requirements (<50ms SLA)**

- **Why it's wrong:** Retries add 50-200ms latency. Circuit breakers add 5-15ms overhead per request. If your SLA is sub-50ms (trading systems, gaming backends, real-time bidding), this destroys your latency budget.
- **Use instead:** Hot standby with immediate failover. Run duplicate systems in parallel, route to fastest response. Zero retry logic - if primary fails, secondary is already running.
- **Example:** High-frequency trading system needs <10ms order execution. Any retry logic is disqualifying. Run 3 identical systems, take first response, kill stragglers.

### 2. **Simple Internal Tools (<10 Users, Non-Critical)**

- **Why it's wrong:** You're adding 500+ lines of error handling code for a tool used by 6 people once a week. Implementation time (8 hours) exceeds total user time saved (maybe 1 hour/year). The complexity creates more bugs than it prevents.
- **Use instead:** Basic try/catch with logging. When it breaks, Slack alerts ops team, they restart manually. Takes 2 minutes, happens once a month.
- **Example:** Internal admin dashboard for updating product descriptions. 5 users, used twice a week. If OpenAI is down, they just wait 10 minutes and try again. Not worth circuit breakers.

### 3. **Batch Processing Where Failures Indicate Bugs**

- **Why it's wrong:** If your ETL pipeline fails, it's usually bad data or logic errors, not transient API issues. Retrying amplifies the problem - you'll process corrupted data 3 times before noticing. Retry logic masks bugs you need to fix.
- **Use instead:** Fail immediately on first error. Log complete error context (input data, stack trace). Fix root cause before rerunning job. Idempotent jobs mean rerunning from scratch is safe.
- **Example:** Nightly job that embeds 10K documents. If embedding fails, it's usually malformed text in input data. Retry logic just fails 3 times on same bad document. Better: fail fast, inspect the bad document, fix data quality issue.

**Red flags that you've chosen wrong approach:**

- 🚩 **Your retry ratio is >30%** - If you're retrying 30% of requests, you have systemic issues, not transient failures. Fix the root cause, don't mask it with retries.
- 🚩 **Circuit breakers open frequently (>1/hour)** - Your thresholds are too aggressive OR your upstream service is fundamentally unreliable. Either tune thresholds or switch services.
- 🚩 **Users complain responses are slow** - Error handling added too much latency. Measure p99 latency before/after adding retries. If it increased >100ms, reconsider.
- 🚩 **Your error handling code is >20% of total codebase** - You've over-engineered it. Simplify or use external orchestration (service mesh).

**[EMPHASIS]** The biggest mistake is adding error handling "because best practices say so" without measuring whether it helps YOUR use case. Always A/B test: deploy without retries to 10% of traffic, compare error rates and latency. If retries don't reduce errors by >50%, don't use them.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ NEW SECTION: DECISION CARD ============ -->

## [28:30] Decision Card: Error Handling Patterns

**[28:30] [SLIDE: Decision Card - Comprehensive Resilience]**

Let me summarize everything in one decision framework you can reference later:

### **✅ BENEFIT**

Reduces user-facing errors by 80-95% through automatic recovery from transient failures; handles 3 AM API outages without human intervention; prevents cascading failures from taking down entire system; enables graceful degradation so users get *something* instead of blank errors.

### **❌ LIMITATION**

Adds 50-200ms latency per request due to retry delays; circuit breakers add 5-15ms overhead even when working; can cause false positives if thresholds too aggressive (circuit opens when service is actually fine); increases system complexity by 20-30% (retry state, circuit breaker state machines, queue management); retry storms can amplify problems if not configured with jitter; queue memory grows linearly with traffic spikes (5KB per queued request).

### **💰 COST**

**Initial:** 6-8 hours implementation for basic patterns; 12-16 hours for production-grade with monitoring. **Ongoing:** 10-20% increased infrastructure cost (queue memory overhead, duplicate retry traffic, monitoring tools). **Complexity:** Adds 3 new failure modes to debug (circuit breaker false positives, queue overflow, degradation getting stuck). **Maintenance:** Weekly review of retry ratios, circuit breaker thresholds, queue depth metrics.

### **🤔 USE WHEN**

External APIs have >1% observed failure rate; user-facing application where downtime directly impacts revenue; acceptable to add 100-200ms p99 latency; team has monitoring infrastructure to track error handling metrics; traffic patterns are predictable enough to size queues; failures are primarily transient (network blips, rate limits) not systematic (bugs, bad data).

### **🚫 AVOID WHEN**

Need <50ms p99 latency → use hot standby redundancy instead; internal tool with <10 users → manual intervention cheaper than complexity; batch processing where failures indicate bugs → fail fast and fix root cause; traffic is extremely spiky (10x variance) and unpredictable → queues will overflow, consider rate limiting at API gateway layer; team lacks monitoring expertise → error handling without observability makes debugging harder.

**[PAUSE]** Take a screenshot of this slide - reference it when architecting your next system.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ NEW SECTION: PRODUCTION CONSIDERATIONS ============ -->

## [29:30] Production Considerations

**[29:30] [SLIDE: What Changes at Scale]**

What we built today works great for development and moderate production load. Here's what you need to consider when scaling to serious traffic.

**Scaling concerns:**

- **Circuit breaker coordination across instances:** Each application instance has its own circuit breaker state. If you have 20 instances, circuits open independently. This means 5% of requests might hit closed circuits while 95% hit open circuits, creating inconsistent user experience. **Mitigation:** Use distributed circuit breaker with Redis/Memcached to share state across instances. Or embrace inconsistency - it's acceptable for 95% availability.

- **Queue memory at high concurrency:** At 1000 req/s with 5s processing time, you need 5000 concurrent slots. With 5KB per request, that's 25MB per application instance. With 10 instances, 250MB total. This grows linearly. **Mitigation:** Set memory-based limits (`max_memory_mb`) not just count-based. Monitor container memory usage. Implement priority-based load shedding.

- **Retry amplification during incidents:** During an outage, retry traffic can be 3-5x normal traffic as requests retry. If outage affects 50% of requests and each retries 3 times, you're at 200% traffic. This can overload healthy services. **Mitigation:** Use exponential backoff with jitter. Set max_retries=2 not 5. Monitor retry ratio metric - if >0.3, disable retries temporarily.

**Cost at scale:**

Development (100 requests/day): ~$0, negligible overhead

**Production (1K users, 10K requests/day):**
- Queue memory overhead: $5-10/month (assumes 512MB avg)
- Duplicate retry traffic: 20-30% increased API costs ($15-30/month if base is $100/month)
- Monitoring tools (Prometheus/Grafana): $20-50/month
- **Total added cost: $40-90/month**

**Production (10K users, 100K requests/day):**
- Queue memory overhead: $30-60/month (assumes 2GB avg across instances)
- Duplicate retry traffic: 20-30% increased API costs ($200-300/month if base is $1000/month)
- Monitoring + alerting (PagerDuty): $100-200/month
- **Total added cost: $330-560/month**

**Break-even analysis:** Error handling infrastructure costs ~30% extra. If without it, your error rate is 5% and each error costs $2 in support time, you break even at: 100K requests/month × 0.05 error rate × $2 support cost = $10K/month saved. With our added cost of $560/month, ROI is 18:1.

**Monitoring requirements specific to error handling:**

You need these metrics or you're flying blind:
- `retry_ratio` (retries / total requests) - Alert if >0.3
- `circuit_breaker_state` per service - Alert if any circuit OPEN >5min
- `queue_depth_percent` - Alert if >70% for >5min
- `degradation_duration` (time spent in DEGRADED/MINIMAL) - Alert if >10min
- `false_positive_rate` (circuit opens when service healthy) - Track manually, tune thresholds if >1/day

**[EMPHASIS]** The number one production issue with error handling is: teams deploy it then never look at metrics. Set up alerts on day 1 or don't deploy it at all.

<!-- ============ END NEW SECTION ============ -->

---

<!-- ============ EXISTING SECTION: Updated timestamp and content ============ -->

## [31:30] Challenges & Wrap-Up

**[31:30] [SLIDE: "Practice Challenges"]**

Time to practice! Here are three challenges at different levels.

### 🟢 **EASY Challenge** (15-30 minutes)

**Task:** Implement a simple retry decorator that retries failed OpenAI API calls up to 3 times with exponential backoff. Include jitter.

**Success criteria:**
- [ ] Decorator works with any function
- [ ] Retries only on transient errors (not 4xx except 429)
- [ ] Implements exponential backoff with jitter
- [ ] Logs each retry attempt with delay time

**Hint:** Start with the `RetryStrategy` class from this video and simplify it. Test with a function that fails first 2 times then succeeds.

---

### 🟡 **MEDIUM Challenge** (30-60 minutes)

**Task:** Build a circuit breaker for your vector database. Test it by simulating failures and watching the circuit open/close. Add monitoring metrics.

**Success criteria:**
- [ ] Circuit breaker tracks failure/success counts
- [ ] Opens after 5 failures, tries HALF_OPEN after 30s
- [ ] Closes after 2 successful HALF_OPEN attempts
- [ ] Exposes metrics (state, failure_count, last_failure_time)
- [ ] Test script demonstrates state transitions

**Hint:** Simulate failures by raising exceptions in your vector DB search function. Use `time.sleep()` to test timeout behavior.

---

### 🔴 **HARD Challenge** (1-3 hours, portfolio-worthy)

**Task:** Create a "chaos engineering" test suite that randomly injects failures (timeouts, errors, slow responses) into your RAG system and verifies it handles them gracefully. Measure MTTR (Mean Time To Recovery).

**Success criteria:**
- [ ] Randomly injects 5+ failure types (timeout, 429, 503, connection error, slow response)
- [ ] Runs for 10 minutes with varying failure rates (0%, 5%, 20%, 50%)
- [ ] Measures user-facing error rate for each failure scenario
- [ ] Calculates MTTR (how long until system recovers)
- [ ] Generates report showing: resilience score, weakest component, recommended threshold tuning
- [ ] Includes test for circuit breaker false positives

**This is portfolio-worthy!** Share your chaos engineering report in Discord when complete. Include graphs of error rate vs failure injection rate.

**No hints - figure it out!** (But solutions will be provided in 48 hours)

---

**[32:30] [SLIDE: "Action Items"]**

Before Module 3, complete these:

**REQUIRED:**
1. [ ] Add retry logic to all external API calls (OpenAI, Pinecone, any HTTP requests)
2. [ ] Implement at least one circuit breaker (recommend: OpenAI client)
3. [ ] Define graceful degradation levels for your system (what happens when each component fails?)
4. [ ] Test failure scenarios manually (kill your vector DB, watch what happens)
5. [ ] Document your error handling strategy in README

**RECOMMENDED:**
1. [ ] Read: [Hystrix Wiki on Circuit Breakers](https://github.com/Netflix/Hystrix/wiki/How-it-Works) (Netflix's battle-tested patterns)
2. [ ] Experiment with different retry threshold values, measure impact on latency
3. [ ] Share your retry metrics in Discord (retry_ratio, avg_retry_delay)

**OPTIONAL:**
1. [ ] Research service mesh options (Istio, Linkerd) for comparison
2. [ ] Build a simple chaos monkey that randomly kills components

**Estimated time investment:** 2-4 hours for required items (this is critical, don't skip)

---

**[33:00] [SLIDE: "Module 2 Complete - Key Takeaways"]**

Let's recap what we covered across all of Module 2:

**Optimization:**
- Caching cuts costs 30-70% (semantic cache best ROI)
- Prompt optimization saves another 30-50% (shorter prompts = lower cost)
- Model selection matters - route intelligently (GPT-3.5 for 80% of queries)

**Monitoring:**
- Prometheus + Grafana = industry standard (battle-tested at scale)
- Structured logging makes debugging 10x easier (JSON logs with trace IDs)
- Alert on what matters (errors, latency, cost) - not vanity metrics

**Reliability:**
- Retry with exponential backoff + jitter (handles transient failures)
- Circuit breakers prevent cascading failures (isolate sick services)
- Graceful degradation keeps users happy (something > nothing)
- Rate limiting protects your infrastructure (queues prevent overload)
- **But:** Know when NOT to use these patterns (latency-sensitive systems, simple tools, systematic failures)

**[EMPHASIS] Critical lesson from today:** Error handling adds complexity. Only use it when the cost-benefit math makes sense. Measure retry ratios and circuit breaker metrics or you'll create more problems than you solve.

---

**[33:30] [SLIDE: "You Now Have a Production-Ready RAG System"]**

Congratulations! Your RAG system is now:
✅ Cost-optimized with multi-layer caching
✅ Prompt-engineered for efficiency  
✅ Fully monitored with metrics and logs
✅ Resilient to failures with circuit breakers
✅ Gracefully degrading under load
✅ Protected from retry storms and false positives
✅ Ready for production traffic

**But remember:** Production-ready doesn't mean perfect. It means:
- You know what breaks and how to fix it (5 failure modes covered)
- You have monitoring to detect problems (metrics + alerts)
- You can make informed decisions (Decision Card framework)
- You understand the limitations (Reality Check + When NOT to Use)

---

**[34:00] [SLIDE: "Coming Up in Module 3"]**

**Module 3: Advanced RAG Techniques**

We'll cover:
- Query rewriting and expansion (handle ambiguous questions)
- Hybrid search (keyword + semantic)
- Multi-step reasoning (complex questions need multiple LLM calls)
- Agent patterns (when to let LLM choose tools)
- Production deployment (Docker, K8s, monitoring at scale)

You've built a solid foundation in Module 2. Module 3 is where we make your RAG system truly intelligent.

See you in Module 3!

**[SLIDE: End Card with Course Branding]**

<!-- ============ END EXISTING SECTION ============ -->

---

# PRODUCTION NOTES (Creator-Only)

## Pre-Recording Checklist

- [ ] **Code tested:** All examples run without errors (especially new failure demonstrations)
- [ ] **Terminal clean:** Clear history, set up fresh session
- [ ] **Applications closed:** Only required apps open
- [ ] **Zoom/font set:** Code at 16-18pt, zoom level tested
- [ ] **Slides ready:** All slides in correct order (14 total now, including new sections)
- [ ] **Demo prepared:** Environment set up for live error demonstrations
- [ ] **Errors reproducible:** Tested all 5 common failures successfully
- [ ] **New sections rehearsed:** Reality Check, Alternative Solutions, When NOT to Use, Decision Card, Production Considerations (add ~13 min to recording time)
- [ ] **Timing practiced:** Full run-through completed (target: 32-34 minutes)
- [ ] **Water nearby:** Hydration important for longer videos!

## During Recording Guidelines

- **State video code clearly:** "M2.4: Error Handling & Reliability"
- **Pace for note-taking:** Slower after complex concepts, especially in new sections
- **Pause meaningfully:** Use [PAUSE] and [WAIT] as marked
- **Zoom on key code:** Highlight important lines, especially diff markers in failure demonstrations
- **Read Decision Card fully:** All 5 fields on screen for 5+ seconds
- **Show actual errors:** Don't just talk about them - reproduce in [TERMINAL]
- **Stay energetic:** Especially during longer sections (new script is 32 min vs original 18 min)
- **Acknowledge difficulty:** "Circuit breakers are complex" builds trust
- **Emphasize trade-offs:** Reality Check and Limitations are KEY sections - don't rush

## Post-Recording Checklist

- [ ] **Review footage:** Check for audio/video issues
- [ ] **Mark timestamps:** Note actual times for editing (especially new sections)
- [ ] **Verify code visible:** All code on screen was readable
- [ ] **Check audio quality:** No background noise/echo
- [ ] **List corrections:** Note any mistakes for annotations
- [ ] **Identify cuts:** Mark sections to trim in editing
- [ ] **Verify new sections:** Reality Check, Alternative Solutions, When NOT to Use, Decision Card, Production Considerations all recorded clearly

## Editing Notes

- **Intro:** Can tighten by 10-15% without losing content
- **Reality Check (NEW):** Keep all content - this sets honest tone
- **Alternative Solutions (NEW):** Keep decision framework diagram on screen for full duration
- **Concept sections (Retry, Circuit Breaker, etc.):** Don't cut - students code along
- **When This Breaks (NEW):** All 5 failures critical - keep complete
- **When NOT to Use (NEW):** Don't rush this - it's high-value content
- **Decision Card (NEW):** Must be readable on screen for 5+ seconds, consider showing as standalone card at end
- **Production Considerations (NEW):** Cost numbers important - ensure visible
- **Challenges:** Can be shorter on screen if in description

---

# GATE TO PUBLISH (Deliverables)

## Code & Technical

- [ ] **Code committed to repo:** All files including new error reproduction scripts
- [ ] **Code tested:** Runs on fresh environment
- [ ] **Dependencies documented:** requirements.txt includes `prometheus-client` if used
- [ ] **README included:** Setup and run instructions
- [ ] **Error scenarios verified:** All 5 common failures reproducible
- [ ] **Reproduction scripts:** Scripts for demonstrating each failure mode

## Video & Assets

- [ ] **Video rendered:** Final version exported (~32-34 minutes)
- [ ] **Captions added:** Subtitles for accessibility
- [ ] **Slides exported:** PDF version available for download (14 slides total now)
- [ ] **Timestamps in description:** All major sections marked (including new sections)
- [ ] **Links working:** All external resources verified
- [ ] **Decision Card graphic:** Standalone PNG/PDF export for reference

## Educational Materials

- [ ] **Challenge solutions prepared:** All 3 levels solved
- [ ] **FAQ document:** Anticipated questions answered (especially about new sections)
- [ ] **Alternative solution examples:** Code for fail-fast approach, service mesh comparison
- [ ] **Decision Card exported:** Standalone reference graphic (high-quality)
- [ ] **Trade-off comparison table:** Reality Check section as downloadable table

## Platform Setup

- [ ] **Video uploaded:** To hosting platform
- [ ] **Description complete:** Including all links and timestamps (updated for 32-34 min duration)
- [ ] **Resources attached:** Code, slides, Decision Card, references
- [ ] **Discord announcement:** Posted in appropriate channel
- [ ] **Prerequisites verified:** Previous videos (M2.1, M2.2, M2.3) accessible

## Quality Assurance (TVH Framework v2.0 Compliance)

- [ ] **Honest teaching verified:** Limitations covered adequately in Reality Check
- [ ] **Decision Card complete:** All 5 fields populated (✅ ❌ 💰 🤔 🚫)
- [ ] **Alternatives discussed:** 3 options presented (Full Resilience, Fail Fast, Service Mesh)
- [ ] **Failures covered:** 5 common errors shown with reproduction
- [ ] **When NOT to use:** Explicitly stated with 3 scenarios
- [ ] **Production considerations:** Scaling/cost addressed with numbers
- [ ] **Word counts verified:** Reality Check (250 words), Alternative Solutions (260 words), When NOT to Use (195 words), Decision Card (115 words), Production Considerations (210 words)

---

## Version History

- **v1.0:** Initial 18-minute script with code patterns only
- **v2.0:** Enhanced 32-minute script with TVH Framework v2.0 compliance (Reality Check, Alternative Solutions, When This Breaks with 5 failures, When NOT to Use, Decision Card, Production Considerations)

---

## Notes on Changes from v1.0 to v2.0

### Added Sections (13 minutes total):
1. **Reality Check (2.5 min):** Honest limitations discussion at [1:30-4:00]
2. **Alternative Solutions (2.5 min):** Three approaches compared at [4:00-6:30]
3. **When This Breaks (5 min):** Five failure modes with reproduction at [21:30-26:30]
4. **When NOT to Use (2 min):** Anti-patterns at [26:30-28:30]
5. **Decision Card (1 min):** Summary framework at [28:30-29:30]
6. **Production Considerations (2 min):** Scaling/cost at [29:30-31:30]

### Preserved Content:
All original code sections maintained (Retry Patterns, Circuit Breaker, Graceful Degradation, Request Queue) - just shifted timestamps

### Enhanced Elements:
- Challenges section now references honest teaching concepts
- Wrap-up emphasizes limitations and Decision Card
- Action items include measuring retry ratios
- Production notes updated for longer recording

**The augmented script maintains the original's technical depth while adding TVH Framework's honest teaching requirements.**