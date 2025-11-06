# Scaling Notes: RAG System Performance & Infrastructure

## Horizontal vs Vertical Scaling

### Horizontal Scaling (Scale Out)
**Definition**: Add more instances of your application behind a load balancer.

**Advantages**:
- Nearly unlimited scaling potential
- Better fault tolerance (one instance fails, others continue)
- Cost-effective for cloud platforms (add/remove instances dynamically)

**Requirements**:
- **Stateless application design** - no local session storage
- Load balancer to distribute traffic (nginx, HAProxy, cloud LB)
- Shared state storage (Redis, database)
- Session management via external store or JWT tokens

**Best for**: Web applications, API services, microservices

### Vertical Scaling (Scale Up)
**Definition**: Increase CPU, RAM, or disk on a single instance.

**Advantages**:
- Simpler deployment (no distributed system complexity)
- No code changes required
- Good for database servers and stateful services

**Limitations**:
- Physical hardware limits (can't scale infinitely)
- Downtime during upgrades
- Single point of failure
- More expensive at high scales

**Best for**: Databases, caching layers, single-tenant applications

---

## Health Checks

Health checks enable automatic failure detection and recovery.

### Types of Health Checks

1. **Liveness Probe**: "Is the application running?"
   ```python
   @app.get("/health")
   def health_check():
       return {"status": "ok"}
   ```

2. **Readiness Probe**: "Is the application ready to serve traffic?"
   ```python
   @app.get("/ready")
   def readiness_check():
       # Check database connection
       db.execute("SELECT 1")
       # Check external dependencies
       redis.ping()
       return {"status": "ready"}
   ```

3. **Startup Probe**: For slow-starting applications
   ```python
   @app.get("/startup")
   def startup_check():
       # Check if models loaded, embeddings ready
       if not embeddings_loaded:
           raise HTTPException(status_code=503)
       return {"status": "started"}
   ```

### Configuration Best Practices
- **Timeout**: 5-10 seconds (longer than typical request)
- **Interval**: Every 10-30 seconds
- **Failure threshold**: 3 consecutive failures before restart
- **Success threshold**: 1 success to mark healthy

---

## Load Balancing Strategies

### Round Robin
- Default strategy: distribute requests evenly across instances
- Simple but doesn't account for instance load

### Least Connections
- Route to instance with fewest active connections
- Better for varying request durations (e.g., RAG queries)

### IP Hash
- Same client always routed to same instance
- Useful for sticky sessions (avoid if possible)

### Weighted Round Robin
- Assign weights based on instance capacity
- Use when instances have different hardware specs

### Implementation Options
- **Cloud providers**: AWS ALB, GCP Load Balancer, Railway/Render built-in
- **Self-hosted**: nginx, HAProxy, Traefik
- **API Gateway**: Kong, AWS API Gateway (adds auth, rate limiting)

---

## Caching Strategies

Caching reduces load on expensive operations (embeddings, LLM calls, database queries).

### 1. Query Result Caching
**Cache**: Full RAG query responses
**Key**: Hash of query text + parameters
**TTL**: 1-24 hours (depends on data freshness requirements)
**Impact**: 10-100x speedup for repeated queries

```python
cache_key = f"query:{hash(query_text)}:{top_k}"
if cached := redis.get(cache_key):
    return cached
result = run_rag_query(query_text, top_k)
redis.set(cache_key, result, ex=3600)  # 1 hour TTL
```

### 2. Embedding Caching
**Cache**: Document embeddings
**Key**: Document ID or content hash
**TTL**: Indefinite (invalidate on document update)
**Impact**: Eliminate redundant embedding API calls

### 3. Retrieval Caching
**Cache**: Vector search results
**Key**: Hash of query embedding + top_k
**TTL**: 5-60 minutes
**Impact**: Reduce vector DB load for similar queries

### Cache Eviction Policies
- **LRU** (Least Recently Used): Default for most use cases
- **LFU** (Least Frequently Used): For stable query patterns
- **TTL-based**: Fixed expiration time

### Cache Warm-up
- Pre-populate cache with common queries at startup
- Run background job to refresh frequently accessed items

---

## Batching Optimizations

Batching reduces overhead and increases throughput 2-10x.

### 1. Embedding Batching
**Problem**: Embedding API calls have overhead per request
**Solution**: Batch multiple texts in single API call

```python
# Bad: 10 API calls for 10 documents
for doc in documents:
    embedding = openai.embed(doc)

# Good: 1 API call for 10 documents
embeddings = openai.embed(documents)  # Batch request
```

**Impact**: 5-10x reduction in API calls and latency

### 2. Database Batching
**Problem**: N+1 query problem (loop with individual queries)
**Solution**: Use bulk operations

```python
# Bad: N queries
for doc_id in doc_ids:
    doc = db.query(Document).filter_by(id=doc_id).first()

# Good: 1 query
docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
```

### 3. Request Coalescing
**Problem**: Multiple identical requests in flight simultaneously
**Solution**: Deduplicate in-flight requests

```python
# If same query arrives while processing, wait for existing result
if query_key in processing_queries:
    return await processing_queries[query_key]
processing_queries[query_key] = process_query(query)
result = await processing_queries[query_key]
del processing_queries[query_key]
return result
```

---

## Connection Pooling

### Database Connection Pooling
**Problem**: Creating new connections is expensive (100-500ms)
**Solution**: Maintain pool of reusable connections

```python
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Base connections kept open
    max_overflow=40,        # Extra connections under load
    pool_timeout=60,        # Wait time before error
    pool_recycle=3600,      # Recycle connections every hour
    pool_pre_ping=True      # Verify connection before use
)
```

**Rule of thumb**: `pool_size = concurrent_users / 10`

### Redis Connection Pooling
```python
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=pool)
```

---

## Auto-scaling Configuration

### Metrics-Based Scaling
- **CPU Usage**: Scale up when >70% for 5 minutes
- **Memory Usage**: Scale up when >80% sustained
- **Request Queue Depth**: Scale up when >100 queued requests
- **Custom Metrics**: p95 latency >2 seconds

### Scaling Parameters
```yaml
# Example: Railway/Render auto-scaling config
scaling:
  min_instances: 2        # Always maintain minimum for redundancy
  max_instances: 10       # Cost control limit
  scale_up_delay: 60s     # Wait before adding instances
  scale_down_delay: 300s  # Wait 5 min before removing (avoid flapping)
  target_cpu: 70%
  target_memory: 80%
```

### Scaling Best Practices
- **Gradual scaling**: Don't jump from 2 to 10 instances instantly
- **Pre-scaling**: Scale up before traffic surge if predictable
- **Cool-down periods**: Prevent rapid scaling oscillation
- **Cost monitoring**: Set alerts for unexpected scaling events

---

## Circuit Breaker Pattern

Prevent cascading failures when external services are down.

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_api():
    response = requests.get("https://external-api.com/data")
    return response.json()
```

**How it works**:
1. **Closed**: Normal operation, requests flow through
2. **Open**: After N failures, stop calling (fail fast)
3. **Half-open**: After timeout, try one request to test recovery

---

## Graceful Degradation Strategies

When under load, maintain partial functionality:

### 1. Reduce top_k
- Retrieve fewer documents (5 → 3)
- Faster retrieval, lower quality

### 2. Skip LLM Generation
- Return raw retrieved documents
- 10x faster, no synthesis

### 3. Serve from Cache Only
- Reject novel queries under extreme load
- Maintain service for cached queries

### 4. Request Prioritization
- VIP users get full service
- Free tier users get degraded service

---

## When to Scale: Decision Matrix

| Symptom | Likely Cause | Scaling Solution |
|---------|--------------|------------------|
| High CPU, low memory | CPU-bound (embeddings) | Vertical scale or GPU instances |
| Low CPU, high memory | Memory leak or large objects | Fix code, then horizontal scale |
| Database slow | DB at capacity | Vertical scale DB or read replicas |
| High p99, normal p50 | Outlier requests or cold starts | Add more instances (horizontal) |
| 429 rate limits | External API throttling | Implement caching, not scaling |

---

## Cost vs Performance Trade-offs

### Caching
- **Cost**: Redis hosting ($10-50/month)
- **Savings**: Reduce OpenAI API calls by 80% ($200+ savings)
- **ROI**: Pays for itself immediately

### Horizontal Scaling
- **Cost**: 2x instances = 2x hosting cost ($40 → $80/month)
- **Benefit**: 2x capacity, better uptime
- **When**: When vertical scaling limits reached

### Database Optimization
- **Cost**: Developer time (4-8 hours)
- **Benefit**: 5-10x query speedup, no infrastructure cost
- **Priority**: Always optimize before scaling

---

## Monitoring & Alerting

### Essential Metrics
1. **Request Rate**: Requests per second (RPS)
2. **Latency Percentiles**: p50, p95, p99
3. **Error Rate**: 4xx, 5xx responses
4. **Resource Utilization**: CPU, memory, disk
5. **External Dependencies**: API latency, rate limits

### Alert Thresholds
- **Critical**: p99 latency >5s or error rate >5%
- **Warning**: CPU >80% sustained or memory >85%
- **Info**: Approaching rate limits (>80% of quota)

### Tools
- **Self-hosted**: Prometheus + Grafana
- **Cloud**: DataDog, New Relic, CloudWatch
- **Free tier**: Railway/Render built-in metrics

---

## Summary: Scaling Playbook

1. **Measure first**: Run load tests to find bottlenecks
2. **Optimize code**: Fix N+1 queries, add caching (often 10x wins)
3. **Vertical scale**: Increase instance size (quick, limited)
4. **Horizontal scale**: Add instances behind load balancer (unlimited)
5. **Auto-scale**: Configure metric-based scaling for elasticity
6. **Monitor**: Track metrics continuously, alert on degradation

**Golden rule**: Premature optimization wastes time. Load test to find real bottlenecks, then scale strategically.
