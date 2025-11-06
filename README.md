# M2.1 — Caching Strategies for Cost Reduction

Reduce RAG system costs by 30-70% with multi-layer Redis caching. This module teaches practical caching strategies, invalidation techniques, and failure patterns to avoid.

## 🎯 Learning Objectives

- Deploy a multi-layer Redis caching system reducing RAG costs by 30-70%
- Configure cache invalidation based on content freshness requirements
- Diagnose and resolve five common production failures
- Recognize scenarios where caching creates more problems than solutions

## 📦 What's Included

- **`M2_1_Caching_Strategies.ipynb`** - Interactive tutorial with 8 sections
- **`m2_1_caching.py`** - Production-ready multi-layer cache implementation
- **`config.py`** - Environment and client configuration
- **`tests_caching.py`** - Smoke tests for key functionality
- **`example_data.txt`** - Sample FAQ queries for testing

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Redis

**Option A: Docker (Recommended)**
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

**Option B: Redis Cloud**
- Sign up at [redis.com/try-free](https://redis.com/try-free)
- Get your connection URL

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
# - OPENAI_API_KEY=sk-...
# - REDIS_URL=redis://localhost:6379/0
```

### 4. Run Notebook

```bash
jupyter notebook M2_1_Caching_Strategies.ipynb
```

### 5. Run Tests

```bash
python tests_caching.py
```

## 🏗️ Architecture

### Three-Layer Design

**Layer 1: Query Cache (Exact + Semantic)**
- Exact match via SHA-256 hash
- Semantic match via fuzzy string similarity (rapidfuzz)
- Stores final LLM responses

**Layer 2: Embedding Cache**
- Caches vector embeddings (1536 dims = ~6KB each)
- Reduces OpenAI API calls for repeated text
- TTL: 2 hours (embeddings rarely change)

**Layer 3: Retrieved-Context Cache**
- Caches document snippets fetched from vector DB
- Keyed by sorted document IDs
- Multiple queries often retrieve same documents

### Request Flow

```
Query → Exact Cache? → Semantic Cache? → Embedding Cache?
        → Vector DB → Context Cache? → LLM → Cache Result
```

## 🔧 How Cache Layers Work

### Exact Cache
```python
from m2_1_caching import MultiLayerCache
import config

cache = MultiLayerCache(config.get_redis(), config.get_openai())

# Check exact match
cached = cache.get_exact("How do I reset my password?")
if not cached:
    # Process query...
    response = {"answer": "Visit settings..."}
    cache.set_exact(query, response)
```

### Semantic Cache
```python
# Similar queries hit cache
cache.set_semantic("What are your hours?", {"answer": "9-5 EST"})

# This will hit cache even with different wording
result = cache.get_semantic("What time are you open?", threshold=0.85)
```

### Embedding Cache
```python
# Automatic caching with stampede protection
embedding = cache.compute_or_get_embedding("machine learning")
# Second call hits cache instantly
embedding2 = cache.compute_or_get_embedding("machine learning")
```

### Context Cache
```python
# Cache retrieved documents
doc_ids = ["doc_123", "doc_456"]
contexts = [{"id": "doc_123", "text": "..."}]
cache.set_context(doc_ids, contexts)

# Retrieves even with different order
cache.get_context(["doc_456", "doc_123"])  # Cache HIT
```

## 🗑️ Invalidation Strategies

### 1. TTL (Time-To-Live)
Automatic expiration via Redis:
```python
cache.set_exact(query, response, ttl=3600)  # 1 hour
```

### 2. Manual Invalidation
Explicit cache busting:
```python
# Invalidate specific query
cache.invalidate_query("How do I reset my password?")

# Clear entire cache layer
cache.invalidate_by_prefix(config.PREFIX_SEMANTIC)
```

### 3. Stale Detection
Timestamp-based cleanup:
```python
# Remove entries older than 1 hour
cache.invalidate_stale(max_age_seconds=3600)
```

### 4. Full Flush
```python
# Clear all cache layers (use with caution)
cache.flush_all()
```

## ⚠️ When NOT to Use Caching

Caching becomes counterproductive when:

1. **Query diversity exceeds 90%** - Hit rates will be <10%, wasting resources
2. **Content updates required within 5-minute windows** - Stale data issues
3. **Traffic below 500 daily queries** - Infrastructure overhead not worth it
4. **Single-server deployments with low volumes** - In-memory caching simpler

### Decision Matrix

| Scenario | Daily Queries | Diversity | Freshness Need | Recommendation |
|----------|---------------|-----------|----------------|----------------|
| FAQ Bot | 5,000+ | <30% | >1 hour | ✅ Cache |
| News Search | 10,000+ | >90% | <5 min | ❌ Don't Cache |
| Support Docs | 2,000+ | 40% | >30 min | ✅ Cache |
| Research Q&A | 500 | 85% | Any | ❌ Don't Cache |

## 🐛 Common Failures & Fixes

### 1. Cache Stampede
**Problem:** Concurrent requests overwhelm backend
**Fix:** Per-key locks (implemented via `StampedeLock`)

### 2. Stale Data
**Problem:** Updates not reflected until TTL expires
**Fix:** Implement immediate invalidation on content updates

### 3. Memory Exhaustion
**Problem:** Large embeddings fill Redis
**Fix:** Configure `maxmemory` and LRU eviction in Redis

### 4. Hash Collisions
**Problem:** Wrong results from weak hashes
**Fix:** Use full SHA-256 (no truncation)

### 5. Low ROI
**Problem:** <20% hit rate wastes infrastructure
**Fix:** Monitor metrics, disable caching if hit rate stays low

## 📊 Monitoring Metrics

```python
# Check cache performance
print(cache.metrics.summary())

# Get hit rate
hit_rate = cache.metrics.get_hit_rate()
if hit_rate < 20:
    print("⚠️ Consider disabling cache - ROI too low")
```

## 🔍 Troubleshooting

### Redis Connection Errors

**Error:** `ConnectionError: Error 111 connecting to localhost:6379`

**Solutions:**
1. Check if Redis is running: `redis-cli ping`
2. Verify REDIS_URL in `.env`
3. For Docker: `docker ps` to confirm container is running
4. Check firewall/port accessibility

### OpenAI API Errors

**Error:** `AuthenticationError: Incorrect API key`

**Solutions:**
1. Verify `OPENAI_API_KEY` in `.env`
2. Check API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Ensure no extra whitespace in `.env` file

### Low Hit Rates

**Problem:** Cache hit rate below 20%

**Diagnosis:**
```python
# Analyze query diversity
queries = ["query1", "query2", ...]
unique_count = len(set(queries))
diversity = unique_count / len(queries)
print(f"Diversity: {diversity*100:.0f}%")
```

**Solutions:**
- If diversity >90%: Disable caching
- If diversity 50-90%: Adjust semantic threshold
- If diversity <50%: Increase TTLs

### Memory Issues

**Error:** Redis OOM (Out of Memory)

**Solutions:**
1. Configure Redis `maxmemory` policy:
   ```
   maxmemory 256mb
   maxmemory-policy allkeys-lru
   ```
2. Reduce embedding cache TTL
3. Disable context cache for large documents
4. Monitor with: `redis-cli INFO memory`

## 🧪 Testing

Run smoke tests to verify functionality:

```bash
python tests_caching.py
```

Tests verify:
- ✅ Key schema stability (SHA-256)
- ✅ Stampede lock prevents concurrent compute
- ✅ TTL expiration works correctly
- ✅ Semantic cache matches paraphrases
- ✅ Safe stubs when services unavailable
- ✅ Metrics tracking accuracy
- ✅ Context cache order independence

## 📚 Configuration Reference

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0

# Cache Layer Toggles
ENABLE_EXACT_CACHE=true
ENABLE_SEMANTIC_CACHE=true
ENABLE_EMBEDDING_CACHE=true
ENABLE_CONTEXT_CACHE=true

# TTL Settings (seconds)
TTL_EXACT_CACHE=3600        # 1 hour
TTL_SEMANTIC_CACHE=1800     # 30 minutes
TTL_EMBEDDING_CACHE=7200    # 2 hours
TTL_CONTEXT_CACHE=1800      # 30 minutes

# Semantic Matching
SEMANTIC_THRESHOLD=0.85     # 0.0 - 1.0

# Redis Connection Pool
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
```

### Cache Key Prefixes

- `exact:` - Exact query matches (SHA-256)
- `semantic:` - Semantic similarity bucket
- `embed:` - Vector embeddings
- `context:` - Retrieved document contexts

## 📈 Cost Projection

Example savings for 10,000 queries/day:

| Scenario | Hit Rate | Daily Cost (No Cache) | Daily Cost (With Cache) | Monthly Savings |
|----------|----------|----------------------|------------------------|-----------------|
| FAQ Bot | 60% | $30.00 | $12.33 | $529.10 |
| Support | 40% | $30.00 | $18.33 | $350.10 |
| High Diversity | 10% | $30.00 | $27.33 | $80.10 |

Assumes:
- 1,500 tokens per query average
- $0.002 per 1K tokens (GPT-3.5-turbo)
- $10/month Redis Cloud basic tier

## 🤝 Contributing

Found a bug or have suggestions? Please open an issue or submit a PR.

## 📄 License

This module is part of the RAG21D learner series.

## 🔗 Related Modules

- **M2.2** - Query Optimization & Prompt Engineering
- **M2.3** - Model Selection & Cost-Performance Trade-offs
- **M3.1** - Chunking Strategies & Vector Databases

---

**Ready to reduce your RAG costs?** Open `M2_1_Caching_Strategies.ipynb` and start learning! 🚀
