# M2.1 — Caching Strategies for Cost Reduction

## 📚 Learning Arc

### **Purpose**
Learn to deploy multi-layer Redis caching to reduce RAG system costs by 30-70% through strategic query, embedding, and context caching. Master cache invalidation, stampede protection, and recognize when caching becomes counterproductive.

### **Concepts Covered**
- Multi-layer cache architecture (exact, semantic, embedding, context)
- Hash-based exact matching with SHA-256 for collision-free keys
- Semantic similarity caching using fuzzy string matching (BM25/MinHash)
- Cache stampede prevention with per-key distributed locks
- TTL (Time-To-Live) strategies for different cache layers
- Invalidation patterns: manual, prefix-based, stale detection
- Metrics tracking and ROI (Return on Investment) analysis
- Trade-offs: query diversity vs. hit rates, freshness vs. staleness
- Production failure modes: stampedes, memory exhaustion, hash collisions

### **After Completing This Module**
You will be able to:
- Implement a production-ready multi-layer cache for RAG systems
- Configure appropriate TTLs based on content freshness requirements
- Diagnose and resolve common caching failures (stampedes, stale data, low ROI)
- Calculate cost savings and project ROI for caching infrastructure
- Recognize scenarios where caching creates more problems than it solves
- Design invalidation strategies that balance freshness and performance
- Integrate caching into existing RAG pipelines without code changes

### **Context in Track**
This is Module 2.1 in the RAG optimization track. Prerequisites include basic RAG system understanding (M1.x modules). This module feeds into:
- M2.2: Query Optimization & Prompt Engineering
- M2.3: Model Selection & Cost-Performance Trade-offs
- M3.1: Chunking Strategies & Vector Databases

---

## 📦 Project Structure

```
rag21d_learners/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── app.py                      # FastAPI application (thin wrapper)
│
├── src/
│   └── m2_1_caching/
│       ├── __init__.py         # Package exports + learning arc
│       ├── config.py           # Configuration & client setup
│       ├── module.py           # Core multi-layer cache implementation
│       └── router.py           # FastAPI endpoints
│
├── notebooks/
│   └── M2_1_Caching_Strategies.ipynb  # Interactive tutorial
│
├── tests/
│   ├── test_caching.py         # Core functionality tests
│   └── test_smoke.py           # API endpoint tests
│
├── data/
│   └── example/
│       └── example_data.txt    # Sample FAQ queries
│
└── scripts/
    └── run_local.ps1           # Windows development server script
```

---

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

### 4. Run the API Server

**Linux/Mac:**
```bash
export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
.\scripts\run_local.ps1
```

**Windows (Command Line):**
```cmd
set PYTHONPATH=%CD%
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/m2_1_caching/health

### 5. Run Jupyter Notebook

```bash
jupyter notebook notebooks/M2_1_Caching_Strategies.ipynb
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 Usage Examples

### Library Usage

```python
from src.m2_1_caching.module import MultiLayerCache
from src.m2_1_caching.config import get_redis, get_openai

# Initialize cache
cache = MultiLayerCache(get_redis(), get_openai())

# Check exact cache
query = "How do I reset my password?"
result = cache.get_exact(query)
if not result:
    # Process query...
    result = {"answer": "Visit settings > security > reset password"}
    cache.set_exact(query, result)

# Semantic cache (matches similar queries)
cache.set_semantic("What are your hours?", {"answer": "9-5 EST"})
result = cache.get_semantic("What time are you open?", threshold=0.85)

# Embedding cache with stampede protection
embedding = cache.compute_or_get_embedding("machine learning")

# Context cache (document retrieval)
doc_ids = ["doc_123", "doc_456"]
contexts = [{"id": "doc_123", "text": "..."}]
cache.set_context(doc_ids, contexts)
```

### API Usage

**Get Cache Metrics:**
```bash
curl http://localhost:8000/m2_1_caching/metrics
```

Response:
```json
{
  "hits": 42,
  "misses": 18,
  "hit_rate": 70.0,
  "stampede_prevented": 3,
  "invalidations": 5,
  "errors": 0,
  "summary": "Hits: 42, Misses: 18, Hit Rate: 70.0%, ..."
}
```

**Invalidate Cache by Prefix:**
```bash
curl -X POST http://localhost:8000/m2_1_caching/invalidate \
  -H "Content-Type: application/json" \
  -d '{"prefix":"semantic:"}'
```

Response:
```json
{
  "prefix": "semantic:",
  "keys_invalidated": 127,
  "message": "Invalidated 127 keys with prefix 'semantic:'"
}
```

**Health Check:**
```bash
curl http://localhost:8000/m2_1_caching/health
```

Response:
```json
{
  "status": "ok",
  "module": "m2_1_caching",
  "redis_available": true,
  "openai_available": true
}
```

### CLI Usage

```bash
python -m src.m2_1_caching.module
```

Output:
```
M2.1 Multi-Layer Caching System
================================

Initializing clients...
✓ Redis connected
✓ OpenAI configured

Cache metrics: Hits: 0, Misses: 0, Hit Rate: 0.0%, ...
```

---

## 🏗️ Architecture

### Three-Layer Design

**Layer 1: Query Cache (Exact + Semantic)**
- Exact match via SHA-256 hash
- Semantic match via fuzzy string similarity (rapidfuzz)
- Stores final LLM responses
- TTL: 1 hour (exact), 30 minutes (semantic)

**Layer 2: Embedding Cache**
- Caches vector embeddings (1536 dims = ~6KB each)
- Reduces OpenAI API calls for repeated text
- TTL: 2 hours (embeddings rarely change)
- Includes stampede protection

**Layer 3: Retrieved-Context Cache**
- Caches document snippets fetched from vector DB
- Keyed by sorted document IDs (order-independent)
- Multiple queries often retrieve same documents
- TTL: 30 minutes

### Request Flow

```
Query → Exact Cache? → Semantic Cache? → Embedding Cache?
        → Vector DB → Context Cache? → LLM → Cache Result
```

---

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

---

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

---

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

---

## 📊 Monitoring Metrics

```python
# Check cache performance
print(cache.metrics.summary())

# Get hit rate
hit_rate = cache.metrics.get_hit_rate()
if hit_rate < 20:
    print("⚠️ Consider disabling cache - ROI too low")
```

---

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

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Set PYTHONPATH before running:
```bash
export PYTHONPATH=$PWD  # Linux/Mac
set PYTHONPATH=%CD%     # Windows CMD
$env:PYTHONPATH="$PWD"  # Windows PowerShell
```

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test suite:
```bash
pytest tests/test_caching.py -v   # Core functionality
pytest tests/test_smoke.py -v     # API endpoints
```

Tests verify:
- ✅ Key schema stability (SHA-256)
- ✅ Stampede lock prevents concurrent compute
- ✅ TTL expiration works correctly
- ✅ Semantic cache matches paraphrases
- ✅ Safe stubs when services unavailable
- ✅ Metrics tracking accuracy
- ✅ Context cache order independence
- ✅ API endpoints return correct responses

---

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

---

## 🤝 Contributing

Found a bug or have suggestions? Please open an issue or submit a PR.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Related Modules

- **M2.2** - Query Optimization & Prompt Engineering
- **M2.3** - Model Selection & Cost-Performance Trade-offs
- **M3.1** - Chunking Strategies & Vector Databases

---

**Ready to reduce your RAG costs?** Start with the [Jupyter notebook](notebooks/M2_1_Caching_Strategies.ipynb) or fire up the [API server](#4-run-the-api-server)! 🚀
