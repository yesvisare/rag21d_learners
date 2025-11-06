# M1.2 — Pinecone Data Model & Advanced Indexing

**Hybrid Search, Namespaces, Failures, Decision Framework**

Complete implementation of production-ready hybrid search combining dense (semantic) and sparse (keyword) vector retrieval with Pinecone and OpenAI.

---

## Overview

### What's New After M1.1

**M1.1 (Dense-Only Search):**
- Single embedding model (OpenAI text-embedding-3-small)
- Pure semantic similarity
- 40-60ms query latency
- Misses exact keyword matches

**M1.2 (Hybrid Search — This Module):**
- ✅ **Dense + Sparse vectors**: Semantic understanding + keyword precision
- ✅ **Alpha parameter tuning**: Query-specific blending (0.2-0.8)
- ✅ **Namespaces**: Multi-tenant isolation within single index
- ✅ **BM25 sparse embeddings**: Exact term matching (TF-IDF based)
- ✅ **GPT-4 reranking**: Post-retrieval quality boost
- ✅ **Production failure handling**: 5 common scenarios with defensive code
- ✅ **Decision framework**: When to use (and avoid) hybrid search

**Key Benefits:**
- 20-40% better recall for mixed-intent queries
- Catches exact matches (product codes, IDs, legal terms)
- Single index for multiple tenants/users
- Real-world production patterns

**Trade-offs:**
- +30-80ms latency vs dense-only
- 4-8 hours alpha tuning per use case
- BM25 refitting required on corpus updates (5-15 min per 10K docs)

---

## Quickstart

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   PINECONE_API_KEY=pc-...
#   PINECONE_REGION=us-east-1
```

### 2. Run the Notebook

```bash
jupyter notebook M1_2_Pinecone_Advanced_Indexing.ipynb
```

The notebook demonstrates:
1. Pinecone data model (index/namespace/vector structure)
2. Dense vs sparse embedding comparison
3. Building a hybrid index with dotproduct metric
4. Alpha parameter tuning (0.2, 0.5, 0.8)
5. Multi-tenant namespaces
6. 5 common production failures + fixes
7. Decision card & cost analysis

### 3. Run Sample Queries (Python)

```python
from m1_2_pinecone_advanced_indexing import (
    build_index, upsert_hybrid_vectors, hybrid_query, smart_alpha_selector
)

# Build index
index = build_index()

# Upsert documents with hybrid vectors
docs = ["Machine learning models require tuning...", "..."]
upsert_hybrid_vectors(docs, namespace="demo")

# Query with auto-selected alpha
query = "explain hyperparameter optimization"
alpha = smart_alpha_selector(query)  # Returns 0.7 (semantic-heavy)
results = hybrid_query(query, alpha=alpha, top_k=5, namespace="demo")

for res in results:
    print(f"[{res['score']:.4f}] {res['text'][:80]}...")
```

### 4. Run Tests

```bash
python tests_hybrid.py
```

---

## Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
│              "Find SEC 10-K filing 2024"                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │  Smart Alpha Selector   │
        │  (Heuristic Analysis)   │
        │  → α = 0.3 (keyword)    │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼─────┐            ┌─────▼────┐
   │  Dense   │            │  Sparse  │
   │ OpenAI   │            │   BM25   │
   │ 1536-dim │            │ TF-IDF   │
   └────┬─────┘            └─────┬────┘
        │                        │
        │ * alpha (0.3)          │ * (1 - alpha) (0.7)
        │                        │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Pinecone Hybrid       │
        │   Query (dotproduct)    │
        │   Namespace: "user-123" │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Top-K Results (10)    │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   GPT-4 Reranker        │
        │   (Optional, top 3)     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Final Results         │
        │   Sorted by Relevance   │
        └─────────────────────────┘
```

**Key Components:**
- **Smart Alpha Selector**: Heuristic rules for query-specific blending
- **Dense Embeddings**: OpenAI text-embedding-3-small (1536-dim)
- **Sparse Embeddings**: BM25Encoder from pinecone-text
- **Hybrid Query**: Weighted combination using dotproduct metric
- **Namespaces**: Tenant isolation (e.g., "user-123")
- **GPT-4 Reranking**: Optional post-processing for top results

---

## Five Common Failures (With Fixes)

All failure modes are handled in `m1_2_pinecone_advanced_indexing.py` with defensive code.

### 1. BM25 Not Fitted ❌

**File:** `m1_2_pinecone_advanced_indexing.py:87` (`embed_sparse_bm25`)

**Symptom:**
```python
ValueError: BM25 encoder must be fitted before encoding queries
```

**Cause:** Called `embed_sparse_bm25(query=...)` before fitting on corpus.

**Fix:**
```python
# ✅ Correct order
embed_sparse_bm25(texts=corpus)  # Fit first
sparse_vec = embed_sparse_bm25(query="test")  # Then encode

# ❌ Wrong order
sparse_vec = embed_sparse_bm25(query="test")  # Error!
```

**Detection:** `check_bm25_fitted()` returns `True` when ready.

---

### 2. Metric Mismatch ❌

**File:** `m1_2_pinecone_advanced_indexing.py:36` (`build_index`)

**Symptom:** Alpha blending produces incorrect scores, results don't match expectations.

**Cause:** Index created with `cosine` or `euclidean` instead of `dotproduct`.

**Why It Matters:** Hybrid search scales vectors by alpha weights:
```python
combined_score = dotproduct(dense * alpha) + dotproduct(sparse * (1 - alpha))
```
Cosine normalization breaks this linear combination.

**Fix:**
```python
# ✅ Correct
build_index(metric="dotproduct")

# ❌ Wrong
build_index(metric="cosine")  # Raises ValueError
```

**Detection:** `check_index_metric()` validates at startup.

---

### 3. Missing Namespace ❌

**File:** `m1_2_pinecone_advanced_indexing.py:179` (`safe_namespace_query`)

**Symptom:** Query returns 0 results despite valid data.

**Cause:** Typo in namespace name or querying before upsert completes.

**Fix:**
```python
# ✅ Use safe_namespace_query
from m1_2_pinecone_advanced_indexing import safe_namespace_query

results = safe_namespace_query(index, "user-123", vector, top_k=5)
# Returns [] with error log if namespace doesn't exist

# ❌ Direct query (no validation)
results = index.query(namespace="user-123", ...)  # Silent failure
```

**Detection:** Checks `namespace in index.describe_index_stats()["namespaces"]` before querying.

---

### 4. Metadata Size Exceeds 40KB ❌

**File:** `m1_2_pinecone_advanced_indexing.py:106` (`validate_metadata_size`)

**Symptom:**
```
PineconeException: Metadata size exceeds 40KB limit
```

**Cause:** Long text fields or too many metadata keys.

**Fix:**
```python
# ✅ Truncate long fields
metadata = {
    "text": doc[:500],  # Limit to 500 chars
    "source": "example_data",
    "id": "123"
}
validate_metadata_size(metadata)  # Passes

# ❌ Full document in metadata
metadata = {"text": doc}  # 100KB → Error
```

**Best Practice:** Store full text externally (S3, database), keep only preview in metadata.

---

### 5. Partial Batch Failures ❌

**File:** `m1_2_pinecone_advanced_indexing.py:123` (`upsert_hybrid_vectors`)

**Symptom:** Some vectors fail silently during batch upsert.

**Cause:** Network errors, rate limits, dimension mismatches.

**Fix:**
```python
result = upsert_hybrid_vectors(docs, namespace="demo")

print(result)
# {
#   "success": 18,
#   "failed": 2,
#   "failed_ids": ["doc_5", "doc_12"],
#   "namespace": "demo"
# }

# ✅ Retry failed IDs
if result["failed_ids"]:
    retry_docs = [docs[int(id.split("_")[1])] for id in result["failed_ids"]]
    upsert_hybrid_vectors(retry_docs, namespace="demo")
```

**Detection:** Function returns `failed_ids` list for retry logic.

---

## Decision Card Summary

### Benefits ✅

- **20-40% Better Recall**: vs dense-only for mixed queries
- **Exact Match Coverage**: Product codes, IDs, legal terms
- **Multi-Tenancy**: Namespaces for user isolation
- **Query Flexibility**: Alpha tuning per query type
- **Production Ready**: Built-in failure handling

### Limitations ⚠️

- **+30-80ms Latency**: vs 40-60ms dense-only
- **Alpha Tuning Cost**: 4-8 hours per use case
- **BM25 Refitting**: 5-15 min per 10K docs on updates
- **Metric Lock-In**: Must use dotproduct (no cosine)
- **Shared Vocabulary**: BM25 corpus across namespaces

### Use Hybrid When ✅

- Mixed-intent queries (semantic + keywords)
- Need exact match + semantic similarity
- Willing to invest tuning time
- Acceptable latency overhead
- Multi-tenant requirements

### Avoid Hybrid When ❌

- 70%+ purely keyword queries → Use Elasticsearch
- 70%+ purely semantic queries → Use dense-only (M1.1)
- Sub-50ms latency required
- Hourly corpus updates
- No alpha tuning bandwidth

---

## Cost Table (Monthly Estimates)

| Scale | Queries/Day | Vectors | Embedding | Pinecone | **Total** |
|-------|-------------|---------|-----------|----------|-----------|
| Dev/Test | 100 | 1,000 | $5 | $15 | **$20** |
| Small | 1,000 | 10,000 | $25 | $75 | **$100** |
| Medium | 10,000 | 100,000 | $250 | $450 | **$700** |
| Large | 100,000 | 1,000,000 | $2,500 | $3,200 | **$5,700** |

**Assumptions:**
- OpenAI text-embedding-3-small: $0.020/1M tokens
- Pinecone serverless (us-east-1): ~$0.40/1M vectors/month + query costs
- 2 queries per upsert (read + write operations)

**Cost Drivers:**
1. **Embedding API**: Scales with text length × query volume
2. **Pinecone Storage**: Fixed per vector count
3. **Pinecone Queries**: Variable by serverless vs p1 pods

---

## Troubleshooting

### Issue: Empty Query Results

**Check:**
1. Namespace exists: `index.describe_index_stats()["namespaces"]`
2. BM25 fitted: `check_bm25_fitted()` returns `True`
3. Index metric: `index.metric == "dotproduct"`

**Fix:** Use `safe_namespace_query()` for automatic validation.

---

### Issue: Incorrect Alpha Behavior

**Symptoms:**
- Alpha=0.2 returns same results as alpha=0.8
- Keyword queries not favoring exact matches

**Check:**
1. Index metric is dotproduct (not cosine)
2. BM25 corpus matches your query domain
3. Sparse vectors have non-zero values: `sparse_vec["values"]`

**Fix:**
```python
# Refit BM25 on your actual corpus
embed_sparse_bm25(texts=your_corpus)

# Validate sparse encoding
sparse_vec = embed_sparse_bm25(query="test")
assert len(sparse_vec["values"]) > 0  # Should have non-zero terms
```

---

### Issue: High Latency (>200ms)

**Typical Breakdown:**
- Dense embedding: 40-60ms
- Sparse encoding: 5-10ms
- Pinecone query: 20-40ms
- Network overhead: 10-20ms
- **Total**: 75-130ms (target)

**If > 200ms, check:**
1. OpenAI API region latency
2. Pinecone index region (use us-east-1 for lowest latency)
3. BM25 encoding optimization (cache fitted encoder)
4. Batch queries when possible

**Optimization:**
```python
# Cache BM25 encoder (already done in module)
# Use connection pooling for OpenAI client
# Enable Pinecone query caching for repeated queries
```

---

### Issue: Metadata Size Errors

**Error:**
```
PineconeException: Metadata exceeds 40KB
```

**Check Metadata Size:**
```python
from m1_2_pinecone_advanced_indexing import validate_metadata_size

metadata = {"text": doc, "source": "...", ...}
try:
    validate_metadata_size(metadata)
except ValueError as e:
    print(e)  # "Metadata too large: 65536 bytes (limit: 40960)"
```

**Fix:**
- Truncate text fields: `doc[:500]`
- Remove unnecessary keys
- Store full text externally (S3, database)
- Reference via `doc_id` in metadata

---

## Link to Next Module

**Current Module: M1.2 ✅**
- Dense + sparse hybrid search
- Alpha parameter tuning
- Namespaces & multi-tenancy
- Production failure handling

**Next Module: M1.3 — Document Pipeline & Chunking**
- Document loaders (PDF, DOCX, HTML, Markdown)
- Chunking strategies (fixed-size, semantic, recursive)
- Metadata extraction & enrichment
- End-to-end ingestion pipeline
- Chunk overlap & boundary handling
- Quality validation & monitoring

**Coming in M1.4:**
- Advanced RAG patterns (HyDE, RAG-Fusion)
- Context compression
- Multi-hop reasoning
- Evaluation metrics (MRR, NDCG)

---

## File Reference

```
rag21d_learners/
├── M1_2_Pinecone_Advanced_Indexing.ipynb  # Main notebook (7 sections)
├── m1_2_pinecone_advanced_indexing.py     # Core module (all functions)
├── config.py                              # API keys & constants
├── example_data.txt                       # Sample corpus (20 lines)
├── requirements.txt                       # Dependencies
├── .env.example                           # Environment template
├── tests_hybrid.py                        # Smoke tests
└── README.md                              # This file
```

---

## Questions?

Open an issue in the repository or review:
- Notebook sections 1-7 for detailed explanations
- `m1_2_pinecone_advanced_indexing.py` for implementation details
- Five failure scenarios in section 6 of the notebook

**Key Takeaway:** Hybrid search adds 20-40% recall improvement at the cost of 30-80ms latency and 4-8 hours tuning time. Use when mixed-intent queries justify the complexity.
