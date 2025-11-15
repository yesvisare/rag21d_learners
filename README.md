# M1.2 — Pinecone Data Model & Advanced Indexing

**Hybrid Search, Namespaces, Failures, Decision Framework**

---

## Purpose

This module implements production-ready **hybrid search** that combines dense (semantic) and sparse (keyword) vector retrieval. It demonstrates how to blend OpenAI embeddings with BM25 sparse vectors using alpha-weighted queries, enabling both semantic understanding and exact keyword matching in a single search operation.

## Concepts Covered

1. **Hybrid Vector Search** — Dense + sparse embeddings with alpha blending (0.0-1.0)
2. **Pinecone Data Model** — Index → Namespace → Vector hierarchy
3. **Production Error Handling** — BM25 not fitted, metric mismatch, missing namespaces, metadata limits, batch failures
4. **Query Optimization** — Smart alpha selection, namespace validation, GPT-4 reranking
5. **Trade-off Analysis** — When to use hybrid vs dense-only, latency costs, BM25 refitting overhead

## After Completing This Module

You will be able to:
- Implement production hybrid search with defensive error handling
- Choose appropriate alpha values based on query characteristics
- Design multi-tenant search systems using namespaces
- Diagnose and fix 5 common hybrid search failures
- Make informed decisions about when hybrid search adds value

## Context in Track

**Prerequisites:** M1.1 (Understanding Vector Databases, Dense-Only Search)

**This Module (M1.2):** Advanced indexing with hybrid search, namespaces, production patterns

**Next:** M1.3 (Document Pipeline & Chunking)

---

## Quickstart

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yesvisare/rag21d_learners.git
cd rag21d_learners

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   PINECONE_API_KEY=pc-...
#   PINECONE_REGION=us-east-1
```

### 2. Run the API

**Linux/Mac:**
```bash
export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Windows PowerShell:**
```powershell
# Method 1: Use the provided script
.\scripts\run_local.ps1

# Method 2: Manual command
$env:PYTHONPATH="$PWD"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Windows CMD:**
```cmd
set PYTHONPATH=%CD%
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/m1_2/health
```

**Ingest Documents:**
```bash
curl -X POST http://localhost:8000/m1_2/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      "Machine learning models require careful hyperparameter tuning",
      "Vector databases enable semantic search capabilities"
    ],
    "namespace": "demo"
  }'
```

**Query with Hybrid Search:**
```bash
curl -X POST http://localhost:8000/m1_2/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain machine learning optimization",
    "alpha": 0.5,
    "top_k": 5,
    "namespace": "demo"
  }'
```

### 4. Run the Notebook

```bash
jupyter notebook notebooks/M1_2_Pinecone_Advanced_Indexing.ipynb
```

### 5. Run Tests

```bash
# Run API tests
pytest tests/test_smoke.py -v

# Run module tests
python tests/test_hybrid.py

# Run all tests
pytest tests/ -v
```

---

## Project Structure

```
rag21d_learners/
├── app.py                              # FastAPI application entry point
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore patterns
├── LICENSE                             # MIT License
├── README.md                           # This file
│
├── src/                                # Source code package
│   └── m1_2_pinecone_hybrid/
│       ├── __init__.py                 # Package initialization with documentation
│       ├── config.py                   # Configuration and client initialization
│       ├── module.py                   # Core hybrid search implementation
│       └── router.py                   # FastAPI routes and endpoints
│
├── notebooks/                          # Jupyter notebooks
│   └── M1_2_Pinecone_Advanced_Indexing.ipynb
│
├── tests/                              # Test suite
│   ├── test_smoke.py                   # API endpoint tests
│   └── test_hybrid.py                  # Module functionality tests
│
├── data/                               # Data files
│   └── example/
│       └── example_data.txt            # Sample documents (20 lines)
│
└── scripts/                            # Utility scripts
    └── run_local.ps1                   # PowerShell script to run API locally
```

---

## API Endpoints

### Health & Metrics

#### `GET /health`
Global health check
```json
{
  "status": "ok",
  "service": "M1.2 Pinecone Hybrid Search API",
  "version": "1.0.0"
}
```

#### `GET /m1_2/health`
Module-specific health check with readiness indicators
```json
{
  "status": "ok",
  "module": "m1_2_pinecone_hybrid",
  "bm25_fitted": true,
  "clients_available": true
}
```

#### `GET /m1_2/metrics`
Metrics stub (extendable)
```json
{
  "status": "ok",
  "message": "Metrics endpoint (stub)",
  "bm25_fitted": true
}
```

### Document Ingestion

#### `POST /m1_2/ingest`
Ingest documents with hybrid vectors (dense + sparse)

**Request:**
```json
{
  "docs": ["Document 1 text", "Document 2 text"],
  "namespace": "demo"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "success": 2,
  "failed": 0,
  "failed_ids": [],
  "namespace": "demo",
  "message": "Ingested 2 documents successfully"
}
```

**Response (No Keys):**
```json
{
  "status": "skipped",
  "namespace": "demo",
  "message": "⚠️ Skipped ingestion (no API keys). Would have ingested 2 documents."
}
```

### Search Queries

#### `POST /m1_2/query`
Execute hybrid search with alpha-weighted blending

**Request:**
```json
{
  "query": "explain machine learning concepts",
  "alpha": 0.7,
  "top_k": 5,
  "namespace": "demo"
}
```

**Parameters:**
- `query` (required): Search query text
- `alpha` (optional): Blending weight 0.0-1.0 (auto-selected if omitted)
  - 0.0 = pure sparse (keyword only)
  - 0.5 = balanced
  - 1.0 = pure dense (semantic only)
- `top_k` (optional, default=5): Number of results (1-100)
- `namespace` (optional, default="demo"): Target namespace

**Response:**
```json
{
  "status": "success",
  "query": "explain machine learning concepts",
  "alpha": 0.7,
  "namespace": "demo",
  "results": [
    {
      "id": "doc_0",
      "score": 0.8234,
      "text": "Machine learning models require...",
      "metadata": {"text": "...", "source": "example_data"}
    }
  ],
  "count": 1
}
```

---

## Module Usage (Python)

```python
from src.m1_2_pinecone_hybrid import (
    build_index,
    upsert_hybrid_vectors,
    hybrid_query,
    smart_alpha_selector
)

# 1. Build or connect to index
index = build_index()

# 2. Ingest documents
docs = [
    "Machine learning models require tuning",
    "Vector databases enable semantic search"
]
result = upsert_hybrid_vectors(docs, namespace="demo")
print(f"Ingested {result['success']} documents")

# 3. Query with auto-selected alpha
query = "explain hyperparameter optimization"
alpha = smart_alpha_selector(query)  # Returns 0.7 (semantic-heavy)
results = hybrid_query(query, alpha=alpha, top_k=5, namespace="demo")

# 4. Process results
for res in results:
    print(f"[{res['score']:.4f}] {res['text'][:80]}...")
```

### CLI Usage

```bash
# Run module directly
python -m src.m1_2_pinecone_hybrid.module

# Output:
# M1.2 Pinecone Advanced Indexing Module
# Import this module in notebooks or scripts
# Key functions:
#   - build_index()
#   - embed_dense_openai(text)
#   - embed_sparse_bm25(texts|query)
#   - upsert_hybrid_vectors(docs, namespace)
#   - hybrid_query(query, alpha, top_k, namespace)
#   - smart_alpha_selector(query)
#   - rerank_results(query, results)
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

All failure modes are handled in `src/m1_2_pinecone_hybrid/module.py` with defensive code.

### 1. BM25 Not Fitted ❌

**File:** `src/m1_2_pinecone_hybrid/module.py:113`

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

**File:** `src/m1_2_pinecone_hybrid/module.py:28`

**Symptom:** Alpha blending produces incorrect scores, results don't match expectations.

**Cause:** Index created with `cosine` or `euclidean` instead of `dotproduct`.

**Why It Matters:** Hybrid search scales vectors by alpha weights — cosine normalization breaks this linear combination.

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

**File:** `src/m1_2_pinecone_hybrid/module.py:271`

**Symptom:** Query returns 0 results despite valid data.

**Cause:** Typo in namespace name or querying before upsert completes.

**Fix:**
```python
# ✅ Use safe_namespace_query
from src.m1_2_pinecone_hybrid import safe_namespace_query

results = safe_namespace_query(index, "user-123", vector, top_k=5)
# Returns [] with error log if namespace doesn't exist
```

**Detection:** Checks `namespace in index.describe_index_stats()["namespaces"]` before querying.

---

### 4. Metadata Size Exceeds 40KB ❌

**File:** `src/m1_2_pinecone_hybrid/module.py:155`

**Symptom:**
```
PineconeException: Metadata size exceeds 40KB limit
```

**Cause:** Long text fields or too many metadata keys.

**Fix:**
```python
# ✅ Truncate long fields (line 225 in module.py)
metadata = {
    "text": doc[:500],  # Limit to 500 chars
    "source": "example_data"
}
validate_metadata_size(metadata)  # Passes
```

**Best Practice:** Store full text externally (S3, database), keep only preview in metadata.

---

### 5. Partial Batch Failures ❌

**File:** `src/m1_2_pinecone_hybrid/module.py:177`

**Symptom:** Some vectors fail silently during batch upsert.

**Cause:** Network errors, rate limits, dimension mismatches.

**Fix:**
```python
result = upsert_hybrid_vectors(docs, namespace="demo")

# Check for failures (line 258-267)
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
3. Sparse vectors have non-zero values

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

---

## Questions?

Open an issue in the repository or review:
- Notebook at `notebooks/M1_2_Pinecone_Advanced_Indexing.ipynb`
- Source code at `src/m1_2_pinecone_hybrid/module.py`
- Five failure scenarios above with file references

**Key Takeaway:** Hybrid search adds 20-40% recall improvement at the cost of 30-80ms latency and 4-8 hours tuning time. Use when mixed-intent queries justify the complexity.
