"""
M1.2 — Pinecone Data Model & Advanced Indexing (Hybrid Search)

## Purpose
This module implements production-ready hybrid search that combines dense (semantic)
and sparse (keyword) vector retrieval. It demonstrates how to blend OpenAI embeddings
with BM25 sparse vectors using alpha-weighted queries, enabling both semantic understanding
and exact keyword matching in a single search operation.

## Concepts Covered
1. **Hybrid Vector Search**
   - Dense embeddings (OpenAI text-embedding-3-small, 1536-dim)
   - Sparse embeddings (BM25/TF-IDF for keyword matching)
   - Alpha parameter tuning (0.0-1.0) for query-specific blending

2. **Pinecone Data Model**
   - Index → Namespace → Vector hierarchy
   - Hybrid vectors (dense + sparse + metadata)
   - Namespace-based multi-tenant isolation

3. **Production Error Handling**
   - BM25 not fitted detection
   - Metric mismatch validation (dotproduct required)
   - Missing namespace checks
   - Metadata size limits (<40KB)
   - Partial batch failure tracking

4. **Query Optimization**
   - Smart alpha selector (heuristic-based)
   - Safe namespace queries with validation
   - GPT-4 reranking (optional post-processing)
   - Exponential backoff for rate limits

5. **Real-World Trade-offs**
   - When to use hybrid vs dense-only
   - Latency costs (+30-80ms vs dense-only)
   - BM25 refitting overhead (5-15 min per 10K docs)
   - Cost analysis ($20-$5,700/month at scale)

## After Completing This Module
You will be able to:
- Implement production hybrid search with defensive error handling
- Choose appropriate alpha values based on query characteristics
- Design multi-tenant search systems using namespaces
- Diagnose and fix 5 common hybrid search failures
- Make informed decisions about when hybrid search adds value

## Context in Track
**Prerequisites:** M1.1 (Understanding Vector Databases, Dense-Only Search)

**This Module (M1.2):** Advanced indexing with hybrid search, namespaces,
production patterns, and decision frameworks

**Next:** M1.3 (Document Pipeline & Chunking) — Document loaders, chunking
strategies, metadata extraction, and end-to-end ingestion

**Track Overview:**
- M1.1: Vector databases fundamentals (dense search)
- M1.2: Advanced indexing (hybrid search, namespaces) ← You are here
- M1.3: Document processing pipeline
- M1.4: Query pipeline & response generation
"""

from src.m1_2_pinecone_hybrid.config import (
    get_clients,
    OPENAI_MODEL,
    PINECONE_INDEX,
    REGION,
    DEFAULT_NAMESPACE,
    SCORE_THRESHOLD
)

from src.m1_2_pinecone_hybrid.module import (
    build_index,
    embed_dense_openai,
    embed_sparse_bm25,
    validate_metadata_size,
    upsert_hybrid_vectors,
    safe_namespace_query,
    smart_alpha_selector,
    hybrid_query,
    rerank_results,
    check_bm25_fitted,
    check_index_metric
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "get_clients",
    "OPENAI_MODEL",
    "PINECONE_INDEX",
    "REGION",
    "DEFAULT_NAMESPACE",
    "SCORE_THRESHOLD",
    # Core functions
    "build_index",
    "embed_dense_openai",
    "embed_sparse_bm25",
    "validate_metadata_size",
    "upsert_hybrid_vectors",
    "safe_namespace_query",
    "smart_alpha_selector",
    "hybrid_query",
    "rerank_results",
    # Health checks
    "check_bm25_fitted",
    "check_index_metric",
]
