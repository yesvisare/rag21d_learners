"""
M2.1 — Caching Strategies for Cost Reduction

**Purpose**
-----------
Learn to deploy multi-layer Redis caching to reduce RAG system costs by 30-70%
through strategic query, embedding, and context caching. Master cache invalidation,
stampede protection, and recognize when caching becomes counterproductive.

**Concepts Covered**
--------------------
- Multi-layer cache architecture (exact, semantic, embedding, context)
- Hash-based exact matching with SHA-256 for collision-free keys
- Semantic similarity caching using fuzzy string matching (BM25/MinHash)
- Cache stampede prevention with per-key distributed locks
- TTL (Time-To-Live) strategies for different cache layers
- Invalidation patterns: manual, prefix-based, stale detection
- Metrics tracking and ROI (Return on Investment) analysis
- Trade-offs: query diversity vs. hit rates, freshness vs. staleness
- Production failure modes: stampedes, memory exhaustion, hash collisions

**After Completing This Module**
---------------------------------
You will be able to:
- Implement a production-ready multi-layer cache for RAG systems
- Configure appropriate TTLs based on content freshness requirements
- Diagnose and resolve common caching failures (stampedes, stale data, low ROI)
- Calculate cost savings and project ROI for caching infrastructure
- Recognize scenarios where caching creates more problems than it solves
  (>90% query diversity, <5min freshness needs, <500 daily queries)
- Design invalidation strategies that balance freshness and performance
- Integrate caching into existing RAG pipelines without code changes

**Context in Track**
--------------------
This is Module 2.1 in the RAG optimization track. Prerequisites include basic
RAG system understanding (M1.x modules). This module feeds into:
- M2.2: Query Optimization & Prompt Engineering
- M2.3: Model Selection & Cost-Performance Trade-offs
- M3.1: Chunking Strategies & Vector Databases

The caching strategies learned here apply across all RAG architectures and
complement other cost-optimization techniques (smaller models, batch processing,
prompt compression).

**Usage**
---------
Library usage:
    from src.m2_1_caching.module import MultiLayerCache
    from src.m2_1_caching.config import get_redis, get_openai

    cache = MultiLayerCache(get_redis(), get_openai())

    # Check cache
    result = cache.get_exact("How do I reset my password?")
    if not result:
        result = {"answer": "Visit settings..."}
        cache.set_exact(query, result)

API usage:
    # Start server
    uvicorn app:app --reload

    # Check metrics
    curl http://localhost:8000/metrics

    # Invalidate cache
    curl -X POST http://localhost:8000/invalidate -d '{"prefix":"semantic:"}'

CLI usage:
    python -m src.m2_1_caching.module

Notebook:
    jupyter notebook notebooks/M2_1_Caching_Strategies.ipynb
"""

from .module import (
    MultiLayerCache,
    CacheMetrics,
    CacheKeyGenerator,
    StampedeLock
)

from .config import (
    get_redis,
    get_openai,
    has_services,
    PREFIX_EXACT,
    PREFIX_SEMANTIC,
    PREFIX_EMBEDDING,
    PREFIX_CONTEXT
)

__version__ = "1.0.0"
__all__ = [
    "MultiLayerCache",
    "CacheMetrics",
    "CacheKeyGenerator",
    "StampedeLock",
    "get_redis",
    "get_openai",
    "has_services",
    "PREFIX_EXACT",
    "PREFIX_SEMANTIC",
    "PREFIX_EMBEDDING",
    "PREFIX_CONTEXT",
]
