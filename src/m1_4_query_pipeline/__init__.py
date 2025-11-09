"""
M1.4 — Query Pipeline & Response Generation

## Purpose
Transform user queries into grounded, cited answers through a production-ready
7-stage RAG pipeline. This module integrates query understanding, hybrid retrieval,
cross-encoder reranking, context preparation, and LLM generation with comprehensive
metrics tracking.

## Concepts Covered
- **Query Classification**: 6 types (factual, how-to, comparison, definition, troubleshooting, opinion)
- **Query Expansion**: LLM-based alternative phrasings for improved recall
- **Hybrid Retrieval**: Dense (semantic) + sparse (BM25) search with auto-tuned alpha
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 for 10-20% better relevance
- **Context Building**: Deduplication, source attribution, length guards
- **Prompt Engineering**: Query-type specific templates for optimal responses
- **Response Generation**: Streaming and non-streaming modes
- **Metrics & Attribution**: Timings, scores, source tracking

## After Completing This Module
You will be able to:
- Build end-to-end RAG query pipelines with 7 sequential stages
- Implement hybrid search combining semantic and keyword retrieval
- Apply query-type specific optimizations (alpha tuning 0.3-0.8)
- Use cross-encoder reranking to improve result quality
- Handle graceful fallbacks for missing API keys and service failures
- Track comprehensive metrics (retrieval_time, generation_time, avg_score)
- Provide source attribution for compliance and trust
- Make informed trade-offs (accuracy vs latency, coverage vs precision)

## Context in RAG21D Track
This module builds on M1.3 (Indexing & Retrieval Strategies) which established
vector DB setup and hybrid search foundations. M1.4 completes the query-time
pipeline by adding:
- Intelligent query understanding and expansion
- Cross-encoder reranking for quality gains
- Production-grade context preparation and prompting
- Comprehensive metrics and error handling

**Prerequisites**: M1.3 (vector indexing, Pinecone setup, embeddings)
**Enables**: Production RAG deployments with 60-80% hallucination reduction
"""

from src.m1_4_query_pipeline.module import (
    QueryType,
    RetrievalResult,
    QueryProcessor,
    SmartRetriever,
    Reranker,
    ContextBuilder,
    PromptBuilder,
    ResponseGenerator,
    ProductionRAG
)

from src.m1_4_query_pipeline.config import (
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    PINECONE_INDEX,
    DEFAULT_NAMESPACE,
    SCORE_THRESHOLD,
    get_clients,
    has_api_keys
)

__version__ = "1.0.0"
__all__ = [
    "QueryType",
    "RetrievalResult",
    "QueryProcessor",
    "SmartRetriever",
    "Reranker",
    "ContextBuilder",
    "PromptBuilder",
    "ResponseGenerator",
    "ProductionRAG",
    "OPENAI_MODEL",
    "EMBEDDING_MODEL",
    "PINECONE_INDEX",
    "DEFAULT_NAMESPACE",
    "SCORE_THRESHOLD",
    "get_clients",
    "has_api_keys"
]
