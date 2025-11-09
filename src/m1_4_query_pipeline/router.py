"""
FastAPI router for M1.4 Query Pipeline endpoints.
Provides REST API interface for the query pipeline.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from src.m1_4_query_pipeline.module import ProductionRAG, QueryType
from src.m1_4_query_pipeline.config import get_clients, has_api_keys

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/m1_4_query_pipeline", tags=["M1.4 Query Pipeline"])

# Request/Response models
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., description="User query to process")
    top_k: Optional[int] = Field(5, description="Number of initial retrieval results", ge=1, le=20)
    rerank_top_k: Optional[int] = Field(3, description="Number of results after reranking", ge=1, le=10)
    namespace: Optional[str] = Field("demo", description="Pinecone namespace")
    temperature: Optional[float] = Field(0.1, description="LLM temperature", ge=0.0, le=2.0)
    use_reranking: Optional[bool] = Field(True, description="Enable cross-encoder reranking")
    use_expansion: Optional[bool] = Field(False, description="Enable query expansion")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str
    query_type: str
    keywords: List[str]
    chunks_retrieved: int
    sources: List[str]
    avg_score: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    total_time: float


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    module: str
    api_keys_configured: bool


class MetricsResponse(BaseModel):
    """Response model for /metrics endpoint."""
    total_queries: int
    avg_latency_ms: float
    success_rate: float
    note: str


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint."""
    status: str
    message: str
    note: str


# Global state for basic metrics (production should use Redis/DB)
_metrics = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "successes": 0
}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and configuration info
    """
    return HealthResponse(
        status="ok",
        module="m1_4_query_pipeline",
        api_keys_configured=has_api_keys()
    )


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Execute complete RAG query pipeline.

    Args:
        request: Query request with parameters

    Returns:
        Query response with answer, sources, and metrics

    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Get clients
        openai_client, pinecone_client = get_clients()

        # Initialize RAG pipeline
        rag = ProductionRAG(
            openai_client=openai_client,
            pinecone_client=pinecone_client,
            use_expansion=request.use_expansion,
            use_reranking=request.use_reranking
        )

        # Execute pipeline
        result = rag.query(
            query=request.query,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            namespace=request.namespace,
            temperature=request.temperature
        )

        # Update metrics
        _metrics["total_queries"] += 1
        _metrics["total_latency_ms"] += result["total_time"] * 1000
        _metrics["successes"] += 1

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        _metrics["total_queries"] += 1
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get basic pipeline metrics.

    Returns:
        Aggregated metrics (safe even without API keys)
    """
    total = _metrics["total_queries"]
    avg_latency = _metrics["total_latency_ms"] / total if total > 0 else 0.0
    success_rate = _metrics["successes"] / total if total > 0 else 1.0

    return MetricsResponse(
        total_queries=total,
        avg_latency_ms=round(avg_latency, 2),
        success_rate=round(success_rate, 3),
        note="In-memory metrics; use external analytics in production"
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(documents: List[str] = None):
    """
    Stub endpoint for document ingestion (handled in M1.3).

    Args:
        documents: Optional list of documents

    Returns:
        Status message indicating ingestion is handled by M1.3
    """
    return IngestResponse(
        status="ok",
        message="Ingestion stub endpoint",
        note="Document ingestion is handled by M1.3 Indexing module. Use /m1_3_indexing/ingest for actual document processing."
    )
