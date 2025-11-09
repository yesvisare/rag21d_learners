"""
FastAPI router for M1.2 Pinecone Hybrid Search endpoints.

Provides REST API access to hybrid search functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from src.m1_2_pinecone_hybrid.module import (
    upsert_hybrid_vectors,
    hybrid_query,
    smart_alpha_selector,
    check_bm25_fitted
)
from src.m1_2_pinecone_hybrid.config import get_clients, DEFAULT_NAMESPACE

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/m1_2",
    tags=["M1.2 Hybrid Search"]
)


# Pydantic models for request/response
class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    docs: List[str] = Field(..., description="List of document texts to ingest", min_items=1)
    namespace: Optional[str] = Field(DEFAULT_NAMESPACE, description="Target namespace for isolation")

    class Config:
        schema_extra = {
            "example": {
                "docs": ["Machine learning models require tuning", "Vector databases enable semantic search"],
                "namespace": "demo"
            }
        }


class QueryRequest(BaseModel):
    """Request model for hybrid search query."""
    query: str = Field(..., description="Search query text", min_length=1)
    alpha: Optional[float] = Field(None, description="Blending weight (0=sparse, 1=dense). Auto-selected if not provided", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=100)
    namespace: Optional[str] = Field(DEFAULT_NAMESPACE, description="Target namespace to query")

    class Config:
        schema_extra = {
            "example": {
                "query": "explain hyperparameter optimization",
                "alpha": 0.5,
                "top_k": 5,
                "namespace": "demo"
            }
        }


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    status: str
    success: int = 0
    failed: int = 0
    failed_ids: List[str] = []
    namespace: str
    message: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query."""
    status: str
    query: str
    alpha: float
    namespace: str
    results: List[Dict[str, Any]]
    count: int
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    module: str
    bm25_fitted: bool
    clients_available: bool


class MetricsResponse(BaseModel):
    """Response model for metrics (stub)."""
    status: str
    message: str
    bm25_fitted: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns module status and readiness indicators.
    """
    openai_client, pinecone_client = get_clients()
    clients_available = openai_client is not None and pinecone_client is not None

    return {
        "status": "ok",
        "module": "m1_2_pinecone_hybrid",
        "bm25_fitted": check_bm25_fitted(),
        "clients_available": clients_available
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents with hybrid vectors (dense + sparse).

    Creates embeddings and stores in specified namespace.
    Returns success/failure counts with graceful degradation if no API keys.
    """
    try:
        # Call upsert function
        result = upsert_hybrid_vectors(
            docs=request.docs,
            namespace=request.namespace
        )

        # Handle skip case (no keys)
        if "skipped" in result:
            return IngestResponse(
                status="skipped",
                namespace=request.namespace,
                message=f"⚠️ Skipped ingestion (no API keys). Would have ingested {result['skipped']} documents."
            )

        # Handle error case
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Success case
        return IngestResponse(
            status="success" if result["failed"] == 0 else "partial",
            success=result.get("success", 0),
            failed=result.get("failed", 0),
            failed_ids=result.get("failed_ids", []),
            namespace=result.get("namespace", request.namespace),
            message=f"Ingested {result.get('success', 0)} documents successfully"
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_hybrid(request: QueryRequest):
    """
    Execute hybrid search query with alpha-weighted blending.

    If alpha not provided, uses smart_alpha_selector for automatic tuning.
    Returns ranked results or empty list if no API keys (with skip message).
    """
    try:
        # Auto-select alpha if not provided
        alpha = request.alpha
        if alpha is None:
            alpha = smart_alpha_selector(request.query)
            logger.info(f"Auto-selected alpha: {alpha}")

        # Execute query
        results = hybrid_query(
            query=request.query,
            alpha=alpha,
            top_k=request.top_k,
            namespace=request.namespace
        )

        # Check if skipped (no keys)
        if not results:
            openai_client, pinecone_client = get_clients()
            if not openai_client or not pinecone_client:
                return QueryResponse(
                    status="skipped",
                    query=request.query,
                    alpha=alpha,
                    namespace=request.namespace,
                    results=[],
                    count=0,
                    message="⚠️ Skipped query (no API keys)"
                )

        return QueryResponse(
            status="success",
            query=request.query,
            alpha=alpha,
            namespace=request.namespace,
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get metrics stub (safe if no keys).

    Returns basic health indicators. Can be extended with query latency,
    success rates, namespace usage, etc.
    """
    return {
        "status": "ok",
        "message": "Metrics endpoint (stub). Extend with query latency, success rates, etc.",
        "bm25_fitted": check_bm25_fitted()
    }
