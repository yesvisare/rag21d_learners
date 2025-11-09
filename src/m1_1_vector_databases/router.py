"""
FastAPI router for M1.1 Vector Databases module.

Provides REST API endpoints for vector database operations:
- Health check
- Data ingestion (init flow)
- Semantic search queries
- Metrics retrieval
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.m1_1_vector_databases import config
from src.m1_1_vector_databases.module import (
    load_example_texts,
    embed_texts_openai,
    create_index_and_wait_pinecone,
    upsert_vectors,
    query_pinecone,
)

logger = logging.getLogger(__name__)

# Create APIRouter with prefix
router = APIRouter(
    prefix="/m1_1",
    tags=["M1.1 Vector Databases"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for semantic search query."""
    query: str = Field(..., description="Natural language query text", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=100)
    threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)
    namespace: Optional[str] = Field("demo", description="Namespace to query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class QueryResponse(BaseModel):
    """Response model for semantic search query."""
    query: str
    results: List[Dict[str, Any]]
    count: int
    threshold: float
    timestamp: str

class IngestResponse(BaseModel):
    """Response model for data ingestion."""
    status: str
    message: str
    vectors_upserted: Optional[int] = None
    namespace: Optional[str] = None
    timestamp: str

class MetricsResponse(BaseModel):
    """Response model for metrics."""
    module: str
    index_name: str
    total_vectors: Optional[int] = None
    namespaces: Optional[Dict[str, Any]] = None
    has_api_keys: bool
    timestamp: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def has_required_keys() -> bool:
    """Check if required API keys are configured."""
    return bool(config.OPENAI_API_KEY and config.PINECONE_API_KEY)

def get_safe_clients():
    """Get clients if keys are available, otherwise return None."""
    if not has_required_keys():
        return None, None
    try:
        return config.get_clients()
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return None, None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status and module info (always succeeds)
    """
    return {
        "status": "ok",
        "module": "m1_1_vector_databases",
        "version": "1.0.0",
        "has_api_keys": has_required_keys(),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/ingest", response_model=IngestResponse)
async def ingest_data():
    """
    Run initialization flow: create index, load data, generate embeddings, upsert.

    If API keys are missing, returns 200 with skip message (safe for learning/testing).

    Returns:
        IngestResponse with status and statistics
    """
    timestamp = datetime.utcnow().isoformat()

    # Check for API keys
    if not has_required_keys():
        logger.warning("Ingestion skipped: API keys not configured")
        return IngestResponse(
            status="skipped",
            message="API keys not configured. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file.",
            timestamp=timestamp
        )

    try:
        # Initialize clients
        openai_client, pinecone_client = config.get_clients()

        # Load example data
        logger.info("Loading example texts...")
        texts = load_example_texts()

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = embed_texts_openai(texts, client=openai_client)

        # Create or connect to index
        logger.info(f"Creating/connecting to index: {config.INDEX_NAME}")
        index = create_index_and_wait_pinecone(
            pinecone_client,
            config.INDEX_NAME,
            config.EMBEDDING_DIM
        )

        # Prepare vectors with metadata
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append((
                f"doc_{i}",
                embedding,
                {
                    "text": text,
                    "source": "data/example/example_data.txt",
                    "chunk_id": i,
                    "timestamp": timestamp,
                    "length": len(text)
                }
            ))

        # Upsert vectors
        logger.info(f"Upserting {len(vectors)} vectors...")
        stats = upsert_vectors(index, vectors, namespace=config.DEFAULT_NAMESPACE)

        return IngestResponse(
            status="success",
            message=f"Successfully upserted {stats['upserted']} vectors to namespace '{stats['namespace']}'",
            vectors_upserted=stats['upserted'],
            namespace=stats['namespace'],
            timestamp=timestamp
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    """
    Perform semantic search query against vector database.

    Args:
        request: QueryRequest with query text and parameters

    Returns:
        QueryResponse with matching results

    Raises:
        HTTPException: If API keys missing or query fails
    """
    timestamp = datetime.utcnow().isoformat()

    # Check for API keys
    if not has_required_keys():
        raise HTTPException(
            status_code=503,
            detail="API keys not configured. Set OPENAI_API_KEY and PINECONE_API_KEY in .env file."
        )

    try:
        # Initialize clients
        openai_client, pinecone_client = config.get_clients()

        # Connect to index
        index = pinecone_client.Index(config.INDEX_NAME)

        # Execute query
        results = query_pinecone(
            index,
            request.query,
            client=openai_client,
            top_k=request.top_k or config.DEFAULT_TOP_K,
            namespace=request.namespace or config.DEFAULT_NAMESPACE,
            filters=request.filters,
            score_threshold=request.threshold or config.SCORE_THRESHOLD
        )

        return QueryResponse(
            query=request.query,
            results=results,
            count=len(results),
            threshold=request.threshold or config.SCORE_THRESHOLD,
            timestamp=timestamp
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        # Check for common errors
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail=f"Index '{config.INDEX_NAME}' not found. Run /ingest first to create index."
            )
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get metrics and statistics about the vector database.

    Returns stub data if API keys are missing (safe for testing).

    Returns:
        MetricsResponse with index statistics
    """
    timestamp = datetime.utcnow().isoformat()

    # Return stub if no keys
    if not has_required_keys():
        return MetricsResponse(
            module="m1_1_vector_databases",
            index_name=config.INDEX_NAME,
            total_vectors=None,
            namespaces=None,
            has_api_keys=False,
            timestamp=timestamp
        )

    try:
        # Initialize Pinecone client
        _, pinecone_client = config.get_clients()

        # Get index stats
        index = pinecone_client.Index(config.INDEX_NAME)
        stats = index.describe_index_stats()

        # Extract namespace info
        namespaces_info = {}
        if 'namespaces' in stats:
            for ns_name, ns_data in stats['namespaces'].items():
                namespaces_info[ns_name] = {
                    "vector_count": ns_data.get('vector_count', 0)
                }

        total = stats.get('total_vector_count', 0)

        return MetricsResponse(
            module="m1_1_vector_databases",
            index_name=config.INDEX_NAME,
            total_vectors=total,
            namespaces=namespaces_info,
            has_api_keys=True,
            timestamp=timestamp
        )

    except Exception as e:
        logger.warning(f"Metrics retrieval failed: {e}")
        # Return partial metrics on error
        return MetricsResponse(
            module="m1_1_vector_databases",
            index_name=config.INDEX_NAME,
            total_vectors=None,
            namespaces=None,
            has_api_keys=True,
            timestamp=timestamp
        )
