"""
FastAPI router for document processing pipeline.

Provides REST endpoints for document ingestion, querying, and health checks.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.m1_3_document_processing.config import get_clients, PINECONE_INDEX
from src.m1_3_document_processing.module import process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["document-processing"])

# Request/Response models
class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: Optional[str] = Field(None, description="Path to single document to process")
    dir_path: Optional[str] = Field(None, description="Path to directory of documents")
    index_name: Optional[str] = Field(PINECONE_INDEX, description="Pinecone index name")
    chunker: str = Field("semantic", description="Chunking strategy: fixed, semantic, or paragraph")
    chunk_size: int = Field(512, description="Target chunk size in characters")

    class Config:
        schema_extra = {
            "example": {
                "file_path": "data/example/example_data.txt",
                "index_name": "production-rag",
                "chunker": "semantic",
                "chunk_size": 512
            }
        }


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    message: str
    chunks_processed: int
    documents_processed: int
    skipped: bool = False
    skip_reason: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for document querying (stub for M1.4)."""
    query: str = Field(..., description="Query string")
    top_k: int = Field(5, description="Number of results to return")
    index_name: Optional[str] = Field(PINECONE_INDEX, description="Pinecone index name")


class QueryResponse(BaseModel):
    """Response model for document querying (stub for M1.4)."""
    status: str
    message: str
    results: list = []


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    status: str
    total_documents_processed: int
    total_chunks_generated: int
    api_keys_configured: Dict[str, bool]


# Global counters (in-memory, reset on restart)
_metrics = {
    "documents_processed": 0,
    "chunks_generated": 0
}


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status and module information.
    """
    return {
        "status": "ok",
        "module": "m1_3_document_processing",
        "version": "1.0.0"
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the processing pipeline.

    Processes documents through extraction → cleaning → chunking → embedding → storage.
    If API keys are not configured, skips embedding/storage and returns metadata only.

    Args:
        request: IngestRequest containing file/directory path and processing options

    Returns:
        IngestResponse with processing results

    Raises:
        HTTPException: If file not found or processing fails
    """
    try:
        # Get API clients (may be None if keys not configured)
        openai_client, pinecone_client = get_clients()

        # Check if we'll skip embedding/storage
        skip_embedding = openai_client is None or pinecone_client is None

        files_to_process = []

        # Collect files to process
        if request.file_path:
            file_path = Path(request.file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            files_to_process.append(file_path)

        elif request.dir_path:
            dir_path = Path(request.dir_path)
            if not dir_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Directory not found: {request.dir_path}")

            # Collect supported files
            files_to_process.extend(dir_path.glob('*.pdf'))
            files_to_process.extend(dir_path.glob('*.txt'))
            files_to_process.extend(dir_path.glob('*.md'))

            if not files_to_process:
                raise HTTPException(status_code=400, detail="No supported files found in directory")
        else:
            raise HTTPException(status_code=400, detail="Either file_path or dir_path must be provided")

        # Process documents
        total_chunks = 0
        for file_path in files_to_process:
            try:
                chunks = process_document(
                    str(file_path),
                    chunker_type=request.chunker,
                    chunk_size=request.chunk_size,
                    openai_client=openai_client if not skip_embedding else None,
                    pinecone_client=pinecone_client if not skip_embedding else None,
                    index_name=request.index_name
                )
                total_chunks += len(chunks)
                _metrics["documents_processed"] += 1
                _metrics["chunks_generated"] += len(chunks)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

        # Build response
        if skip_embedding:
            return IngestResponse(
                status="success",
                message=f"Processed {len(files_to_process)} document(s) into {total_chunks} chunks (embedding/storage skipped)",
                chunks_processed=total_chunks,
                documents_processed=len(files_to_process),
                skipped=True,
                skip_reason="⚠️ API keys not configured (OPENAI_API_KEY or PINECONE_API_KEY missing)"
            )
        else:
            return IngestResponse(
                status="success",
                message=f"Processed and stored {len(files_to_process)} document(s) into {total_chunks} chunks",
                chunks_processed=total_chunks,
                documents_processed=len(files_to_process),
                skipped=False
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents (stub for M1.4 - retrieval module).

    This endpoint is a placeholder. Actual retrieval and query functionality
    will be implemented in M1.4 Query Pipeline & Response Generation.

    Args:
        request: QueryRequest containing query string and parameters

    Returns:
        QueryResponse indicating feature not yet implemented
    """
    return QueryResponse(
        status="not_implemented",
        message="Query functionality will be implemented in M1.4 (Query Pipeline & Response Generation)",
        results=[]
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get pipeline metrics.

    Returns in-memory counters (reset on service restart).
    Safe to call even when API keys are not configured.

    Returns:
        MetricsResponse with processing statistics
    """
    # Check which API keys are configured
    openai_client, pinecone_client = get_clients()

    return MetricsResponse(
        status="ok",
        total_documents_processed=_metrics["documents_processed"],
        total_chunks_generated=_metrics["chunks_generated"],
        api_keys_configured={
            "openai": openai_client is not None,
            "pinecone": pinecone_client is not None
        }
    )
