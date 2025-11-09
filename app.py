"""
FastAPI application for M1.3 Document Processing Pipeline.

This is a thin wrapper that imports the processing router and provides
a complete REST API for document ingestion and retrieval.

Run with:
    Windows: powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
    Unix: PYTHONPATH=$PWD uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.m1_3_document_processing.router import router as processing_router

# Create FastAPI app
app = FastAPI(
    title="M1.3 Document Processing Pipeline",
    description="Production RAG pipeline: Extraction → Cleaning → Chunking → Embedding → Storage",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(processing_router)


@app.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        Welcome message and available endpoints
    """
    return {
        "message": "M1.3 Document Processing Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "ingest": "/api/v1/ingest (POST)",
            "query": "/api/v1/query (POST) - stub for M1.4",
            "metrics": "/api/v1/metrics"
        }
    }


@app.get("/health")
async def health():
    """
    Top-level health check.

    Returns:
        Status OK
    """
    return {"status": "ok", "service": "document-processing-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
