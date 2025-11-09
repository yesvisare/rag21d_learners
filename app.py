"""
M1.4 Query Pipeline & Response Generation - FastAPI Application

Thin wrapper around the query pipeline module, providing REST API endpoints.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.m1_4_query_pipeline.router import router as query_pipeline_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="M1.4 Query Pipeline & Response Generation",
    description="Production RAG query pipeline with 7-stage processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_pipeline_router)


@app.get("/")
async def root():
    """Root endpoint with module information."""
    return {
        "module": "M1.4 Query Pipeline & Response Generation",
        "version": "1.0.0",
        "description": "Complete 7-stage RAG pipeline: Query→Retrieval→Rerank→Context→LLM→Answer",
        "endpoints": {
            "health": "GET /m1_4_query_pipeline/health",
            "query": "POST /m1_4_query_pipeline/query",
            "metrics": "GET /m1_4_query_pipeline/metrics",
            "ingest_stub": "POST /m1_4_query_pipeline/ingest"
        },
        "docs": "/docs",
        "cli_usage": "python -m src.m1_4_query_pipeline.module --ask 'Your question'"
    }


@app.get("/health")
async def app_health():
    """Application-level health check."""
    return {
        "status": "ok",
        "application": "M1.4 Query Pipeline API"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
