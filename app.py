"""
FastAPI Application for RAG21D Learning Modules

Thin wrapper that imports and registers all module routers.
Run with: uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import logging

# Import routers from modules
from src.m1_1_vector_databases.router import router as m1_1_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="RAG21D Learning Modules API",
    description=(
        "REST API for RAG (Retrieval-Augmented Generation) learning modules. "
        "Each module provides hands-on experience with production RAG patterns."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include module routers
app.include_router(m1_1_router)

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    """Global health check."""
    return {
        "status": "ok",
        "service": "rag21d-learning-api",
        "version": "1.0.0",
        "modules": [
            "M1.1 - Vector Databases"
        ]
    }

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup message."""
    logger.info("=" * 70)
    logger.info("RAG21D Learning API Started")
    logger.info("=" * 70)
    logger.info("Available modules:")
    logger.info("  - M1.1 Vector Databases: /m1_1/*")
    logger.info("")
    logger.info("Documentation: http://localhost:8000/docs")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown message."""
    logger.info("RAG21D Learning API Shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
