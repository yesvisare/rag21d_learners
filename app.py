"""
FastAPI application for M2.1 Caching Strategies.

Thin wrapper that combines the caching router with standard FastAPI features.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.m2_1_caching.router import router as caching_router

# Create FastAPI app
app = FastAPI(
    title="M2.1 Caching Strategies API",
    description="Multi-layer Redis caching system for RAG cost reduction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (configure as needed for your deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include caching router
app.include_router(caching_router)


@app.get("/")
async def root():
    """
    Root endpoint.

    Returns:
        Welcome message and module information.
    """
    return {
        "message": "M2.1 Caching Strategies API",
        "version": "1.0.0",
        "module_endpoint": "/m2_1_caching",
        "docs": "/docs",
        "health": "/m2_1_caching/health"
    }


@app.get("/health")
async def global_health():
    """
    Global health check endpoint.

    Returns:
        Application health status.
    """
    return {"status": "ok", "service": "m2_1_caching_api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
