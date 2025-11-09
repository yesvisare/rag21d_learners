"""
Main FastAPI application for M2.2 Prompt Optimization & Model Selection.

This is a thin wrapper that imports and includes the module router.
Run with: uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.m2_2_prompt_optimization.router import router

# Create FastAPI app
app = FastAPI(
    title="M2.2 Prompt Optimization & Model Selection",
    description="""
    Production-ready prompt optimization API for RAG systems.

    **Features:**
    - Intelligent model routing based on query complexity
    - A/B testing framework for prompt templates
    - Token estimation and cost projection
    - 5 production-tested prompt templates

    **Learn more:** See `/docs` for interactive API documentation.
    """,
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "M2.2 Prompt Optimization & Model Selection",
        "version": "2.2.0",
        "module": "m2_2_prompt_optimization",
        "endpoints": {
            "health": "/m2_2_prompt_optimization/health",
            "route": "/m2_2_prompt_optimization/route",
            "compare": "/m2_2_prompt_optimization/compare",
            "metrics": "/m2_2_prompt_optimization/metrics",
            "docs": "/docs",
        },
        "description": "Reduce RAG LLM costs by 30-50% through intelligent prompt optimization"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
