"""
FastAPI application for M1.2 Pinecone Hybrid Search.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Or on Windows PowerShell:
    $env:PYTHONPATH="$PWD"; uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.m1_2_pinecone_hybrid.router import router as hybrid_router

# Create FastAPI app
app = FastAPI(
    title="M1.2 — Pinecone Hybrid Search API",
    description="""
    Production-ready hybrid search API combining dense (OpenAI) and sparse (BM25) vectors.

    **Features:**
    - Hybrid vector search with alpha-weighted blending
    - Multi-tenant namespace isolation
    - Smart alpha selection (keyword vs semantic)
    - Production error handling (5 failure modes)
    - Graceful degradation when API keys missing

    **Endpoints:**
    - `GET /m1_2/health` — Health check and readiness
    - `POST /m1_2/ingest` — Ingest documents with hybrid vectors
    - `POST /m1_2/query` — Execute hybrid search query
    - `GET /m1_2/metrics` — Metrics stub

    **Quick Start:**
    1. Configure `.env` with OPENAI_API_KEY and PINECONE_API_KEY
    2. POST to `/m1_2/ingest` with documents
    3. POST to `/m1_2/query` with search query

    See `/docs` for interactive API documentation.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include router
app.include_router(hybrid_router)


@app.get("/")
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def global_health():
    """Global health check."""
    return {
        "status": "ok",
        "service": "M1.2 Pinecone Hybrid Search API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting M1.2 Pinecone Hybrid Search API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
