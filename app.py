"""
M2.4 — Error Handling & Reliability FastAPI Application

Main application entry point for demonstration endpoints.
Run with: uvicorn app:app --reload

No external API keys required for demo endpoints.
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.m2_4_error_handling.router import router as m2_4_router

# Initialize FastAPI app
app = FastAPI(
    title="M2.4 — Error Handling & Reliability",
    description=(
        "Production-ready resilience patterns for RAG systems. "
        "Demonstrates retry strategies, circuit breakers, graceful degradation, "
        "and request queueing—all without requiring external API keys."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include M2.4 router
app.include_router(m2_4_router)


@app.get("/")
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Global health check endpoint."""
    return {
        "status": "ok",
        "module": "M2.4 — Error Handling & Reliability",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
