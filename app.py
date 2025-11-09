"""
M2.3 Production Monitoring Dashboard - FastAPI Application

This application provides a REST API for monitoring RAG systems with Prometheus metrics.

Endpoints:
- GET /monitoring/health - Health check
- POST /monitoring/cost/estimate - Estimate query costs
- POST /monitoring/simulate - Generate demo metrics
- GET /monitoring/metrics-info - List available metrics

Prometheus metrics are exposed on a separate HTTP server (default: port 8000)
This FastAPI app runs on a different port (default: 8001 when using uvicorn)

Usage:
    # Start with uvicorn
    uvicorn app:app --reload --host 0.0.0.0 --port 8001

    # Or use the PowerShell script
    ./scripts/run_local.ps1

    # Access API docs at http://localhost:8001/docs
    # Access Prometheus metrics at http://localhost:8000/metrics
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.m2_3_monitoring import start_metrics_server, METRICS_PORT
from src.m2_3_monitoring.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Starts Prometheus metrics server on startup.
    """
    # Startup: Start Prometheus metrics HTTP server
    print("\n" + "="*60)
    print("M2.3 Production Monitoring Dashboard - Starting")
    print("="*60)

    success = start_metrics_server(port=METRICS_PORT)
    if success:
        print(f"✓ Prometheus metrics available at http://localhost:{METRICS_PORT}/metrics")
    else:
        print(f"⚠ Warning: Could not start metrics server on port {METRICS_PORT}")

    print("✓ FastAPI application starting...")
    print("="*60 + "\n")

    yield

    # Shutdown: cleanup if needed
    print("\n" + "="*60)
    print("M2.3 Production Monitoring Dashboard - Shutting down")
    print("="*60 + "\n")


# Create FastAPI application
app = FastAPI(
    title="M2.3 Production Monitoring Dashboard",
    description="""
    Production-grade monitoring for RAG (Retrieval-Augmented Generation) systems.

    This API provides endpoints for:
    - **Health checks** - Monitor service availability
    - **Cost estimation** - Calculate query costs based on token usage
    - **Demo simulation** - Generate sample metrics for testing dashboards
    - **Metrics info** - List all available Prometheus metrics

    ## Metrics Collection

    Prometheus metrics are exposed on a **separate HTTP server** (default: port 8000).
    Configure Prometheus to scrape `http://localhost:8000/metrics` to collect:

    - Query latency percentiles (p50/p95/p99)
    - Token usage by model
    - Per-query and cumulative costs
    - Cache hit rates
    - Error rates by type
    - Rate limit headroom
    - Response relevance scores

    ## Getting Started

    1. Start the application: `uvicorn app:app --reload --port 8001`
    2. Access API docs: http://localhost:8001/docs
    3. View Prometheus metrics: http://localhost:8000/metrics
    4. Start Prometheus/Grafana: `docker compose -f docker/docker-compose.monitoring.yml up -d`
    5. Import dashboard: Upload `grafana/grafana_dash.json` in Grafana

    ## Related Resources

    - Notebook: `notebooks/M2_3_Production_Monitoring_Dashboard.ipynb`
    - Docker Compose: `docker/docker-compose.monitoring.yml`
    - Grafana Dashboard: `grafana/grafana_dash.json`
    - Tests: `tests/test_monitoring.py`, `tests/test_smoke.py`
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Include monitoring router
app.include_router(router)


@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "M2.3 Production Monitoring Dashboard",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/monitoring/health",
            "cost_estimate": "/monitoring/cost/estimate",
            "simulate": "/monitoring/simulate",
            "metrics_info": "/monitoring/metrics-info",
            "prometheus_metrics": f"http://localhost:{METRICS_PORT}/metrics"
        },
        "documentation": "See /docs for interactive API documentation"
    }


@app.get("/health")
async def health():
    """
    Simple health check endpoint (also available at /monitoring/health).
    """
    return {"status": "ok", "service": "m2_3_monitoring"}


# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting M2.3 Production Monitoring Dashboard...")
    print("\nNote: For development, use:")
    print("  uvicorn app:app --reload --host 0.0.0.0 --port 8001")
    print("\nOr use the PowerShell script:")
    print("  ./scripts/run_local.ps1")
    print()

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
