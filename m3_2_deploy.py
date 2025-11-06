"""
M3.2 Cloud Deployment - FastAPI Application
Supports Railway and Render platforms with health checks
"""

import os
import sys
from typing import Dict
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="M3.2 Cloud Deployment Demo",
    description="Railway/Render deployment example with health endpoints",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint - basic welcome message"""
    return {
        "message": "M3.2 Cloud Deployment Active",
        "platform": os.getenv("PLATFORM", "unknown"),
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers
    Returns 200 if service is up
    """
    return {"status": "healthy", "service": "m3.2-deploy"}


@app.get("/ready")
async def readiness_check(response: Response):
    """
    Readiness check - validates environment and dependencies
    Returns 200 if ready to serve traffic, 503 otherwise
    """
    checks = {
        "environment": False,
        "secrets": False
    }

    # Check critical environment variables
    admin_secret = os.getenv("ADMIN_SECRET")
    if admin_secret and admin_secret != "changeme":
        checks["secrets"] = True

    # Check platform detection
    platform = os.getenv("PLATFORM") or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER")
    if platform:
        checks["environment"] = True

    all_ready = all(checks.values())

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "checks": checks,
            "warning": "Some checks failed - review environment variables"
        }

    return {
        "status": "ready",
        "checks": checks,
        "platform": platform
    }


@app.get("/env-check")
async def environment_info():
    """
    Display environment information (safe subset)
    Never expose actual secret values
    """
    return {
        "platform": os.getenv("PLATFORM", "local"),
        "railway_env": os.getenv("RAILWAY_ENVIRONMENT"),
        "render_detected": bool(os.getenv("RENDER")),
        "python_version": sys.version.split()[0],
        "env_vars_present": {
            "ADMIN_SECRET": bool(os.getenv("ADMIN_SECRET")),
            "PORT": bool(os.getenv("PORT"))
        }
    }


def get_port() -> int:
    """
    Get port from environment or default to 8000
    Railway/Render inject PORT automatically
    """
    return int(os.getenv("PORT", "8000"))


def get_host() -> str:
    """Get host - use 0.0.0.0 for cloud platforms"""
    return "0.0.0.0"


if __name__ == "__main__":
    port = get_port()
    host = get_host()

    print(f"🚀 Starting M3.2 Deployment on {host}:{port}")
    print(f"📍 Platform: {os.getenv('PLATFORM', 'local')}")
    print(f"🔍 Health check: http://{host}:{port}/health")
    print(f"✅ Ready check: http://{host}:{port}/ready")

    # Production: Use gunicorn with uvicorn workers
    # Dev: Direct uvicorn
    uvicorn.run(
        "m3_2_deploy:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
