#!/usr/bin/env python3
"""
M3.1 Docker Containerization - FastAPI Application
Minimal RAG shim with health checks and CLI interface.
"""

import os
import sys
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Production API - Docker Demo",
    description="Containerized RAG system with health checks",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    environment: str
    redis_configured: bool


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    max_sources: Optional[int] = 3


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    source_count: int
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, any]:
    """
    Health check endpoint for Docker healthcheck.

    Returns:
        Dict with health status information
    """
    redis_host = os.getenv("REDIS_HOST")
    env_name = os.getenv("ENVIRONMENT", "development")

    logger.info(f"Health check called - Environment: {env_name}")

    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": env_name,
        "redis_configured": redis_host is not None
    }


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "RAG Production API - Docker Containerization Demo",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> Dict[str, any]:
    """
    Query the RAG system (demo implementation).

    Args:
        request: Query request with question and parameters

    Returns:
        Query response with answer and metadata

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Processing query: {request.question[:50]}...")

        # Demo response - in production, this would call actual RAG pipeline
        return {
            "answer": f"Demo response for: {request.question}",
            "source_count": request.max_sources,
            "message": "This is a demo implementation. Connect RAG pipeline for real queries."
        }
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Run the FastAPI server with uvicorn.

    Args:
        host: Host to bind to (default: 0.0.0.0 for Docker)
        port: Port to bind to (default: 8000)
    """
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def check_health() -> int:
    """
    Perform health check (for CLI usage).

    Returns:
        Exit code (0 = healthy, 1 = unhealthy)
    """
    import httpx

    try:
        port = int(os.getenv("PORT", "8000"))
        response = httpx.get(f"http://localhost:{port}/health", timeout=5.0)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"Health check: {data['status']}")
            print(f"✓ Status: {data['status']}")
            print(f"✓ Version: {data['version']}")
            print(f"✓ Environment: {data['environment']}")
            return 0
        else:
            logger.error(f"Health check failed: HTTP {response.status_code}")
            print(f"✗ Health check failed: HTTP {response.status_code}")
            return 1

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        print(f"✗ Health check error: {str(e)}")
        return 1


def main() -> None:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Production API - Docker Containerization Demo"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check application health"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    args = parser.parse_args()

    if args.health:
        sys.exit(check_health())
    elif args.serve:
        run_server(host=args.host, port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
