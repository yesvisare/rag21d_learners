"""
FastAPI application with API key auth, rate limiting, and security headers
"""
import time
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from src.m3_3_api_security.auth import verify_api_key, key_manager
from src.m3_3_api_security.limits import check_rate_limit_dependency
from src.m3_3_api_security.security import (
    add_security_headers,
    configure_cors,
    QueryRequest,
    HealthResponse,
    AuditLogger
)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RAG API with Security",
    description="M3.3 - API Development & Security demonstration",
    version="1.0.0"
)

# Configure CORS
configure_cors(app)

# Add security headers middleware
app.middleware("http")(add_security_headers)


# Public endpoints (no auth required)
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.time()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.time()
    )


# Admin endpoints (require admin secret)
@app.post("/admin/keys/generate")
async def generate_api_key(name: str, admin_secret: str):
    """
    Generate a new API key (admin only).

    WARNING: The key is returned ONCE. Save it securely!
    """
    try:
        api_key = key_manager.generate_key(name, admin_secret)
        AuditLogger.log_auth_event("KEY_GENERATED", name, True)
        return {
            "api_key": api_key,
            "name": name,
            "warning": "Store this key securely. It cannot be retrieved later."
        }
    except ValueError as e:
        AuditLogger.log_auth_event("KEY_GENERATION_FAILED", name, False)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@app.post("/admin/keys/revoke")
async def revoke_api_key(api_key: str, admin_secret: str):
    """Revoke an API key (admin only)."""
    try:
        success = key_manager.revoke_key(api_key, admin_secret)
        if success:
            AuditLogger.log_auth_event("KEY_REVOKED", api_key[:10] + "...", True)
            return {"status": "revoked"}
        else:
            return {"status": "not_found"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@app.get("/admin/keys/list")
async def list_api_keys(admin_secret: str):
    """List all API keys (admin only, does not expose actual keys)."""
    try:
        keys = key_manager.list_keys(admin_secret)
        return {"keys": keys, "count": len(keys)}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


# Protected endpoints (require API key + rate limiting)
@app.post("/query")
async def query_rag(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
    rate_limit_info: dict = Depends(check_rate_limit_dependency)
):
    """
    Query the RAG system (requires valid API key).

    Returns:
        - answer: Generated response
        - sources: Retrieved documents
        - rate_limit_info: Remaining quota
    """
    start_time = time.time()

    try:
        # Simulate RAG processing (replace with actual RAG logic)
        result = {
            "answer": f"Mock response for: {request.query}",
            "sources": [
                {"text": "Mock source 1", "score": 0.95},
                {"text": "Mock source 2", "score": 0.87}
            ][:request.top_k],
            "rate_limit": {
                "tokens_remaining": rate_limit_info.get("tokens_remaining"),
                "hour_remaining": rate_limit_info.get("hour_remaining")
            }
        }

        # Audit log
        response_time = (time.time() - start_time) * 1000
        AuditLogger.log_request(
            api_key_hash=api_key[:16],
            endpoint="/query",
            status_code=200,
            response_time_ms=response_time,
            metadata={"query_length": len(request.query)}
        )

        return result

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        AuditLogger.log_request(
            api_key_hash=api_key[:16],
            endpoint="/query",
            status_code=500,
            response_time_ms=response_time,
            metadata={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your request"
        )


@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get usage statistics for the authenticated API key."""
    # In production, fetch from database
    return {
        "message": "Stats endpoint (implement based on your needs)",
        "note": "Would show request counts, quota usage, etc."
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all error handler (never leaks internal details)."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please contact support.",
            "error_code": "INTERNAL_ERROR"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
