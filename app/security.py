"""
Security utilities: headers, CORS, input validation
"""
import re
from typing import Dict
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


# Security Headers Configuration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}


async def add_security_headers(request: Request, call_next):
    """Middleware to add security headers to all responses."""
    response = await call_next(request)

    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value

    return response


def configure_cors(app, origins: list = None):
    """
    Configure CORS middleware.

    Default: restrictive (only localhost)
    Production: specify exact origins
    """
    if origins is None:
        origins = [
            "http://localhost:3000",
            "http://localhost:8000"
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"]
    )


# Input Validation Models
class QueryRequest(BaseModel):
    """Validated query request with injection protection."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

    @validator("query")
    def validate_query(cls, v):
        """
        Basic injection protection.

        Notes:
        - This is NOT foolproof against prompt injection
        - LLM-level guards (OpenAI moderation, Anthropic filters) recommended
        - Consider: semantic analysis, user reputation, human-in-loop for suspicious patterns
        """
        # Strip excessive whitespace
        v = " ".join(v.split())

        # Block obvious system prompt attempts
        suspicious_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"you\s+are\s+now",
            r"system[:.]",
            r"<\s*script",
            r"javascript:",
            r"on\w+\s*="  # Event handlers
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Suspicious pattern detected in query")

        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: float


class ErrorResponse(BaseModel):
    """Standardized error response (never leaks internals)."""
    detail: str
    error_code: str = None

    @classmethod
    def from_exception(cls, exc: Exception, safe_mode: bool = True):
        """
        Create error response from exception.

        safe_mode=True: Never expose internal details
        safe_mode=False: Development mode (include traceback)
        """
        if safe_mode:
            return cls(
                detail="An error occurred. Please contact support.",
                error_code="INTERNAL_ERROR"
            )
        else:
            return cls(
                detail=str(exc),
                error_code=type(exc).__name__
            )


# Audit logging helper
class AuditLogger:
    """Simple audit logger (prints to stdout; use structured logging in production)."""

    @staticmethod
    def log_request(
        api_key_hash: str,
        endpoint: str,
        status_code: int,
        response_time_ms: float,
        metadata: Dict = None
    ):
        """Log API request for audit trail."""
        print(
            f"[AUDIT] key_hash={api_key_hash[:16]}... "
            f"endpoint={endpoint} "
            f"status={status_code} "
            f"time={response_time_ms:.2f}ms "
            f"meta={metadata or {}}"
        )

    @staticmethod
    def log_auth_event(event_type: str, key_name: str, success: bool):
        """Log authentication events."""
        print(f"[AUTH] type={event_type} key={key_name} success={success}")
