#!/usr/bin/env python3
"""
M3.1 Docker Containerization - Smoke Tests
Basic functionality tests using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test that the root endpoint returns 200 and expected content."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint(client):
    """Test that the health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data
    assert "redis_configured" in data


def test_query_endpoint_success(client):
    """Test that the query endpoint accepts valid requests."""
    payload = {
        "question": "What is machine learning?",
        "max_sources": 3
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "source_count" in data
    assert "message" in data
    assert data["source_count"] == 3


def test_query_endpoint_default_max_sources(client):
    """Test that max_sources defaults to 3 if not provided."""
    payload = {
        "question": "Test question"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source_count"] == 3


def test_query_endpoint_validation(client):
    """Test that the query endpoint validates required fields."""
    # Missing question field
    payload = {
        "max_sources": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422  # Validation error


def test_openapi_docs_available(client):
    """Test that OpenAPI documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_available(client):
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
