"""
Smoke tests for M1.2 FastAPI endpoints.

Tests API health and basic functionality without requiring API keys.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_redirect():
    """Test root endpoint redirects to docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307  # Redirect
    assert "/docs" in response.headers["location"]


def test_global_health():
    """Test global health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "M1.2 Pinecone Hybrid Search API"
    assert "version" in data


def test_module_health():
    """Test module-specific health endpoint."""
    response = client.get("/m1_2/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m1_2_pinecone_hybrid"
    assert "bm25_fitted" in data
    assert "clients_available" in data


def test_metrics_endpoint():
    """Test metrics stub endpoint."""
    response = client.get("/m1_2/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "bm25_fitted" in data


def test_ingest_without_keys():
    """Test ingestion gracefully skips when no API keys."""
    payload = {
        "docs": ["Test document 1", "Test document 2"],
        "namespace": "test"
    }
    response = client.post("/m1_2/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should either skip or succeed
    assert data["status"] in ["skipped", "success", "partial"]
    if data["status"] == "skipped":
        assert "message" in data
        assert "no API keys" in data["message"] or "no keys" in data["message"]


def test_query_without_keys():
    """Test query gracefully skips when no API keys."""
    payload = {
        "query": "test query",
        "alpha": 0.5,
        "top_k": 3,
        "namespace": "test"
    }
    response = client.post("/m1_2/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should either skip or succeed (with empty results if no data)
    assert data["status"] in ["skipped", "success"]
    assert "alpha" in data
    assert data["alpha"] == 0.5


def test_query_auto_alpha():
    """Test query with automatic alpha selection."""
    payload = {
        "query": "explain machine learning concepts",
        # No alpha specified - should auto-select
        "top_k": 5,
        "namespace": "test"
    }
    response = client.post("/m1_2/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "alpha" in data
    # Alpha should be auto-selected (0.0-1.0)
    assert 0.0 <= data["alpha"] <= 1.0


def test_ingest_validation():
    """Test ingestion with invalid payload."""
    # Empty docs list
    payload = {
        "docs": [],
        "namespace": "test"
    }
    response = client.post("/m1_2/ingest", json=payload)
    assert response.status_code == 422  # Validation error


def test_query_validation():
    """Test query with invalid payload."""
    # Empty query
    payload = {
        "query": "",
        "namespace": "test"
    }
    response = client.post("/m1_2/query", json=payload)
    assert response.status_code == 422  # Validation error

    # Invalid alpha
    payload = {
        "query": "test",
        "alpha": 1.5,  # Out of range
        "namespace": "test"
    }
    response = client.post("/m1_2/query", json=payload)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
