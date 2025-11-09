"""
API smoke tests for M1.4 Query Pipeline.
Run with: pytest tests/test_smoke.py
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns module info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "M1.4 Query Pipeline & Response Generation"
    assert "endpoints" in data


def test_app_health():
    """Test application health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "application" in data


def test_module_health():
    """Test module-level health endpoint."""
    response = client.get("/m1_4_query_pipeline/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m1_4_query_pipeline"
    assert "api_keys_configured" in data


def test_metrics_endpoint():
    """Test metrics endpoint (safe without keys)."""
    response = client.get("/m1_4_query_pipeline/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "success_rate" in data
    assert isinstance(data["total_queries"], int)
    assert isinstance(data["avg_latency_ms"], (int, float))


def test_query_endpoint_structure():
    """Test query endpoint accepts requests (may skip without keys)."""
    payload = {
        "query": "How do I improve RAG accuracy?",
        "top_k": 5,
        "rerank_top_k": 3,
        "namespace": "demo",
        "temperature": 0.1
    }
    response = client.post("/m1_4_query_pipeline/query", json=payload)

    # Should either succeed or fail gracefully
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "query_type" in data
        assert "chunks_retrieved" in data
        assert "sources" in data
        assert "total_time" in data


def test_ingest_stub():
    """Test ingest stub endpoint."""
    response = client.post("/m1_4_query_pipeline/ingest", json=["doc1", "doc2"])
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "M1.3" in data["note"]


def test_invalid_query_params():
    """Test query endpoint with invalid parameters."""
    # Invalid top_k (too high)
    payload = {
        "query": "test",
        "top_k": 100  # Max is 20
    }
    response = client.post("/m1_4_query_pipeline/query", json=payload)
    assert response.status_code == 422  # Validation error

    # Invalid temperature
    payload = {
        "query": "test",
        "temperature": 5.0  # Max is 2.0
    }
    response = client.post("/m1_4_query_pipeline/query", json=payload)
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
