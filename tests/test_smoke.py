"""
Smoke tests for M2.4 FastAPI endpoints.

Tests the API surface without requiring external dependencies.
Run with: pytest tests/test_smoke.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_redirects_to_docs():
    """Test that root path redirects to /docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307  # Redirect
    assert "/docs" in response.headers["location"]


def test_global_health():
    """Test global health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "module" in data
    assert "version" in data


def test_m2_4_health():
    """Test M2.4 module health endpoint."""
    response = client.get("/m2_4/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m2_4_error_handling"
    assert data["version"] == "1.0.0"


def test_circuit_state():
    """Test circuit breaker state endpoint."""
    response = client.get("/m2_4/state")
    assert response.status_code == 200

    data = response.json()
    assert "state" in data
    assert data["state"] in ["closed", "open", "half_open"]
    assert "failure_count" in data
    assert "description" in data


def test_queue_stats():
    """Test queue statistics endpoint."""
    response = client.get("/m2_4/queue/stats")
    assert response.status_code == 200

    data = response.json()
    assert "current_size" in data
    assert "processed" in data
    assert "rejected" in data
    assert "capacity" in data
    assert "utilization_pct" in data
    assert isinstance(data["utilization_pct"], (int, float))


def test_simulate_retry():
    """Test simulation endpoint with retry pattern."""
    response = client.post("/m2_4/simulate", json={
        "n": 10,
        "failure_rate": 0.3,
        "operation": "retry"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["operation"] == "retry"
    assert data["total_requests"] == 10
    assert "successful" in data
    assert "failed" in data
    assert "retried" in data
    assert "summary" in data


def test_simulate_circuit_breaker():
    """Test simulation endpoint with circuit breaker pattern."""
    # Reset state first
    client.post("/m2_4/reset")

    response = client.post("/m2_4/simulate", json={
        "n": 10,
        "failure_rate": 0.8,
        "operation": "circuit_breaker"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["operation"] == "circuit_breaker"
    assert "circuit_state" in data


def test_simulate_fallback():
    """Test simulation endpoint with fallback pattern."""
    response = client.post("/m2_4/simulate", json={
        "n": 10,
        "failure_rate": 0.5,
        "operation": "fallback"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["operation"] == "fallback"
    assert "fallback_hits" in data


def test_simulate_queue():
    """Test simulation endpoint with queue pattern."""
    response = client.post("/m2_4/simulate", json={
        "n": 10,
        "failure_rate": 0.0,
        "operation": "queue"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["operation"] == "queue"
    assert "queue_stats" in data


def test_simulate_defaults():
    """Test simulation endpoint with default parameters."""
    response = client.post("/m2_4/simulate", json={})
    assert response.status_code == 200

    data = response.json()
    assert data["total_requests"] == 10  # Default
    assert "summary" in data


def test_reset_state():
    """Test reset endpoint."""
    response = client.post("/m2_4/reset")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert data["circuit_state"] in ["closed", "open", "half_open"]
    assert "queue_size" in data


def test_simulate_validation():
    """Test simulation endpoint validation."""
    # Test n out of range
    response = client.post("/m2_4/simulate", json={"n": 0})
    assert response.status_code == 422  # Validation error

    # Test failure_rate out of range
    response = client.post("/m2_4/simulate", json={"failure_rate": 1.5})
    assert response.status_code == 422  # Validation error


def test_openapi_docs():
    """Test that OpenAPI docs are available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "M2.4 — Error Handling & Reliability"
