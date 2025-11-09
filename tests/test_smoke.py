"""
Smoke tests for M2.1 Caching API endpoints.

Tests FastAPI router health, metrics, and invalidation endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from app import app

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "m2_1_caching" in data["message"].lower()


def test_global_health_endpoint():
    """Test global health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_module_root_endpoint():
    """Test module root endpoint."""
    response = client.get("/m2_1_caching/")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "M2.1 Caching Strategies"
    assert "endpoints" in data


def test_health_check_endpoint():
    """Test module health check endpoint."""
    response = client.get("/m2_1_caching/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m2_1_caching"
    assert "redis_available" in data
    assert "openai_available" in data


def test_metrics_endpoint():
    """Test metrics endpoint returns expected structure."""
    response = client.get("/m2_1_caching/metrics")
    assert response.status_code == 200
    data = response.json()

    # Check all required fields
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data
    assert "stampede_prevented" in data
    assert "invalidations" in data
    assert "errors" in data
    assert "summary" in data

    # Check types
    assert isinstance(data["hits"], int)
    assert isinstance(data["misses"], int)
    assert isinstance(data["hit_rate"], (int, float))
    assert isinstance(data["summary"], str)


def test_invalidate_endpoint_with_valid_prefix():
    """Test invalidation endpoint with valid prefix."""
    # Only test if Redis is available
    health_response = client.get("/m2_1_caching/health")
    if not health_response.json().get("redis_available"):
        pytest.skip("Redis not available")

    response = client.post(
        "/m2_1_caching/invalidate",
        json={"prefix": "exact:"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prefix" in data
    assert "keys_invalidated" in data
    assert "message" in data
    assert data["prefix"] == "exact:"


def test_invalidate_endpoint_with_invalid_prefix():
    """Test invalidation endpoint rejects invalid prefix."""
    # Only test if Redis is available
    health_response = client.get("/m2_1_caching/health")
    if not health_response.json().get("redis_available"):
        pytest.skip("Redis not available")

    response = client.post(
        "/m2_1_caching/invalidate",
        json={"prefix": "invalid_prefix"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_invalidate_endpoint_without_redis():
    """Test invalidation fails gracefully without Redis."""
    # This test is tricky - if Redis IS available, we skip
    # If Redis is NOT available, we expect 503
    health_response = client.get("/m2_1_caching/health")
    if health_response.json().get("redis_available"):
        pytest.skip("Redis is available - cannot test failure case")

    response = client.post(
        "/m2_1_caching/invalidate",
        json={"prefix": "exact:"}
    )
    assert response.status_code == 503


def test_api_documentation_available():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
