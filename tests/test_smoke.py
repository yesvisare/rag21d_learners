"""
Smoke tests for M2.3 Production Monitoring Dashboard FastAPI endpoints

Tests the REST API using FastAPI TestClient to ensure basic functionality
works without requiring external dependencies.
"""
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest


# Import the app
from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns service information"""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["service"] == "M2.3 Production Monitoring Dashboard"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert "endpoints" in data

    print("✓ Root endpoint works")


def test_health_check_root(client):
    """Test the root health endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "m2_3_monitoring"

    print("✓ Root health check works")


def test_health_check_monitoring(client):
    """Test the monitoring module health endpoint"""
    response = client.get("/monitoring/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["module"] == "m2_3_monitoring"
    assert data["version"] == "1.0.0"

    print("✓ Monitoring health check works")


def test_cost_estimate_endpoint(client):
    """Test the cost estimation endpoint"""
    payload = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "model": "gpt-4"
    }

    response = client.post("/monitoring/cost/estimate", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["input_tokens"] == 1000
    assert data["output_tokens"] == 500
    assert data["model"] == "gpt-4"
    assert "cost_usd" in data
    assert data["cost_usd"] > 0

    # Verify breakdown
    assert "breakdown" in data
    assert "input_cost_usd" in data["breakdown"]
    assert "output_cost_usd" in data["breakdown"]

    print(f"✓ Cost estimate works: ${data['cost_usd']:.6f}")


def test_cost_estimate_validation(client):
    """Test that cost estimation validates inputs"""
    # Test negative tokens
    payload = {
        "input_tokens": -100,
        "output_tokens": 500
    }

    response = client.post("/monitoring/cost/estimate", json=payload)
    assert response.status_code == 422  # Validation error

    print("✓ Cost estimate validation works")


def test_simulate_endpoint(client):
    """Test the query simulation endpoint"""
    payload = {
        "count": 3,
        "operation": "test_simulation",
        "model": "test-model"
    }

    response = client.post("/monitoring/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "queries_simulated" in data
    assert data["queries_simulated"] == 3
    assert "metrics_recorded" in data
    assert len(data["metrics_recorded"]) == 3

    # Check that each result has expected fields
    for result in data["metrics_recorded"]:
        assert "query_id" in result
        # May have 'tokens', 'relevance', 'cached' or 'error'

    print("✓ Simulation endpoint works")


def test_simulate_validation(client):
    """Test that simulation endpoint validates count range"""
    # Test count too high
    payload = {
        "count": 150  # Max is 100
    }

    response = client.post("/monitoring/simulate", json=payload)
    assert response.status_code == 422  # Validation error

    print("✓ Simulation validation works")


def test_metrics_info_endpoint(client):
    """Test the metrics info endpoint"""
    response = client.get("/monitoring/metrics-info")

    assert response.status_code == 200
    data = response.json()

    assert "total_metrics" in data
    assert "metrics" in data
    assert isinstance(data["metrics"], list)
    assert data["total_metrics"] > 0

    # Check that RAG metrics are present
    metric_names = [m["name"] for m in data["metrics"]]
    assert any("rag_" in name for name in metric_names)

    print(f"✓ Metrics info endpoint works ({data['total_metrics']} metrics)")


def test_prometheus_metrics_exposed():
    """Test that Prometheus metrics are properly formatted"""
    # Generate metrics in Prometheus format
    metrics_output = generate_latest(REGISTRY)

    assert isinstance(metrics_output, bytes)

    # Decode and check content
    metrics_str = metrics_output.decode('utf-8')

    # Should contain HELP and TYPE declarations
    assert '# HELP' in metrics_str
    assert '# TYPE' in metrics_str

    # Should contain our RAG metrics
    assert 'rag_' in metrics_str

    print("✓ Prometheus metrics are properly formatted")


def test_api_documentation(client):
    """Test that API documentation is accessible"""
    response = client.get("/docs")

    assert response.status_code == 200
    # FastAPI serves HTML for docs
    assert "text/html" in response.headers["content-type"]

    print("✓ API documentation is accessible")


def test_openapi_schema(client):
    """Test that OpenAPI schema is accessible"""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "M2.3 Production Monitoring Dashboard"
    assert "paths" in data

    # Check that our endpoints are documented
    assert "/monitoring/health" in data["paths"]
    assert "/monitoring/cost/estimate" in data["paths"]
    assert "/monitoring/simulate" in data["paths"]

    print("✓ OpenAPI schema is valid")


if __name__ == "__main__":
    # Run tests with pytest
    print("="*60)
    print("M2.3 Monitoring - FastAPI Smoke Tests")
    print("="*60 + "\n")

    pytest.main([__file__, "-v"])
