"""
Smoke tests for FastAPI endpoints.

Tests basic API functionality without requiring API keys or external services.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    print("✓ Root endpoint returns API info")


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "document-processing-api"
    print("✓ Health endpoint OK")


def test_api_health_endpoint():
    """Test API health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m1_3_document_processing"
    print("✓ API v1 health endpoint OK")


def test_metrics_endpoint():
    """Test metrics endpoint (safe without API keys)."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_documents_processed" in data
    assert "total_chunks_generated" in data
    assert "api_keys_configured" in data
    assert isinstance(data["api_keys_configured"], dict)
    print("✓ Metrics endpoint returns stats")


def test_query_endpoint_stub():
    """Test query endpoint returns not implemented."""
    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_implemented"
    assert "M1.4" in data["message"]
    print("✓ Query endpoint returns stub message")


def test_ingest_endpoint_missing_path():
    """Test ingest endpoint validation."""
    response = client.post(
        "/api/v1/ingest",
        json={"chunker": "semantic"}
    )
    assert response.status_code == 400
    assert "file_path or dir_path" in response.json()["detail"]
    print("✓ Ingest endpoint validates required fields")


def test_ingest_endpoint_file_not_found():
    """Test ingest endpoint handles missing files."""
    response = client.post(
        "/api/v1/ingest",
        json={"file_path": "nonexistent.txt"}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    print("✓ Ingest endpoint handles missing files")


def test_ingest_endpoint_success():
    """Test ingest endpoint with real file (may skip embedding)."""
    response = client.post(
        "/api/v1/ingest",
        json={
            "file_path": "data/example/example_data.txt",
            "chunker": "semantic",
            "chunk_size": 512
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_processed"] > 0
    assert data["documents_processed"] == 1
    # May or may not skip based on API keys
    print(f"✓ Ingest endpoint processed document (skipped={data['skipped']})")


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("\n=== Running FastAPI Smoke Tests ===\n")

    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_api_health_endpoint,
        test_metrics_endpoint,
        test_query_endpoint_stub,
        test_ingest_endpoint_missing_path,
        test_ingest_endpoint_file_not_found,
        test_ingest_endpoint_success,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n=== Smoke Tests: {passed} passed, {failed} failed ===\n")


if __name__ == '__main__':
    run_all_smoke_tests()
