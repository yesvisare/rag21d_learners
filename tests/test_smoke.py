"""
Smoke tests for FastAPI endpoints.

Run with: pytest tests/test_smoke.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "m2_2_prompt_optimization"
    assert data["version"] == "2.2.0"
    assert "endpoints" in data


def test_health():
    """Test health check endpoint."""
    response = client.get("/m2_2_prompt_optimization/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["module"] == "m2_2_prompt_optimization"


def test_route_simple_query():
    """Test model routing with a simple query."""
    response = client.post(
        "/m2_2_prompt_optimization/route",
        json={
            "query": "What is your return policy?",
            "context": "We allow 30-day returns."
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert "model" in data
    assert "tier" in data
    assert "complexity_score" in data
    assert "reason" in data

    # Simple query should route to FAST tier
    assert data["tier"] in ["FAST", "BALANCED"]


def test_route_complex_query():
    """Test model routing with a complex query."""
    response = client.post(
        "/m2_2_prompt_optimization/route",
        json={
            "query": "Compare and analyze the performance differences between our Q3 and Q4 results, explaining the key factors that drive these variations.",
            "context": "Q3 revenue was $5M, Q4 revenue was $7M..."
        }
    )
    assert response.status_code == 200
    data = response.json()

    # Complex query should have higher complexity score
    assert data["complexity_score"] >= 3


def test_route_forced_tier():
    """Test forcing a specific model tier."""
    response = client.post(
        "/m2_2_prompt_optimization/route",
        json={
            "query": "Test query",
            "force_tier": "PREMIUM"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "PREMIUM"
    assert data["reason"] == "forced_selection"


def test_route_invalid_tier():
    """Test error handling for invalid tier."""
    response = client.post(
        "/m2_2_prompt_optimization/route",
        json={
            "query": "Test query",
            "force_tier": "INVALID"
        }
    )
    assert response.status_code == 400
    assert "Invalid tier" in response.json()["detail"]


def test_compare_templates():
    """Test template comparison endpoint."""
    response = client.post(
        "/m2_2_prompt_optimization/compare",
        json={
            "templates": ["baseline_comparison", "cost_optimization"],
            "test_cases": [
                {"question": "What is your return policy?"}
            ],
            "documents": [
                {"content": "Our return policy allows returns within 30 days.", "score": 0.95}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 2  # Should compare 2 templates

    # Verify result structure
    for result in data["results"]:
        assert "template_name" in result
        assert "avg_input_tokens" in result
        assert "avg_cost_per_query" in result


def test_compare_invalid_template():
    """Test error handling for invalid template name."""
    response = client.post(
        "/m2_2_prompt_optimization/compare",
        json={
            "templates": ["invalid_template_name"],
            "test_cases": [{"question": "Test?"}],
            "documents": [{"content": "Test content"}]
        }
    )
    assert response.status_code == 400
    assert "Unknown template" in response.json()["detail"]


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/m2_2_prompt_optimization/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "note" in data  # Placeholder for now


if __name__ == "__main__":
    print("Running smoke tests...")
    print("\nTest 1: Root endpoint...")
    test_root()
    print("✓ Passed")

    print("\nTest 2: Health check...")
    test_health()
    print("✓ Passed")

    print("\nTest 3: Route simple query...")
    test_route_simple_query()
    print("✓ Passed")

    print("\nTest 4: Route complex query...")
    test_route_complex_query()
    print("✓ Passed")

    print("\nTest 5: Force model tier...")
    test_route_forced_tier()
    print("✓ Passed")

    print("\nTest 6: Invalid tier...")
    test_route_invalid_tier()
    print("✓ Passed")

    print("\nTest 7: Compare templates...")
    test_compare_templates()
    print("✓ Passed")

    print("\nTest 8: Invalid template...")
    test_compare_invalid_template()
    print("✓ Passed")

    print("\nTest 9: Metrics...")
    test_metrics()
    print("✓ Passed")

    print("\n" + "="*60)
    print("✅ All smoke tests passed!")
    print("="*60)
