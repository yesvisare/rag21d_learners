"""
M3.2 Cloud Deployment - Sanity Tests
Smoke tests for health endpoints and environment validation
"""

import os
import sys
from fastapi.testclient import TestClient

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m3_2_deploy import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    print("✅ Health check passed")


def test_root_endpoint():
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert data["status"] == "running"
    print("✅ Root endpoint passed")


def test_readiness_check():
    """Test readiness endpoint"""
    response = client.get("/ready")
    # Accept both 200 (ready) and 503 (not ready) in tests
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "checks" in data
    print(f"✅ Readiness check passed (status: {data['status']})")


def test_env_check():
    """Test environment info endpoint"""
    response = client.get("/env-check")
    assert response.status_code == 200
    data = response.json()
    assert "platform" in data
    assert "env_vars_present" in data
    print("✅ Environment check passed")


def test_environment_variables():
    """Validate environment variable setup"""
    # Check if critical env vars are accessible
    admin_secret = os.getenv("ADMIN_SECRET")
    if admin_secret:
        assert admin_secret != "", "ADMIN_SECRET should not be empty"
        print(f"✅ ADMIN_SECRET is set")
    else:
        print("⚠️  ADMIN_SECRET not set (ok for local dev)")

    # Check PORT
    port = os.getenv("PORT", "8000")
    assert port.isdigit(), "PORT must be numeric"
    print(f"✅ PORT is valid: {port}")


if __name__ == "__main__":
    print("\n🧪 Running M3.2 Deployment Sanity Tests\n")

    try:
        test_health_endpoint()
        test_root_endpoint()
        test_readiness_check()
        test_env_check()
        test_environment_variables()

        print("\n✅ All sanity tests passed!\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
