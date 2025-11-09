#!/usr/bin/env python3
"""
M1.1 Vector Databases - Smoke Tests

Simple but useful tests to ensure basic functionality works:
- Configuration loads correctly
- Example data loads successfully
- Cosine similarity calculations work
- Basic validation of vector operations
- API endpoints respond correctly

Run: python -m pytest tests/test_smoke.py -v
Or: python tests/test_smoke.py (standalone mode)
"""

import sys
from pathlib import Path

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# TEST: Configuration Loading
# ============================================================================

def test_config_loads():
    """Test that configuration module loads without errors."""
    print("Testing configuration loading...", end=" ")
    try:
        from src.m1_1_vector_databases import config
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_config_constants():
    """Test that required configuration constants are defined."""
    print("Testing configuration constants...", end=" ")
    try:
        from src.m1_1_vector_databases import config

        required_constants = [
            'EMBEDDING_MODEL',
            'EMBEDDING_DIM',
            'INDEX_NAME',
            'DEFAULT_NAMESPACE',
            'SCORE_THRESHOLD',
            'BATCH_SIZE'
        ]

        missing = []
        for const in required_constants:
            if not hasattr(config, const):
                missing.append(const)

        if missing:
            print(f"✗ FAIL: Missing constants: {missing}")
            return False

        print("✓ PASS")
        return True

    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


# ============================================================================
# TEST: Example Data Loading
# ============================================================================

def test_example_data_exists():
    """Test that example_data.txt exists."""
    print("Testing example data file exists...", end=" ")
    try:
        data_file = Path("data/example/example_data.txt")
        if not data_file.exists():
            print("✗ FAIL: data/example/example_data.txt not found")
            return False
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_load_example_texts():
    """Test that example texts load correctly."""
    print("Testing example data loading...", end=" ")
    try:
        from src.m1_1_vector_databases.module import load_example_texts

        texts = load_example_texts("data/example/example_data.txt")

        if not texts:
            print("✗ FAIL: No texts loaded")
            return False

        if len(texts) < 10:
            print(f"✗ FAIL: Too few texts loaded ({len(texts)} < 10)")
            return False

        if not all(isinstance(t, str) and t.strip() for t in texts):
            print("✗ FAIL: Invalid text format")
            return False

        print(f"✓ PASS (loaded {len(texts)} texts)")
        return True

    except FileNotFoundError:
        print("✗ FAIL: data/example/example_data.txt not found")
        return False
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


# ============================================================================
# TEST: Cosine Similarity
# ============================================================================

def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors."""
    print("Testing cosine similarity (identical vectors)...", end=" ")
    try:
        from src.m1_1_vector_databases.module import cosine_similarity

        vec1 = [1.0, 2.0, 3.0, 4.0]
        vec2 = [1.0, 2.0, 3.0, 4.0]

        similarity = cosine_similarity(vec1, vec2)

        # Should be very close to 1.0 (allowing for floating point precision)
        if abs(similarity - 1.0) > 0.0001:
            print(f"✗ FAIL: Expected ~1.0, got {similarity}")
            return False

        print("✓ PASS")
        return True

    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    print("Testing cosine similarity (orthogonal vectors)...", end=" ")
    try:
        from src.m1_1_vector_databases.module import cosine_similarity

        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]

        similarity = cosine_similarity(vec1, vec2)

        # Should be very close to 0.0 (orthogonal)
        if abs(similarity - 0.0) > 0.0001:
            print(f"✗ FAIL: Expected ~0.0, got {similarity}")
            return False

        print("✓ PASS")
        return True

    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_cosine_similarity_opposite():
    """Test cosine similarity with opposite vectors."""
    print("Testing cosine similarity (opposite vectors)...", end=" ")
    try:
        from src.m1_1_vector_databases.module import cosine_similarity

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]

        similarity = cosine_similarity(vec1, vec2)

        # Should be very close to -1.0 (opposite)
        if abs(similarity - (-1.0)) > 0.0001:
            print(f"✗ FAIL: Expected ~-1.0, got {similarity}")
            return False

        print("✓ PASS")
        return True

    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_cosine_similarity_dimension_mismatch():
    """Test that cosine similarity raises error for dimension mismatch."""
    print("Testing cosine similarity (dimension mismatch)...", end=" ")
    try:
        from src.m1_1_vector_databases.module import cosine_similarity

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]

        try:
            similarity = cosine_similarity(vec1, vec2)
            print("✗ FAIL: Should have raised ValueError")
            return False
        except ValueError:
            print("✓ PASS")
            return True

    except Exception as e:
        print(f"✗ FAIL: Unexpected error: {str(e)}")
        return False


# ============================================================================
# TEST: Dependencies
# ============================================================================

def test_dependencies_installed():
    """Test that required dependencies are installed."""
    print("Testing required dependencies...", end=" ")

    required_packages = [
        ('openai', 'OpenAI SDK'),
        ('pinecone', 'Pinecone client'),
        ('numpy', 'NumPy'),
        ('dotenv', 'python-dotenv'),
        ('tqdm', 'TQDM')
    ]

    missing = []

    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"✗ FAIL: Missing packages: {', '.join(missing)}")
        print("         Run: pip install -r requirements.txt")
        return False

    print("✓ PASS")
    return True


# ============================================================================
# TEST: API Keys (Optional - warns if missing)
# ============================================================================

def test_api_keys_configured():
    """Test that API keys are configured (warning only, not failure)."""
    print("Testing API keys configuration...", end=" ")
    try:
        import config

        has_openai = bool(config.OPENAI_API_KEY)
        has_pinecone = bool(config.PINECONE_API_KEY)

        if has_openai and has_pinecone:
            print("✓ PASS")
            return True
        else:
            missing = []
            if not has_openai:
                missing.append("OPENAI_API_KEY")
            if not has_pinecone:
                missing.append("PINECONE_API_KEY")

            print(f"⚠ WARNING: Missing {', '.join(missing)}")
            print("           Set keys in .env file to run full functionality")
            return True  # Don't fail, just warn

    except Exception as e:
        print(f"⚠ WARNING: {str(e)}")
        return True  # Don't fail on API key check


# ============================================================================
# TEST: API Endpoints
# ============================================================================

def test_api_health():
    """Test that API health endpoint responds correctly."""
    print("Testing API health endpoint...", end=" ")
    try:
        from fastapi.testclient import TestClient
        from app import app

        client = TestClient(app)
        response = client.get("/m1_1/health")

        if response.status_code != 200:
            print(f"✗ FAIL: Expected status 200, got {response.status_code}")
            return False

        data = response.json()
        if data.get("status") != "ok":
            print(f"✗ FAIL: Expected status 'ok', got {data.get('status')}")
            return False

        if data.get("module") != "m1_1_vector_databases":
            print(f"✗ FAIL: Wrong module name")
            return False

        print("✓ PASS")
        return True

    except ImportError:
        print("⚠ SKIP: FastAPI/TestClient not available")
        return True  # Don't fail, just skip
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all smoke tests."""
    print("=" * 70)
    print("M1.1 VECTOR DATABASES - SMOKE TESTS")
    print("=" * 70)
    print()

    tests = [
        # Configuration tests
        ("Config Loading", test_config_loads),
        ("Config Constants", test_config_constants),

        # Dependencies
        ("Dependencies", test_dependencies_installed),

        # Data loading tests
        ("Example Data File", test_example_data_exists),
        ("Load Example Texts", test_load_example_texts),

        # Math operations
        ("Cosine Similarity (Identical)", test_cosine_similarity_identical),
        ("Cosine Similarity (Orthogonal)", test_cosine_similarity_orthogonal),
        ("Cosine Similarity (Opposite)", test_cosine_similarity_opposite),
        ("Cosine Similarity (Dimension Error)", test_cosine_similarity_dimension_mismatch),

        # API tests
        ("API Health Endpoint", test_api_health),

        # API keys (warning only)
        ("API Keys", test_api_keys_configured),
    ]

    passed = 0
    failed = 0
    warnings = 0

    for test_name, test_func in tests:
        result = test_func()
        if result:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print()

    if failed == 0:
        print("✓ All tests passed!")
        print()
        print("Next steps:")
        print("  1. Configure API keys in .env file")
        print("  2. Run: python -m src.m1_1_vector_databases.module --init")
        print("  3. Run: python -m src.m1_1_vector_databases.module --query \"vector search\"")
        print("  4. Start API: uvicorn app:app --reload")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
