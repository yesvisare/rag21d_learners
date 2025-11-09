"""
Smoke tests for M1.2 Pinecone Advanced Indexing.

Tests cover:
1. Config loading
2. Smart alpha selector logic
3. Metadata size validator
4. Safe batch upsert error handling
5. Hybrid query execution (skips if no keys)

Run with: python tests/test_hybrid.py
Or: pytest tests/test_hybrid.py -v
"""

import sys
from src.m1_2_pinecone_hybrid.module import (
    smart_alpha_selector,
    validate_metadata_size,
    upsert_hybrid_vectors,
    hybrid_query,
    check_bm25_fitted
)
from src.m1_2_pinecone_hybrid.config import get_clients, OPENAI_MODEL, PINECONE_INDEX


def test_config_loads():
    """Test 1: Config module loads successfully."""
    print("Test 1: Config loading...")
    try:
        openai_client, pinecone_client = get_clients()
        assert OPENAI_MODEL == "text-embedding-3-small"
        assert PINECONE_INDEX == "hybrid-rag"
        print("  ✓ Config loaded successfully")
        print(f"    Model: {OPENAI_MODEL}")
        print(f"    Index: {PINECONE_INDEX}")
        print(f"    Clients: OpenAI={openai_client is not None}, Pinecone={pinecone_client is not None}")
        return True
    except Exception as e:
        print(f"  ❌ Config load failed: {e}")
        return False


def test_smart_alpha_selector():
    """Test 2: Smart alpha selector returns valid floats."""
    print("\nTest 2: Smart alpha selector...")
    try:
        # Keyword-heavy query
        alpha_keyword = smart_alpha_selector("product ID 12345")
        assert isinstance(alpha_keyword, float)
        assert 0.0 <= alpha_keyword <= 1.0
        assert alpha_keyword <= 0.4, f"Expected keyword-heavy (<=0.4), got {alpha_keyword}"
        print(f"  ✓ Keyword query → α={alpha_keyword}")

        # Semantic-heavy query
        alpha_semantic = smart_alpha_selector("explain the philosophical implications of machine learning")
        assert isinstance(alpha_semantic, float)
        assert 0.0 <= alpha_semantic <= 1.0
        assert alpha_semantic >= 0.6, f"Expected semantic-heavy (>=0.6), got {alpha_semantic}"
        print(f"  ✓ Semantic query → α={alpha_semantic}")

        # Balanced query
        alpha_balanced = smart_alpha_selector("machine learning models")
        assert isinstance(alpha_balanced, float)
        assert 0.0 <= alpha_balanced <= 1.0
        print(f"  ✓ Balanced query → α={alpha_balanced}")

        return True
    except Exception as e:
        print(f"  ❌ Alpha selector failed: {e}")
        return False


def test_metadata_size_validator():
    """Test 3: Metadata size validator catches oversized metadata."""
    print("\nTest 3: Metadata size validator...")
    try:
        # Small metadata (should pass)
        small_meta = {"text": "short text", "id": "123", "source": "test"}
        validate_metadata_size(small_meta, max_size=40960)
        print(f"  ✓ Small metadata passed (< 40KB)")

        # Large metadata (should fail)
        large_meta = {"text": "x" * 50000, "more": "data"}
        try:
            validate_metadata_size(large_meta, max_size=40960)
            print(f"  ❌ Large metadata should have raised ValueError")
            return False
        except ValueError as e:
            print(f"  ✓ Large metadata caught: {str(e)[:60]}...")
            return True

    except Exception as e:
        print(f"  ❌ Metadata validator test failed: {e}")
        return False


def test_safe_batch_upsert():
    """Test 4: Safe batch upsert reports failures cleanly."""
    print("\nTest 4: Safe batch upsert...")
    try:
        # Test with small sample
        test_docs = ["Test doc 1", "Test doc 2"]
        result = upsert_hybrid_vectors(test_docs, namespace="test-namespace")

        # Should return dict with keys
        assert isinstance(result, dict)
        assert "success" in result or "skipped" in result

        if "skipped" in result:
            print(f"  ✓ Skipped upsert (no API keys): {result['skipped']} docs")
        else:
            print(f"  ✓ Upsert result: success={result.get('success', 0)}, failed={result.get('failed', 0)}")
            if result.get("failed_ids"):
                print(f"    Failed IDs: {result['failed_ids']}")

        return True
    except Exception as e:
        print(f"  ❌ Batch upsert test failed: {e}")
        return False


def test_hybrid_query():
    """Test 5: Hybrid query prints at least 1 line (skips if no keys)."""
    print("\nTest 5: Hybrid query execution...")
    try:
        query = "machine learning models"

        # Capture if query executes (even if returns empty)
        results = hybrid_query(query, alpha=0.5, top_k=3, namespace="demo")

        # Should return list (empty if no keys/data)
        assert isinstance(results, list)

        if len(results) == 0:
            print(f"  ✓ Query executed (0 results, likely no data or keys)")
        else:
            print(f"  ✓ Query returned {len(results)} results")
            for i, res in enumerate(results[:3], 1):
                print(f"    {i}. [{res.get('score', 0):.4f}] {res.get('text', '')[:50]}...")

        return True
    except Exception as e:
        print(f"  ❌ Hybrid query test failed: {e}")
        return False


def run_all_tests():
    """Run all smoke tests and report summary."""
    print("=" * 60)
    print("M1.2 Hybrid Search Smoke Tests")
    print("=" * 60)

    tests = [
        test_config_loads,
        test_smart_alpha_selector,
        test_metadata_size_validator,
        test_safe_batch_upsert,
        test_hybrid_query
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Unexpected error in {test.__name__}: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
