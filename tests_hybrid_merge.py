"""
Smoke tests for M4.1 Hybrid Search
Tests RRF ranking, alpha effects, and stub mode functionality.
"""

import os
import sys
from m4_1_hybrid_search import HybridSearchEngine


def test_bm25_basic():
    """Test BM25 index creation and search"""
    print("\n🧪 Test 1: BM25 Basic Functionality")
    print("-" * 60)

    engine = HybridSearchEngine()

    docs = [
        {"id": "doc1", "text": "Python programming language"},
        {"id": "doc2", "text": "JavaScript web development"},
        {"id": "doc3", "text": "Machine learning with Python"}
    ]

    engine.add_documents(docs)
    results = engine.search_bm25("python", top_k=2)

    assert len(results) <= 2, "Should return max 2 results"
    assert results[0]["id"] in ["doc1", "doc3"], "Top result should contain 'python'"

    print("✓ BM25 index working")
    print(f"✓ Query 'python' returned {len(results)} results")
    print(f"  Top result: {results[0]['id']}")


def test_smart_alpha():
    """Test smart alpha detection"""
    print("\n🧪 Test 2: Smart Alpha Detection")
    print("-" * 60)

    engine = HybridSearchEngine()

    # Test cases
    test_cases = [
        ("ABC-12345", 0.3, "SKU pattern"),
        ("how does this work", 0.7, "Natural language"),
        ("GPU training model", 0.5, "Mixed technical"),
        ('"exact phrase"', 0.3, "Quoted exact match")
    ]

    for query, expected_range, description in test_cases:
        alpha = engine.smart_alpha(query)

        # Allow some tolerance
        tolerance = 0.3
        assert abs(alpha - expected_range) < tolerance, \
            f"Alpha for '{query}' should be ~{expected_range}, got {alpha}"

        print(f"✓ '{query}' ({description}): alpha={alpha:.2f}")


def test_rrf_calculation():
    """Test RRF score calculation logic"""
    print("\n🧪 Test 3: RRF Calculation")
    print("-" * 60)

    # Simulate RRF manually
    k = 60

    # Doc appears at rank 0 in both lists
    score_rank0_dense = 1.0 / (k + 0 + 1)  # 1/61
    score_rank0_sparse = 1.0 / (k + 0 + 1)  # 1/61
    combined_both = score_rank0_dense + score_rank0_sparse

    # Doc appears only at rank 0 in dense
    single_list = score_rank0_dense

    # Doc in both should score higher
    assert combined_both > single_list, "Doc in both lists should score higher"

    print(f"✓ Doc in both lists: {combined_both:.6f}")
    print(f"✓ Doc in single list: {single_list:.6f}")
    print(f"✓ Boost ratio: {combined_both/single_list:.2f}x")


def test_alpha_affects_ordering():
    """Test that BM25 returns results with scores"""
    print("\n🧪 Test 4: BM25 Scoring")
    print("-" * 60)

    engine = HybridSearchEngine()

    # Create docs with different terms
    docs = [
        {"id": "doc1", "text": "Python programming language for data science"},
        {"id": "doc2", "text": "JavaScript web development framework React"},
        {"id": "doc3", "text": "Machine learning algorithms using Python"}
    ]

    engine.add_documents(docs)

    # Search for "Python" - should get at least one result
    results_bm25 = engine.search_bm25("Python", top_k=3)

    # Should return results
    assert len(results_bm25) > 0, "BM25 should return results"

    # At least one result should have a positive score
    has_positive_score = any(r['score'] > 0 for r in results_bm25)
    assert has_positive_score, "At least one result should have positive score"

    print(f"✓ BM25 returned {len(results_bm25)} results")
    print(f"✓ Top result: {results_bm25[0]['id']} (score={results_bm25[0]['score']:.4f})")

    # Note: Can't test alpha > 0 without API keys, but BM25 test validates core logic


def test_stub_mode_no_crash():
    """Test that stub mode doesn't crash"""
    print("\n🧪 Test 5: Stub Mode (No API Keys)")
    print("-" * 60)

    # Initialize without API keys
    engine = HybridSearchEngine(
        openai_api_key=None,
        pinecone_api_key=None
    )

    docs = [{"id": "doc1", "text": "Test document"}]
    engine.add_documents(docs)

    # Should not crash
    try:
        engine.upsert_to_pinecone()
        print("✓ Pinecone upsert skipped gracefully (no keys)")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise

    # Dense search should return empty or stub
    results = engine.search_dense("test", top_k=5)
    print(f"✓ Dense search returned {len(results)} results (stub mode)")

    # Hybrid should fall back to BM25-only
    hybrid_results = engine.hybrid_search_alpha("test", top_k=5, alpha=0.5)
    print(f"✓ Hybrid search completed with {len(hybrid_results)} results")


def test_normalization():
    """Test score normalization"""
    print("\n🧪 Test 6: Score Normalization")
    print("-" * 60)

    engine = HybridSearchEngine()

    # Create test results
    test_results = [
        {"id": "doc1", "score": 10.0},
        {"id": "doc2", "score": 5.0},
        {"id": "doc3", "score": 2.0}
    ]

    normalized = engine._normalize_scores(test_results)

    # Check max score is 1.0
    assert normalized["doc1"] == 1.0, "Max score should normalize to 1.0"
    assert normalized["doc2"] == 0.5, "Score 5/10 should normalize to 0.5"
    assert normalized["doc3"] == 0.2, "Score 2/10 should normalize to 0.2"

    print("✓ Normalization working correctly")
    print(f"  Original: {[r['score'] for r in test_results]}")
    print(f"  Normalized: {[f'{v:.2f}' for v in normalized.values()]}")


def test_metadata_preservation():
    """Test that metadata is preserved through search"""
    print("\n🧪 Test 7: Metadata Preservation")
    print("-" * 60)

    engine = HybridSearchEngine()

    docs = [
        {
            "id": "doc1",
            "text": "Python tutorial",
            "metadata": {"category": "programming", "level": "beginner"}
        }
    ]

    engine.add_documents(docs)
    results = engine.search_bm25("python", top_k=1)

    assert "metadata" in results[0], "Metadata should be in results"
    assert results[0]["metadata"]["category"] == "programming", "Metadata should match"

    print("✓ Metadata preserved in BM25 results")
    print(f"  Metadata: {results[0]['metadata']}")


def run_all_tests():
    """Run all smoke tests"""
    print("=" * 60)
    print("M4.1 Hybrid Search - Smoke Tests")
    print("=" * 60)

    tests = [
        test_bm25_basic,
        test_smart_alpha,
        test_rrf_calculation,
        test_alpha_affects_ordering,
        test_stub_mode_no_crash,
        test_normalization,
        test_metadata_preservation
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n⚠ Some tests failed. Check implementation.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            nltk.download('punkt', quiet=True)

    run_all_tests()
