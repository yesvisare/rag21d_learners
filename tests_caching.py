"""
Smoke tests for M2.1 Caching System
Tests key schema stability, stampede locks, TTL, semantic cache, and safe stubs.
"""
import time
import hashlib
from m2_1_caching import (
    CacheKeyGenerator,
    MultiLayerCache,
    CacheMetrics,
    StampedeLock
)
import config


def test_key_schema_stable():
    """Verify cache key generation is consistent."""
    keygen = CacheKeyGenerator()
    query = "test query"

    # Exact keys should be deterministic
    key1 = keygen.exact_key(query)
    key2 = keygen.exact_key(query)
    assert key1 == key2, "Exact keys not stable"

    # Should use full SHA-256
    expected_hash = hashlib.sha256(query.encode()).hexdigest()
    assert expected_hash in key1, "Not using full SHA-256"

    # Check prefix
    assert key1.startswith(config.PREFIX_EXACT), "Wrong prefix"
    print("✓ Key schema stable")


def test_stampede_lock_works():
    """Verify per-key lock prevents stampedes."""
    redis_client = config.get_redis()
    if not redis_client:
        print("⚠️ Skipping stampede test (no Redis)")
        return

    cache = MultiLayerCache(redis_client, None)
    test_key = "stampede_test_key"

    # Clean up first
    redis_client.delete(CacheKeyGenerator.lock_key(test_key))

    # Acquire lock
    with StampedeLock(redis_client, test_key) as acquired:
        assert acquired, "Failed to acquire lock"

        # Verify lock exists
        lock_key = CacheKeyGenerator.lock_key(test_key)
        assert redis_client.exists(lock_key), "Lock not created"

    # Lock should be released
    lock_key = CacheKeyGenerator.lock_key(test_key)
    assert not redis_client.exists(lock_key), "Lock not released"
    print("✓ Stampede lock works")


def test_ttl_applied():
    """Verify TTL is set correctly."""
    redis_client = config.get_redis()
    if not redis_client:
        print("⚠️ Skipping TTL test (no Redis)")
        return

    cache = MultiLayerCache(redis_client, None)
    query = "ttl test query"
    response = {"answer": "test"}

    # Set with short TTL
    cache.set_exact(query, response, ttl=2)

    # Check TTL exists
    key = CacheKeyGenerator.exact_key(query)
    ttl = redis_client.ttl(key)
    assert ttl > 0, "TTL not set"
    assert ttl <= 2, "TTL too long"

    # Wait and verify expiration
    time.sleep(3)
    assert not redis_client.exists(key), "Entry not expired"
    print("✓ TTL applied correctly")


def test_semantic_similarity_cache():
    """Verify semantic cache returns hits for paraphrases."""
    redis_client = config.get_redis()
    if not redis_client:
        print("⚠️ Skipping semantic test (no Redis)")
        return

    cache = MultiLayerCache(redis_client, None)

    # Clean semantic cache
    cache.invalidate_by_prefix(config.PREFIX_SEMANTIC)

    query1 = "How do I reset my password?"
    query2 = "How can I reset my password?"  # Very similar
    response = {"answer": "Go to settings"}

    # Store original
    cache.set_semantic(query1, response)

    # Try to retrieve with paraphrase (lower threshold for test)
    cached = cache.get_semantic(query2, threshold=0.90)
    assert cached is not None, "Semantic match failed for high similarity"
    print("✓ Semantic similarity cache works")


def test_safe_no_key_stubs():
    """Verify system handles missing services gracefully."""
    # No Redis, no OpenAI
    cache = MultiLayerCache(None, None)

    # Should not crash, just return None
    result = cache.get_exact("test")
    assert result is None, "Should return None when no Redis"

    # Set should not crash
    cache.set_exact("test", {"data": "test"})

    # Embedding without OpenAI
    embedding = cache.compute_or_get_embedding("test text")
    assert embedding is None, "Should return None when no OpenAI"

    print("✓ Safe no-key stubs work")


def test_metrics_tracking():
    """Verify metrics are tracked correctly."""
    metrics = CacheMetrics()

    # Record some events
    metrics.record_hit("exact")
    metrics.record_hit("semantic")
    metrics.record_miss("exact")

    # Check counts
    assert metrics.hits == 2, "Hit count wrong"
    assert metrics.misses == 1, "Miss count wrong"

    # Check hit rate
    hit_rate = metrics.get_hit_rate()
    assert abs(hit_rate - 66.67) < 0.1, f"Hit rate calculation wrong: {hit_rate}"

    print("✓ Metrics tracking works")


def test_context_cache_key_order():
    """Verify context cache handles different doc ID orders."""
    redis_client = config.get_redis()
    if not redis_client:
        print("⚠️ Skipping context test (no Redis)")
        return

    cache = MultiLayerCache(redis_client, None)

    # Different order, same docs
    docs1 = ["doc_a", "doc_b", "doc_c"]
    docs2 = ["doc_c", "doc_a", "doc_b"]

    # Keys should be identical (sorted internally)
    key1 = CacheKeyGenerator.context_key(docs1)
    key2 = CacheKeyGenerator.context_key(docs2)
    assert key1 == key2, "Context keys not order-independent"

    # Set and retrieve
    contexts = [{"id": "doc_a"}, {"id": "doc_b"}, {"id": "doc_c"}]
    cache.set_context(docs1, contexts)

    # Retrieve with different order
    cached = cache.get_context(docs2)
    assert cached is not None, "Context cache miss with different order"
    print("✓ Context cache order-independent")


def run_all_tests():
    """Run all smoke tests."""
    print("=== Running M2.1 Caching Smoke Tests ===\n")

    tests = [
        test_key_schema_stable,
        test_stampede_lock_works,
        test_ttl_applied,
        test_semantic_similarity_cache,
        test_safe_no_key_stubs,
        test_metrics_tracking,
        test_context_cache_key_order
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

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
