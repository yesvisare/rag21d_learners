"""
M2.1 - Multi-Layer Caching System for RAG Cost Reduction

Implements:
- Exact query cache (hash-based)
- Semantic query cache (BM25 + MinHash)
- Embedding cache (vector storage)
- Retrieved-context cache (document snippets)
- TTL management
- Cache invalidation hooks
- Cache stampede protection (per-key locks)
- Metrics and logging
"""
import hashlib
import json
import time
import threading
from typing import Optional, List, Dict, Any
import numpy as np

# Graceful imports
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

from . import config


class CacheMetrics:
    """
    Track cache performance metrics.

    Attributes:
        hits: Number of cache hits across all layers.
        misses: Number of cache misses across all layers.
        stampede_prevented: Count of concurrent requests protected by locks.
        invalidations: Number of cache keys explicitly invalidated.
        errors: Number of errors encountered during cache operations.
    """

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.stampede_prevented = 0
        self.invalidations = 0
        self.errors = 0

    def record_hit(self, layer: str):
        """
        Record a cache hit.

        Args:
            layer: Name of the cache layer (exact, semantic, embedding, context).
        """
        self.hits += 1
        print(f"✓ Cache HIT [{layer}]")

    def record_miss(self, layer: str):
        """
        Record a cache miss.

        Args:
            layer: Name of the cache layer.
        """
        self.misses += 1
        print(f"✗ Cache MISS [{layer}]")

    def record_stampede_prevented(self):
        """Record that a stampede was prevented by lock acquisition."""
        self.stampede_prevented += 1

    def record_invalidation(self, keys_count: int = 1):
        """
        Record cache invalidation.

        Args:
            keys_count: Number of keys invalidated.
        """
        self.invalidations += keys_count

    def record_error(self, error: str):
        """
        Record a cache error.

        Args:
            error: Description of the error encountered.
        """
        self.errors += 1
        print(f"⚠️ Cache error: {error}")

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate percentage.

        Returns:
            Hit rate as percentage (0-100), or 0.0 if no attempts.
        """
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def summary(self) -> str:
        """
        Get formatted metrics summary.

        Returns:
            Human-readable metrics string.
        """
        return (
            f"Hits: {self.hits}, Misses: {self.misses}, "
            f"Hit Rate: {self.get_hit_rate():.1f}%, "
            f"Stampede prevented: {self.stampede_prevented}, "
            f"Invalidations: {self.invalidations}, Errors: {self.errors}"
        )


class CacheKeyGenerator:
    """
    Generate consistent cache keys with proper namespacing.

    Rationale:
        Uses full SHA-256 hashes (no truncation) to eliminate collision risk.
        Prefixes ensure different cache layers don't interfere.
    """

    @staticmethod
    def exact_key(query: str) -> str:
        """
        Generate key for exact query match.

        Args:
            query: User query string.

        Returns:
            Prefixed SHA-256 hash key.
        """
        hash_val = hashlib.sha256(query.encode()).hexdigest()
        return f"{config.PREFIX_EXACT}{hash_val}"

    @staticmethod
    def semantic_key(query: str, bucket: int = 0) -> str:
        """
        Generate key for semantic similarity bucket.

        Args:
            query: User query string.
            bucket: Optional bucket index for partitioning.

        Returns:
            Prefixed hash key with bucket identifier.

        Rationale:
            Truncates to 16 chars for semantic keys since exact collision
            avoidance less critical here (fuzzy matching already tolerates variance).
        """
        hash_val = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{config.PREFIX_SEMANTIC}bucket_{bucket}:{hash_val}"

    @staticmethod
    def embedding_key(text: str) -> str:
        """
        Generate key for embedding cache.

        Args:
            text: Text to be embedded.

        Returns:
            Prefixed SHA-256 hash key.
        """
        hash_val = hashlib.sha256(text.encode()).hexdigest()
        return f"{config.PREFIX_EMBEDDING}{hash_val}"

    @staticmethod
    def context_key(doc_ids: List[str]) -> str:
        """
        Generate key for retrieved context cache.

        Args:
            doc_ids: List of document IDs.

        Returns:
            Prefixed SHA-256 hash key.

        Rationale:
            Sorts doc_ids to ensure same documents in different order produce
            identical cache keys (order-independent caching).
        """
        combined = "|".join(sorted(doc_ids))
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        return f"{config.PREFIX_CONTEXT}{hash_val}"

    @staticmethod
    def lock_key(key: str) -> str:
        """
        Generate lock key for stampede protection.

        Args:
            key: Original cache key.

        Returns:
            Lock-prefixed key.
        """
        return f"lock:{key}"


class StampedeLock:
    """
    Per-key lock for cache stampede protection.

    Prevents multiple concurrent requests from computing the same expensive
    operation (e.g., embedding API call) when cache is cold.

    Rationale:
        Uses Redis SET NX (set if not exists) for distributed locking.
        Implements wait-with-timeout to handle race conditions gracefully.
        Lua script ensures atomic lock release (delete only if owner).
    """

    def __init__(self, redis_client, key: str, timeout: int = 10):
        """
        Initialize stampede lock.

        Args:
            redis_client: Redis client instance.
            key: Cache key to protect.
            timeout: Lock auto-expiration in seconds (prevents deadlocks).
        """
        self.redis = redis_client
        self.key = CacheKeyGenerator.lock_key(key)
        self.timeout = timeout
        self.lock_value = f"{threading.get_ident()}_{time.time()}"

    def __enter__(self):
        """
        Acquire lock with exponential backoff retry.

        Returns:
            True if lock acquired, False if timeout.

        Rationale:
            Max wait of 5s prevents indefinite blocking. Sleep 0.1s between
            retries balances responsiveness vs. Redis load.
        """
        max_wait = 5  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            if self.redis.set(self.key, self.lock_value, nx=True, ex=self.timeout):
                return True
            time.sleep(0.1)  # Wait before retry
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Release lock only if current thread owns it.

        Rationale:
            Lua script ensures atomicity: only delete if value matches.
            Prevents one thread from releasing another's lock.
        """
        try:
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            self.redis.eval(script, 1, self.key, self.lock_value)
        except Exception:
            pass  # Ignore release errors (lock may have expired)


class MultiLayerCache:
    """
    Multi-layer caching system with TTL, invalidation, and stampede protection.

    Layers (checked in order):
        1. Exact query cache (SHA-256 hash match)
        2. Semantic query cache (fuzzy similarity)
        3. Embedding cache (vector storage)
        4. Retrieved-context cache (document snippets)
    """

    def __init__(self, redis_client=None, openai_client=None):
        """
        Initialize multi-layer cache.

        Args:
            redis_client: Redis client instance (None for graceful degradation).
            openai_client: OpenAI client instance (None for demo mode).
        """
        self.redis = redis_client
        self.openai = openai_client
        self.metrics = CacheMetrics()
        self.keygen = CacheKeyGenerator()

    # ========== LAYER 1: EXACT QUERY CACHE ==========

    def get_exact(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check exact query cache.

        Args:
            query: User query string.

        Returns:
            Cached response dict or None if miss.

        Raises:
            None: Errors are caught and recorded in metrics.
        """
        if not self.redis or not config.ENABLE_EXACT_CACHE:
            return None

        try:
            key = self.keygen.exact_key(query)
            cached = self.redis.get(key)
            if cached:
                self.metrics.record_hit("exact")
                return json.loads(cached)
            self.metrics.record_miss("exact")
            return None
        except Exception as e:
            self.metrics.record_error(f"exact get: {e}")
            return None

    def set_exact(self, query: str, response: Dict[str, Any], ttl: int = None):
        """
        Store in exact query cache.

        Args:
            query: User query string.
            response: Response dict to cache.
            ttl: Time-to-live in seconds (uses config default if None).
        """
        if not self.redis or not config.ENABLE_EXACT_CACHE:
            return

        try:
            key = self.keygen.exact_key(query)
            ttl = ttl or config.TTL_EXACT_CACHE
            self.redis.setex(key, ttl, json.dumps(response))
        except Exception as e:
            self.metrics.record_error(f"exact set: {e}")

    # ========== LAYER 2: SEMANTIC QUERY CACHE ==========

    def get_semantic(self, query: str, threshold: float = None) -> Optional[Dict[str, Any]]:
        """
        Check semantic similarity cache using fuzzy matching.

        Args:
            query: User query string.
            threshold: Similarity threshold 0-1 (uses config default if None).

        Returns:
            Cached response dict or None if no match above threshold.

        Rationale:
            Scans semantic cache with SCAN (cursor-based iteration) to avoid
            blocking Redis on large keyspaces. Trade-off: O(N) scan vs. exact
            hash lookup, but enables fuzzy matching for paraphrased queries.
        """
        if not self.redis or not config.ENABLE_SEMANTIC_CACHE or not fuzz:
            return None

        threshold = threshold or config.SEMANTIC_THRESHOLD
        try:
            # Scan semantic cache keys (cursor-based iteration for large keyspaces)
            pattern = f"{config.PREFIX_SEMANTIC}*"
            for key in self.redis.scan_iter(match=pattern, count=100):
                cached_data = self.redis.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    cached_query = data.get("query", "")
                    # Use rapidfuzz for similarity (Levenshtein-based ratio)
                    similarity = fuzz.ratio(query.lower(), cached_query.lower()) / 100.0
                    if similarity >= threshold:
                        self.metrics.record_hit("semantic")
                        return data.get("response")

            self.metrics.record_miss("semantic")
            return None
        except Exception as e:
            self.metrics.record_error(f"semantic get: {e}")
            return None

    def set_semantic(self, query: str, response: Dict[str, Any], ttl: int = None):
        """
        Store in semantic cache.

        Args:
            query: User query string.
            response: Response dict to cache.
            ttl: Time-to-live in seconds.
        """
        if not self.redis or not config.ENABLE_SEMANTIC_CACHE:
            return

        try:
            key = self.keygen.semantic_key(query)
            ttl = ttl or config.TTL_SEMANTIC_CACHE
            data = {"query": query, "response": response, "timestamp": time.time()}
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as e:
            self.metrics.record_error(f"semantic set: {e}")

    # ========== LAYER 3: EMBEDDING CACHE ==========

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding.

        Args:
            text: Text to embed.

        Returns:
            Cached embedding vector or None if miss.
        """
        if not self.redis or not config.ENABLE_EMBEDDING_CACHE:
            return None

        try:
            key = self.keygen.embedding_key(text)
            cached = self.redis.get(key)
            if cached:
                self.metrics.record_hit("embedding")
                return json.loads(cached)
            self.metrics.record_miss("embedding")
            return None
        except Exception as e:
            self.metrics.record_error(f"embedding get: {e}")
            return None

    def set_embedding(self, text: str, embedding: List[float], ttl: int = None):
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded.
            embedding: Vector embedding (list of floats).
            ttl: Time-to-live in seconds.
        """
        if not self.redis or not config.ENABLE_EMBEDDING_CACHE:
            return

        try:
            key = self.keygen.embedding_key(text)
            ttl = ttl or config.TTL_EMBEDDING_CACHE
            self.redis.setex(key, ttl, json.dumps(embedding))
        except Exception as e:
            self.metrics.record_error(f"embedding set: {e}")

    def compute_or_get_embedding(self, text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
        """
        Get embedding from cache or compute via OpenAI.

        Args:
            text: Text to embed.
            model: OpenAI embedding model name.

        Returns:
            Embedding vector or None if unavailable.

        Rationale:
            Double-check pattern inside lock prevents race condition where
            multiple threads pass initial cache check simultaneously. First
            thread computes, others find cached result on second check.
        """
        # Check cache first
        cached = self.get_embedding(text)
        if cached:
            return cached

        # Compute if OpenAI available
        if not self.openai:
            print("⚠️ Skipping embedding computation (no OpenAI client)")
            return None

        try:
            key = self.keygen.embedding_key(text)
            # Stampede protection
            with StampedeLock(self.redis, key):
                # Double-check cache (another thread may have computed while we waited)
                cached = self.get_embedding(text)
                if cached:
                    self.metrics.record_stampede_prevented()
                    return cached

                # Compute embedding via API
                response = self.openai.embeddings.create(input=text, model=model)
                embedding = response.data[0].embedding

                # Cache it
                self.set_embedding(text, embedding)
                return embedding
        except Exception as e:
            self.metrics.record_error(f"embedding compute: {e}")
            return None

    # ========== LAYER 4: RETRIEVED-CONTEXT CACHE ==========

    def get_context(self, doc_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached document contexts.

        Args:
            doc_ids: List of document IDs.

        Returns:
            Cached contexts or None if miss.
        """
        if not self.redis or not config.ENABLE_CONTEXT_CACHE:
            return None

        try:
            key = self.keygen.context_key(doc_ids)
            cached = self.redis.get(key)
            if cached:
                self.metrics.record_hit("context")
                return json.loads(cached)
            self.metrics.record_miss("context")
            return None
        except Exception as e:
            self.metrics.record_error(f"context get: {e}")
            return None

    def set_context(self, doc_ids: List[str], contexts: List[Dict[str, Any]], ttl: int = None):
        """
        Store retrieved contexts in cache.

        Args:
            doc_ids: List of document IDs.
            contexts: Document context dicts.
            ttl: Time-to-live in seconds.
        """
        if not self.redis or not config.ENABLE_CONTEXT_CACHE:
            return

        try:
            key = self.keygen.context_key(doc_ids)
            ttl = ttl or config.TTL_CONTEXT_CACHE
            self.redis.setex(key, ttl, json.dumps(contexts))
        except Exception as e:
            self.metrics.record_error(f"context set: {e}")

    # ========== INVALIDATION ==========

    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all keys matching a prefix.

        Args:
            prefix: Cache key prefix (e.g., "exact:", "semantic:").

        Returns:
            Number of keys invalidated.

        Rationale:
            Uses SCAN instead of KEYS to avoid blocking Redis on large datasets.
            Count=1000 balances iteration speed vs. memory usage.
        """
        if not self.redis:
            return 0

        try:
            count = 0
            pattern = f"{prefix}*"
            for key in self.redis.scan_iter(match=pattern, count=1000):
                self.redis.delete(key)
                count += 1
            self.metrics.record_invalidation(count)
            print(f"🗑️ Invalidated {count} keys with prefix '{prefix}'")
            return count
        except Exception as e:
            self.metrics.record_error(f"invalidation: {e}")
            return 0

    def invalidate_query(self, query: str):
        """
        Invalidate caches for a specific query.

        Args:
            query: Query string to invalidate.
        """
        if not self.redis:
            return

        try:
            # Invalidate exact cache
            exact_key = self.keygen.exact_key(query)
            self.redis.delete(exact_key)

            # Invalidate semantic matches (scan and check)
            pattern = f"{config.PREFIX_SEMANTIC}*"
            for key in self.redis.scan_iter(match=pattern, count=100):
                cached_data = self.redis.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    if data.get("query") == query:
                        self.redis.delete(key)

            self.metrics.record_invalidation(1)
        except Exception as e:
            self.metrics.record_error(f"query invalidation: {e}")

    def invalidate_stale(self, max_age_seconds: int = 3600):
        """
        Invalidate entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds before invalidation.

        Rationale:
            Only semantic cache stores timestamps. Other layers rely on TTL
            for expiration. This method enables aggressive freshness policies
            beyond default TTLs.
        """
        if not self.redis:
            return

        try:
            count = 0
            current_time = time.time()
            pattern = f"{config.PREFIX_SEMANTIC}*"

            for key in self.redis.scan_iter(match=pattern, count=100):
                cached_data = self.redis.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    timestamp = data.get("timestamp", 0)
                    if current_time - timestamp > max_age_seconds:
                        self.redis.delete(key)
                        count += 1

            self.metrics.record_invalidation(count)
            print(f"🗑️ Invalidated {count} stale entries")
        except Exception as e:
            self.metrics.record_error(f"stale invalidation: {e}")

    def flush_all(self):
        """
        Clear all cache layers (use with caution).

        Raises:
            None: Errors are caught and recorded in metrics.
        """
        if not self.redis:
            return

        try:
            for prefix in [config.PREFIX_EXACT, config.PREFIX_SEMANTIC,
                          config.PREFIX_EMBEDDING, config.PREFIX_CONTEXT]:
                self.invalidate_by_prefix(prefix)
        except Exception as e:
            self.metrics.record_error(f"flush all: {e}")


# CLI support for direct module execution
if __name__ == "__main__":
    print("M2.1 Multi-Layer Caching System")
    print("================================")
    print("\nInitializing clients...")

    redis_client = config.get_redis()
    openai_client = config.get_openai()

    cache = MultiLayerCache(redis_client, openai_client)

    if redis_client:
        print("✓ Redis connected")
    else:
        print("✗ Redis unavailable (demo mode)")

    if openai_client:
        print("✓ OpenAI configured")
    else:
        print("✗ OpenAI unavailable (demo mode)")

    print(f"\nCache metrics: {cache.metrics.summary()}")
    print("\nUse as library: from src.m2_1_caching.module import MultiLayerCache")
