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
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

# Graceful imports
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

import config


class CacheMetrics:
    """Track cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.stampede_prevented = 0
        self.invalidations = 0
        self.errors = 0

    def record_hit(self, layer: str):
        self.hits += 1
        print(f"✓ Cache HIT [{layer}]")

    def record_miss(self, layer: str):
        self.misses += 1
        print(f"✗ Cache MISS [{layer}]")

    def record_stampede_prevented(self):
        self.stampede_prevented += 1

    def record_invalidation(self, keys_count: int = 1):
        self.invalidations += keys_count

    def record_error(self, error: str):
        self.errors += 1
        print(f"⚠️ Cache error: {error}")

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Hits: {self.hits}, Misses: {self.misses}, "
            f"Hit Rate: {self.get_hit_rate():.1f}%, "
            f"Stampede prevented: {self.stampede_prevented}, "
            f"Invalidations: {self.invalidations}, Errors: {self.errors}"
        )


class CacheKeyGenerator:
    """Generate consistent cache keys with proper namespacing."""

    @staticmethod
    def exact_key(query: str) -> str:
        """Generate key for exact query match."""
        hash_val = hashlib.sha256(query.encode()).hexdigest()
        return f"{config.PREFIX_EXACT}{hash_val}"

    @staticmethod
    def semantic_key(query: str, bucket: int = 0) -> str:
        """Generate key for semantic similarity bucket."""
        hash_val = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{config.PREFIX_SEMANTIC}bucket_{bucket}:{hash_val}"

    @staticmethod
    def embedding_key(text: str) -> str:
        """Generate key for embedding cache."""
        hash_val = hashlib.sha256(text.encode()).hexdigest()
        return f"{config.PREFIX_EMBEDDING}{hash_val}"

    @staticmethod
    def context_key(doc_ids: List[str]) -> str:
        """Generate key for retrieved context cache."""
        combined = "|".join(sorted(doc_ids))
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        return f"{config.PREFIX_CONTEXT}{hash_val}"

    @staticmethod
    def lock_key(key: str) -> str:
        """Generate lock key for stampede protection."""
        return f"lock:{key}"


class StampedeLock:
    """Per-key lock for cache stampede protection."""

    def __init__(self, redis_client, key: str, timeout: int = 10):
        self.redis = redis_client
        self.key = CacheKeyGenerator.lock_key(key)
        self.timeout = timeout
        self.lock_value = f"{threading.get_ident()}_{time.time()}"

    def __enter__(self):
        # Try to acquire lock
        max_wait = 5  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            if self.redis.set(self.key, self.lock_value, nx=True, ex=self.timeout):
                return True
            time.sleep(0.1)
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release lock only if we own it
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
            pass


class MultiLayerCache:
    """Multi-layer caching system with TTL, invalidation, and stampede protection."""

    def __init__(self, redis_client=None, openai_client=None):
        self.redis = redis_client
        self.openai = openai_client
        self.metrics = CacheMetrics()
        self.keygen = CacheKeyGenerator()

    # ========== LAYER 1: EXACT QUERY CACHE ==========

    def get_exact(self, query: str) -> Optional[Dict[str, Any]]:
        """Check exact query cache."""
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
        """Store in exact query cache."""
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
        """Check semantic similarity cache using fuzzy matching."""
        if not self.redis or not config.ENABLE_SEMANTIC_CACHE or not fuzz:
            return None

        threshold = threshold or config.SEMANTIC_THRESHOLD
        try:
            # Scan semantic cache keys
            pattern = f"{config.PREFIX_SEMANTIC}*"
            for key in self.redis.scan_iter(match=pattern, count=100):
                cached_data = self.redis.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    cached_query = data.get("query", "")
                    # Use rapidfuzz for similarity
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
        """Store in semantic cache."""
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
        """Retrieve cached embedding."""
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
        """Store embedding in cache."""
        if not self.redis or not config.ENABLE_EMBEDDING_CACHE:
            return

        try:
            key = self.keygen.embedding_key(text)
            ttl = ttl or config.TTL_EMBEDDING_CACHE
            self.redis.setex(key, ttl, json.dumps(embedding))
        except Exception as e:
            self.metrics.record_error(f"embedding set: {e}")

    def compute_or_get_embedding(self, text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
        """Get embedding from cache or compute via OpenAI."""
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
                # Double-check cache
                cached = self.get_embedding(text)
                if cached:
                    self.metrics.record_stampede_prevented()
                    return cached

                # Compute
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
        """Retrieve cached document contexts."""
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
        """Store retrieved contexts in cache."""
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
        """Invalidate all keys matching a prefix."""
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
        """Invalidate caches for a specific query."""
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
        """Invalidate entries older than max_age_seconds."""
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
        """Clear all cache layers (use with caution)."""
        if not self.redis:
            return

        try:
            for prefix in [config.PREFIX_EXACT, config.PREFIX_SEMANTIC,
                          config.PREFIX_EMBEDDING, config.PREFIX_CONTEXT]:
                self.invalidate_by_prefix(prefix)
        except Exception as e:
            self.metrics.record_error(f"flush all: {e}")
