"""
M1.2 — Pinecone Data Model & Advanced Indexing
Implements hybrid (dense + sparse) vector search using Pinecone + OpenAI.

Features:
- Hybrid search combining semantic (dense) and keyword (sparse) retrieval
- Namespace-based multi-tenant isolation
- Alpha parameter tuning for query-specific blending
- Production error handling and validation
- GPT-4 reranking for result quality
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pinecone_text.sparse import BM25Encoder
from config import get_clients, OPENAI_MODEL, PINECONE_INDEX, DEFAULT_NAMESPACE, SCORE_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global BM25 encoder instance
_bm25_encoder = None


def build_index(dimension: int = 1536, metric: str = "dotproduct"):
    """
    Create or connect to a Pinecone hybrid index.

    Args:
        dimension: Embedding dimension (default 1536 for text-embedding-3-small)
        metric: Distance metric (must be 'dotproduct' for hybrid search)

    Returns:
        Pinecone Index object or None if clients unavailable
    """
    openai_client, pinecone_client = get_clients()

    if not pinecone_client:
        logger.warning("⚠️ Skipping index creation (no keys found)")
        logger.info(f"Would create index: name={PINECONE_INDEX}, dimension={dimension}, metric={metric}")
        return None

    # Check metric
    if metric != "dotproduct":
        logger.error(f"❌ FAILURE CHECK #2: Metric mismatch! Hybrid search requires 'dotproduct', got '{metric}'")
        raise ValueError(f"Hybrid search requires dotproduct metric, got {metric}")

    try:
        # Check if index exists
        existing_indexes = [idx.name for idx in pinecone_client.list_indexes()]

        if PINECONE_INDEX not in existing_indexes:
            logger.info(f"Creating new index: {PINECONE_INDEX}")
            pinecone_client.create_index(
                name=PINECONE_INDEX,
                dimension=dimension,
                metric=metric,
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            logger.info(f"✓ Index {PINECONE_INDEX} created successfully")
        else:
            logger.info(f"✓ Connected to existing index: {PINECONE_INDEX}")

        index = pinecone_client.Index(PINECONE_INDEX)
        logger.info(f"Index stats: {index.describe_index_stats()}")
        return index

    except Exception as e:
        logger.error(f"Error building index: {e}")
        return None


def embed_dense_openai(text: str, max_retries: int = 3) -> Optional[List[float]]:
    """
    Generate dense embeddings using OpenAI with exponential backoff.

    Args:
        text: Input text to embed
        max_retries: Maximum retry attempts for rate limiting

    Returns:
        List of floats (1536-dim vector) or None if failed
    """
    openai_client, _ = get_clients()

    if not openai_client:
        logger.warning("⚠️ Skipping embedding (no OpenAI key)")
        return None

    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                input=text,
                model=OPENAI_MODEL
            )
            embedding = response.data[0].embedding
            logger.info(f"✓ Dense embedding generated: dim={len(embedding)}")
            return embedding

        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to generate embedding after {max_retries} attempts")
                return None


def embed_sparse_bm25(texts: List[str] = None, query: str = None) -> Dict[str, Any]:
    """
    Generate sparse BM25 embeddings for corpus or query.

    Args:
        texts: List of documents to fit BM25 (corpus mode)
        query: Single query string to encode (query mode)

    Returns:
        Dict with 'indices' and 'values' keys for sparse vector

    Raises:
        ValueError: If BM25 not fitted before querying (FAILURE CHECK #1)
    """
    global _bm25_encoder

    # Corpus mode: fit BM25
    if texts:
        logger.info(f"Fitting BM25 on {len(texts)} documents...")
        _bm25_encoder = BM25Encoder()
        _bm25_encoder.fit(texts)
        logger.info("✓ BM25 encoder fitted")
        return {"status": "fitted"}

    # Query mode: encode
    if query:
        if _bm25_encoder is None:
            logger.error("❌ FAILURE CHECK #1: BM25 not fitted! Call embed_sparse_bm25(texts=...) first")
            raise ValueError("BM25 encoder must be fitted before encoding queries")

        sparse_vec = _bm25_encoder.encode_queries(query)
        # Convert to dict format expected by Pinecone
        result = {
            "indices": sparse_vec["indices"].tolist(),
            "values": sparse_vec["values"].tolist()
        }
        logger.info(f"✓ Sparse encoding: {len(result['indices'])} non-zero terms")
        return result

    raise ValueError("Must provide either 'texts' (fit mode) or 'query' (encode mode)")


def validate_metadata_size(metadata: Dict[str, Any], max_size: int = 40960) -> bool:
    """
    Validate metadata doesn't exceed Pinecone's 40KB limit.

    Args:
        metadata: Metadata dictionary
        max_size: Maximum allowed size in bytes (default 40KB)

    Returns:
        True if valid, raises ValueError otherwise (FAILURE CHECK #4)
    """
    import sys
    size = sys.getsizeof(str(metadata))

    if size > max_size:
        logger.error(f"❌ FAILURE CHECK #4: Metadata size {size} bytes exceeds {max_size} byte limit")
        raise ValueError(f"Metadata too large: {size} bytes (limit: {max_size})")

    logger.info(f"✓ Metadata size OK: {size} bytes")
    return True


def upsert_hybrid_vectors(
    docs: List[str],
    namespace: str = DEFAULT_NAMESPACE,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Upsert documents with both dense and sparse vectors.

    Args:
        docs: List of document texts
        namespace: Pinecone namespace for isolation
        batch_size: Vectors per batch

    Returns:
        Dict with success/failure counts and failed IDs (FAILURE CHECK #5)
    """
    openai_client, pinecone_client = get_clients()

    if not openai_client or not pinecone_client:
        logger.warning("⚠️ Skipping upsert (no keys found)")
        logger.info(f"Would upsert {len(docs)} docs to namespace '{namespace}'")
        return {"skipped": len(docs), "reason": "no_api_keys"}

    try:
        index = pinecone_client.Index(PINECONE_INDEX)
    except Exception as e:
        logger.error(f"Failed to connect to index: {e}")
        return {"error": str(e)}

    # Fit BM25 on corpus
    embed_sparse_bm25(texts=docs)

    vectors_to_upsert = []
    failed_ids = []
    success_count = 0

    for idx, doc in enumerate(docs):
        try:
            # Generate dense embedding
            dense_vec = embed_dense_openai(doc)
            if not dense_vec:
                failed_ids.append(f"doc_{idx}")
                continue

            # Generate sparse embedding
            sparse_vec = embed_sparse_bm25(query=doc)

            # Validate metadata
            metadata = {"text": doc[:500], "source": "example_data"}  # Truncate for size
            validate_metadata_size(metadata)

            # Prepare vector
            vector = {
                "id": f"doc_{idx}",
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": metadata
            }
            vectors_to_upsert.append(vector)

        except Exception as e:
            logger.warning(f"Failed to process doc {idx}: {e}")
            failed_ids.append(f"doc_{idx}")
            continue

    # Batch upsert with failure handling
    try:
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors to namespace '{namespace}'...")
        index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        success_count = len(vectors_to_upsert)
        logger.info(f"✓ Upserted {success_count} vectors successfully")

    except Exception as e:
        logger.error(f"❌ FAILURE CHECK #5: Batch upsert failed: {e}")
        return {
            "success": 0,
            "failed": len(vectors_to_upsert),
            "failed_ids": [v["id"] for v in vectors_to_upsert],
            "error": str(e)
        }

    result = {
        "success": success_count,
        "failed": len(failed_ids),
        "failed_ids": failed_ids,
        "namespace": namespace
    }

    if failed_ids:
        logger.warning(f"⚠️ Partial success: {len(failed_ids)} failures")

    return result


def safe_namespace_query(
    index,
    namespace: str,
    vector: Dict[str, Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Query with namespace validation to prevent empty results.

    Args:
        index: Pinecone index object
        namespace: Target namespace
        vector: Query vector (dense + sparse)
        top_k: Number of results

    Returns:
        List of matches (FAILURE CHECK #3: validates namespace exists)
    """
    if not index:
        logger.warning("⚠️ No index available")
        return []

    try:
        # Check namespace exists
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if namespace not in namespaces:
            logger.error(f"❌ FAILURE CHECK #3: Namespace '{namespace}' not found!")
            logger.info(f"Available namespaces: {list(namespaces.keys())}")
            return []

        # Execute query
        results = index.query(
            vector=vector.get("dense"),
            sparse_vector=vector.get("sparse"),
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )

        logger.info(f"✓ Retrieved {len(results.matches)} matches from namespace '{namespace}'")
        return results.matches

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []


def smart_alpha_selector(query: str) -> float:
    """
    Heuristic alpha selection based on query characteristics.

    Args:
        query: User query text

    Returns:
        Alpha value (0.0-1.0) where:
            - 0.2-0.3: Keyword-heavy (product names, codes)
            - 0.5: Balanced (default)
            - 0.7-0.8: Semantic-heavy (natural language questions)

    Reference:
        - Lower alpha = more sparse (keyword) weight
        - Higher alpha = more dense (semantic) weight
    """
    query_lower = query.lower()

    # Keyword indicators: short, has numbers/codes, specific terms
    keyword_signals = [
        len(query.split()) <= 3,  # Short query
        any(char.isdigit() for char in query),  # Contains numbers
        any(term in query_lower for term in ["code", "id", "number", "sku", "part"])
    ]

    # Semantic indicators: questions, long queries, conceptual
    semantic_signals = [
        len(query.split()) >= 8,  # Long query
        any(q in query_lower for q in ["how", "what", "why", "explain", "describe"]),
        any(term in query_lower for term in ["concept", "understand", "meaning", "difference"])
    ]

    keyword_score = sum(keyword_signals)
    semantic_score = sum(semantic_signals)

    if keyword_score >= 2:
        alpha = 0.3
        logger.info(f"Alpha selector: keyword-heavy query → α={alpha}")
    elif semantic_score >= 2:
        alpha = 0.7
        logger.info(f"Alpha selector: semantic-heavy query → α={alpha}")
    else:
        alpha = 0.5
        logger.info(f"Alpha selector: balanced query → α={alpha}")

    return alpha


def hybrid_query(
    query: str,
    alpha: float = 0.5,
    top_k: int = 5,
    namespace: str = DEFAULT_NAMESPACE
) -> List[Dict[str, Any]]:
    """
    Execute hybrid search with alpha-weighted dense + sparse retrieval.

    Args:
        query: Search query text
        alpha: Blending weight (0=pure sparse, 1=pure dense)
        top_k: Number of results to return
        namespace: Target namespace

    Returns:
        List of ranked results with scores and metadata
    """
    start_time = time.time()

    openai_client, pinecone_client = get_clients()

    if not openai_client or not pinecone_client:
        logger.warning("⚠️ Skipping query (no keys found)")
        logger.info(f"Would query: '{query}' with α={alpha}, top_k={top_k}, namespace='{namespace}'")
        return []

    try:
        index = pinecone_client.Index(PINECONE_INDEX)

        # Generate dense query vector
        dense_vec = embed_dense_openai(query)
        if not dense_vec:
            return []

        # Generate sparse query vector
        sparse_vec = embed_sparse_bm25(query=query)

        # Scale vectors by alpha
        dense_scaled = [v * alpha for v in dense_vec]
        sparse_scaled = {
            "indices": sparse_vec["indices"],
            "values": [v * (1 - alpha) for v in sparse_vec["values"]]
        }

        # Query with namespace validation
        vector_payload = {"dense": dense_scaled, "sparse": sparse_scaled}
        matches = safe_namespace_query(index, namespace, vector_payload, top_k)

        # Format results
        results = []
        for match in matches:
            results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", "")[:80],  # Preview
                "metadata": match.metadata
            })

        elapsed = time.time() - start_time
        logger.info(f"✓ Hybrid query completed in {elapsed:.3f}s | α={alpha} | Results: {len(results)}")

        # Print results
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Alpha: {alpha} | Namespace: {namespace} | Time: {elapsed:.3f}s")
        print(f"{'='*60}")
        for i, res in enumerate(results[:3], 1):
            print(f"{i}. [{res['score']:.4f}] {res['text']}...")
        print(f"{'='*60}\n")

        return results

    except Exception as e:
        logger.error(f"Hybrid query failed: {e}")
        return []


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Rerank top results using GPT-4 for relevance scoring.

    Args:
        query: Original query
        results: Initial retrieval results
        top_n: Number of results to return after reranking

    Returns:
        Reranked results with GPT-4 relevance scores
    """
    openai_client, _ = get_clients()

    if not openai_client:
        logger.warning("⚠️ Skipping reranking (no OpenAI key)")
        return results[:top_n]

    if not results:
        return []

    try:
        # Prepare context for GPT-4
        context = "\n".join([
            f"{i+1}. {res.get('text', res.get('metadata', {}).get('text', ''))}"
            for i, res in enumerate(results[:10])
        ])

        prompt = f"""Rank these search results by relevance to the query.
Query: {query}

Results:
{context}

Return ONLY a comma-separated list of result numbers in order of relevance (e.g., "3,1,5").
"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )

        # Parse ranking
        ranking_str = response.choices[0].message.content.strip()
        ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(",") if x.strip().isdigit()]

        # Reorder results
        reranked = [results[i] for i in ranking_indices[:top_n] if i < len(results)]
        logger.info(f"✓ Reranked {len(reranked)} results using GPT-4")

        return reranked

    except Exception as e:
        logger.warning(f"Reranking failed, returning original order: {e}")
        return results[:top_n]


# Health check functions for testing
def check_bm25_fitted() -> bool:
    """Check if BM25 encoder is fitted (FAILURE CHECK #1)."""
    return _bm25_encoder is not None


def check_index_metric(expected_metric: str = "dotproduct") -> bool:
    """Check index uses correct metric (FAILURE CHECK #2)."""
    _, pinecone_client = get_clients()
    if not pinecone_client:
        return False

    try:
        index_info = pinecone_client.describe_index(PINECONE_INDEX)
        actual_metric = index_info.metric
        return actual_metric == expected_metric
    except:
        return False


if __name__ == "__main__":
    print("M1.2 Pinecone Advanced Indexing Module")
    print("Import this module in notebooks or scripts")
    print("\nKey functions:")
    print("  - build_index()")
    print("  - embed_dense_openai(text)")
    print("  - embed_sparse_bm25(texts|query)")
    print("  - upsert_hybrid_vectors(docs, namespace)")
    print("  - hybrid_query(query, alpha, top_k, namespace)")
    print("  - smart_alpha_selector(query)")
    print("  - rerank_results(query, results)")
