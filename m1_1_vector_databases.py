#!/usr/bin/env python3
"""
M1.1 Vector Databases - Production-Style Reference Implementation

This module demonstrates end-to-end vector search flow with:
- Embedding generation with exponential backoff for rate limiting
- Pinecone index creation with readiness polling
- Batch vector upsertion with rich metadata
- Semantic search with filtering and score thresholding
- Proper error handling and logging

Usage:
    Initialize index and upsert data:
        python m1_1_vector_databases.py --init

    Query the index:
        python m1_1_vector_databases.py --query "What is vector search?" --top_k 5 --threshold 0.7

    Query with namespace:
        python m1_1_vector_databases.py --query "machine learning" --namespace demo --top_k 3
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI, RateLimitError
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Import configuration
import config

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_example_texts(file_path: str = "example_data.txt") -> List[str]:
    """
    Load example text chunks from data file.

    Args:
        file_path: Path to the text file (one document per line)

    Returns:
        List of text chunks

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the file is empty
    """
    logger.info(f"Loading example texts from: {file_path}")

    data_file = Path(file_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please ensure example_data.txt exists in the current directory."
        )

    with open(data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        raise ValueError(f"Data file is empty: {file_path}")

    logger.info(f"✓ Loaded {len(texts)} text chunks")
    return texts


# ============================================================================
# EMBEDDING GENERATION WITH RETRY LOGIC
# ============================================================================

def embed_texts_openai(
    texts: List[str],
    client: Optional[OpenAI] = None,
    model: str = config.EMBEDDING_MODEL,
    max_retries: int = config.MAX_RETRIES
) -> List[List[float]]:
    """
    Generate embeddings for text chunks with exponential backoff for rate limiting.

    Args:
        texts: List of text chunks to embed
        client: OpenAI client instance (creates new one if None)
        model: Embedding model name
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding vectors (each vector is a list of floats)

    Raises:
        RateLimitError: If rate limit is exceeded after all retries
        ValueError: If texts list is empty
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")

    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    logger.info(f"Generating embeddings for {len(texts)} texts using {model}")

    embeddings = []

    for i, text in enumerate(tqdm(texts, desc="Embedding texts")):
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=text
                )
                embedding = response.data[0].embedding

                # Validate embedding dimension
                expected_dim = config.MODEL_DIMENSIONS.get(model)
                actual_dim = len(embedding)

                if expected_dim and actual_dim != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch! Model {model} produced {actual_dim} dimensions, "
                        f"but expected {expected_dim}. Check your model configuration."
                    )

                embeddings.append(embedding)
                break  # Success - exit retry loop

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise

                wait_time = config.INITIAL_RETRY_DELAY * (config.RETRY_BACKOFF_MULTIPLIER ** attempt)
                logger.warning(
                    f"Rate limit hit on text {i+1}/{len(texts)}. "
                    f"Waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                )
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Failed to embed text {i+1}: {str(e)}")
                raise

    logger.info(f"✓ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
    return embeddings


# ============================================================================
# COSINE SIMILARITY HELPER
# ============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the angle between vectors:
    - 1.0 = identical direction (highly similar)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (opposite meaning)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different dimensions or are zero-length
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec1)} != {len(vec2)}\n"
            f"Both vectors must have the same dimension for similarity calculation."
        )

    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    # Calculate dot product and norms
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate similarity for zero-length vectors")

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


# ============================================================================
# PINECONE INDEX CREATION WITH READINESS POLLING
# ============================================================================

def create_index_and_wait_pinecone(
    pc: Pinecone,
    index_name: str,
    dimension: int = config.EMBEDDING_DIM,
    metric: str = config.METRIC,
    timeout: int = config.INDEX_READY_TIMEOUT
) -> Any:
    """
    Create Pinecone index and wait until it's ready for operations.

    Args:
        pc: Pinecone client instance
        index_name: Name for the new index
        dimension: Vector dimension (must match embedding model)
        metric: Distance metric ("cosine", "euclidean", or "dotproduct")
        timeout: Maximum seconds to wait for index initialization

    Returns:
        Pinecone Index instance ready for operations

    Raises:
        TimeoutError: If index doesn't become ready within timeout
        ValueError: If dimension or metric is invalid
    """
    logger.info(f"Creating Pinecone index: {index_name}")
    logger.info(f"  Dimension: {dimension}")
    logger.info(f"  Metric: {metric}")
    logger.info(f"  Region: {config.PINECONE_REGION}")

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        logger.warning(f"Index '{index_name}' already exists. Using existing index.")
        index = pc.Index(index_name)

        # Verify dimension matches
        stats = index.describe_index_stats()
        logger.info(f"Existing index stats: {stats}")

        return index

    # Create new index
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region=config.PINECONE_REGION
            )
        )
        logger.info(f"Index creation initiated: {index_name}")

    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        raise

    # Wait for index to be ready
    logger.info("Waiting for index initialization...")
    start_time = time.time()

    while True:
        try:
            description = pc.describe_index(index_name)
            status = description.status.get('ready', False)

            if status:
                elapsed = time.time() - start_time
                logger.info(f"✓ Index ready after {elapsed:.1f} seconds")
                break

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Index not ready after {timeout} seconds. "
                    f"This is unusual - check Pinecone console for status."
                )

            logger.info("  Index still initializing...")
            time.sleep(config.INDEX_READY_CHECK_INTERVAL)

        except Exception as e:
            if "not found" in str(e).lower():
                logger.warning("  Index not yet visible in API...")
                time.sleep(config.INDEX_READY_CHECK_INTERVAL)
            else:
                raise

    # Return connected index
    index = pc.Index(index_name)
    logger.info(f"✓ Connected to index: {index_name}")

    return index


# ============================================================================
# VECTOR UPSERTION WITH BATCHING AND METADATA
# ============================================================================

def upsert_vectors(
    index: Any,
    vectors: List[Tuple[str, List[float], Dict[str, Any]]],
    namespace: str = config.DEFAULT_NAMESPACE,
    batch_size: int = config.BATCH_SIZE
) -> Dict[str, int]:
    """
    Upsert vectors to Pinecone index with batching and rich metadata.

    Args:
        index: Pinecone Index instance
        vectors: List of (id, vector, metadata) tuples
        namespace: Namespace for data organization
        batch_size: Number of vectors per batch (recommended: 100-200)

    Returns:
        Dictionary with upsert statistics

    Raises:
        ValueError: If vectors list is empty or contains invalid data
    """
    if not vectors:
        raise ValueError("Cannot upsert empty vector list")

    logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'")
    logger.info(f"Batch size: {batch_size}")

    total_upserted = 0

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        # Format batch for Pinecone
        formatted_batch = []
        for vec_id, vec_values, vec_metadata in batch:
            # Validate metadata includes required fields
            if 'text' not in vec_metadata:
                logger.warning(
                    f"Vector {vec_id} missing 'text' in metadata. "
                    f"This will cause issues when retrieving results!"
                )

            formatted_batch.append({
                "id": vec_id,
                "values": vec_values,
                "metadata": vec_metadata
            })

        # Upsert batch
        try:
            upsert_response = index.upsert(
                vectors=formatted_batch,
                namespace=namespace
            )
            upserted_count = upsert_response.get('upserted_count', len(formatted_batch))
            total_upserted += upserted_count

            logger.info(
                f"  Batch {i // batch_size + 1}: "
                f"Upserted {upserted_count}/{len(formatted_batch)} vectors"
            )

        except Exception as e:
            logger.error(f"Failed to upsert batch {i // batch_size + 1}: {str(e)}")
            raise

    # Verify upsert
    time.sleep(2)  # Brief wait for consistency
    stats = index.describe_index_stats()
    namespace_stats = stats.get('namespaces', {}).get(namespace, {})
    vector_count = namespace_stats.get('vector_count', 0)

    logger.info(f"✓ Upsert complete: {total_upserted} vectors")
    logger.info(f"  Namespace '{namespace}' now contains {vector_count} vectors")

    return {
        'upserted': total_upserted,
        'namespace': namespace,
        'total_in_namespace': vector_count
    }


# ============================================================================
# SEMANTIC SEARCH WITH FILTERING AND THRESHOLDING
# ============================================================================

def query_pinecone(
    index: Any,
    query_text: str,
    client: Optional[OpenAI] = None,
    top_k: int = config.DEFAULT_TOP_K,
    namespace: str = config.DEFAULT_NAMESPACE,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: float = config.SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Query Pinecone index with semantic search and score filtering.

    Args:
        index: Pinecone Index instance
        query_text: Natural language query
        client: OpenAI client (creates new one if None)
        top_k: Maximum number of results to return
        namespace: Namespace to query
        filters: Metadata filters (e.g., {"category": {"$eq": "technology"}})
        score_threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        List of matching documents with scores and metadata

    Raises:
        ValueError: If query_text is empty or score_threshold is invalid
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")

    if not 0 <= score_threshold <= 1:
        raise ValueError(
            f"Score threshold must be between 0 and 1, got {score_threshold}"
        )

    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    logger.info(f"Querying: '{query_text}'")
    logger.info(f"  Namespace: {namespace}")
    logger.info(f"  Top-k: {top_k}")
    logger.info(f"  Score threshold: {score_threshold}")
    if filters:
        logger.info(f"  Filters: {filters}")

    # Generate query embedding
    try:
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=query_text
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {str(e)}")
        raise

    # Query Pinecone
    try:
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace
        }

        if filters:
            query_params["filter"] = filters

        results = index.query(**query_params)

    except Exception as e:
        logger.error(f"Pinecone query failed: {str(e)}")
        if "not ready" in str(e).lower():
            logger.error(
                "Index may not be ready yet. Wait for initialization to complete."
            )
        raise

    # Process and filter results
    matches = results.get('matches', [])

    if not matches:
        logger.warning("No results found for query")
        return []

    # Apply score threshold
    filtered_matches = []
    for match in matches:
        score = match.get('score', 0)

        if score >= score_threshold:
            filtered_matches.append({
                'id': match.get('id'),
                'score': score,
                'text': match.get('metadata', {}).get('text', '[No text in metadata]'),
                'metadata': match.get('metadata', {})
            })
        else:
            logger.debug(
                f"Filtered out result with score {score:.3f} "
                f"(below threshold {score_threshold})"
            )

    logger.info(f"✓ Found {len(filtered_matches)}/{len(matches)} results above threshold")

    if not filtered_matches:
        logger.warning(
            f"All results below threshold {score_threshold}. "
            f"Consider lowering threshold or rephrasing query. "
            f"Highest score was: {max(m.get('score', 0) for m in matches):.3f}"
        )

    return filtered_matches


# ============================================================================
# CLI COMMANDS
# ============================================================================

def command_init():
    """Initialize index and upsert example data."""
    logger.info("=" * 70)
    logger.info("INITIALIZING VECTOR DATABASE")
    logger.info("=" * 70)

    try:
        # Get clients
        openai_client, pinecone_client = config.get_clients()

        # Load data
        texts = load_example_texts()

        # Generate embeddings
        embeddings = embed_texts_openai(texts, client=openai_client)

        # Create index
        index = create_index_and_wait_pinecone(
            pinecone_client,
            config.INDEX_NAME
        )

        # Prepare vectors with rich metadata
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append((
                f"doc_{i}",  # ID
                embedding,   # Vector
                {           # Metadata
                    "text": text,
                    "source": "example_data.txt",
                    "chunk_id": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "length": len(text)
                }
            ))

        # Upsert vectors
        stats = upsert_vectors(index, vectors)

        logger.info("=" * 70)
        logger.info("✓ INITIALIZATION COMPLETE")
        logger.info(f"  Index: {config.INDEX_NAME}")
        logger.info(f"  Namespace: {config.DEFAULT_NAMESPACE}")
        logger.info(f"  Vectors: {stats['upserted']}")
        logger.info("=" * 70)
        logger.info("\nNext step: Run a query!")
        logger.info(f'  python {__file__} --query "What is vector search?"')

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        logger.info("\nPlease check your .env file and API keys.")
        return 1

    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        return 1

    return 0


def command_query(
    query_text: str,
    top_k: int = config.DEFAULT_TOP_K,
    namespace: str = config.DEFAULT_NAMESPACE,
    threshold: float = config.SCORE_THRESHOLD
):
    """Execute semantic search query."""
    logger.info("=" * 70)
    logger.info("QUERYING VECTOR DATABASE")
    logger.info("=" * 70)

    try:
        # Get clients
        openai_client, pinecone_client = config.get_clients()

        # Connect to index
        index = pinecone_client.Index(config.INDEX_NAME)

        # Query
        results = query_pinecone(
            index,
            query_text,
            client=openai_client,
            top_k=top_k,
            namespace=namespace,
            score_threshold=threshold
        )

        # Display results
        logger.info("=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)

        if not results:
            print("\nNo results found above threshold.")
            print(f"Try lowering --threshold (current: {threshold})")
            return 0

        print(f"\nQuery: '{query_text}'\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text']}")
            print(f"   Source: {result['metadata'].get('source', 'N/A')}")
            print(f"   Chunk: {result['metadata'].get('chunk_id', 'N/A')}")
            print()

        logger.info("=" * 70)

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        if "not found" in str(e).lower():
            logger.error(f"\nIndex '{config.INDEX_NAME}' not found.")
            logger.info("Run initialization first:")
            logger.info(f"  python {__file__} --init")
        return 1

    return 0


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="M1.1 Vector Databases - Production Reference Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize index and upsert data:
    %(prog)s --init

  Query the index:
    %(prog)s --query "What is vector search?"

  Query with custom parameters:
    %(prog)s --query "machine learning" --top_k 3 --threshold 0.8

  Query specific namespace:
    %(prog)s --query "climate change" --namespace demo --top_k 5
        """
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize index and upsert example data'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Natural language query text'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f'Number of results to return (default: {config.DEFAULT_TOP_K})'
    )

    parser.add_argument(
        '--namespace',
        type=str,
        default=config.DEFAULT_NAMESPACE,
        help=f'Namespace to query (default: {config.DEFAULT_NAMESPACE})'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=config.SCORE_THRESHOLD,
        help=f'Minimum similarity score (default: {config.SCORE_THRESHOLD})'
    )

    args = parser.parse_args()

    # Execute commands
    if args.init:
        return command_init()

    elif args.query:
        return command_query(
            args.query,
            top_k=args.top_k,
            namespace=args.namespace,
            threshold=args.threshold
        )

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
