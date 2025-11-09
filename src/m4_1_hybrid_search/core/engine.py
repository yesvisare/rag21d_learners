"""
M4.1 Hybrid Search Implementation
Combines BM25 sparse retrieval with dense vector embeddings.
Supports alpha weighting and Reciprocal Rank Fusion (RRF) merging.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class HybridSearchEngine:
    """
    Hybrid search combining BM25 sparse retrieval with dense embeddings.

    Features:
    - Dense embeddings via OpenAI text-embedding-3-small
    - Sparse BM25 retrieval with NLTK tokenization
    - Alpha weighting and RRF merge strategies
    - Smart alpha tuning based on query characteristics
    - Metadata filtering and namespace support
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: str = "default"
    ):
        """
        Initialize the hybrid search engine.

        Args:
            openai_api_key: OpenAI API key for embeddings
            pinecone_api_key: Pinecone API key for vector storage
            index_name: Pinecone index name
            namespace: Namespace for document organization
        """
        self.namespace = namespace
        self.documents = []
        self.tokenized_docs = []
        self.bm25_index = None

        # Initialize OpenAI client
        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None

        # Initialize Pinecone
        if PINECONE_AVAILABLE and pinecone_api_key and index_name:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
        else:
            self.pc = None
            self.index = None

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the search engine.

        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
        """
        self.documents = documents

        # Tokenize documents for BM25
        self.tokenized_docs = [
            word_tokenize(doc["text"].lower())
            for doc in documents
        ]

        # Build BM25 index
        self.bm25_index = BM25Okapi(self.tokenized_docs)

        print(f"✓ Added {len(documents)} documents to BM25 index")

    def upsert_to_pinecone(self, batch_size: int = 100) -> None:
        """
        Generate embeddings and upsert documents to Pinecone.

        Args:
            batch_size: Number of documents to process per batch
        """
        if not self.openai_client or not self.index:
            print("⚠ OpenAI or Pinecone not configured. Skipping upsert.")
            return

        vectors = []

        for i, doc in enumerate(self.documents):
            # Generate embedding
            embedding = self._get_embedding(doc["text"])

            # Prepare vector with metadata
            vector = {
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    "text": doc["text"],
                    **(doc.get("metadata", {}))
                }
            }
            vectors.append(vector)

            # Upsert batch
            if len(vectors) >= batch_size or i == len(self.documents) - 1:
                self.index.upsert(
                    vectors=vectors,
                    namespace=self.namespace
                )
                print(f"✓ Upserted batch {i//batch_size + 1} ({len(vectors)} vectors)")
                vectors = []

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        if not self.openai_client:
            # Return stub embedding if client not available
            return [0.0] * 1536

        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def smart_alpha(self, query: str) -> float:
        """
        Dynamically determine alpha based on query characteristics.

        Args:
            query: Search query text

        Returns:
            Alpha value between 0.0 and 1.0
        """
        # Check for technical patterns (codes, SKUs, exact terms)
        has_code = bool(re.search(r'\b[A-Z0-9]{3,}-\d+\b', query))  # Pattern like ABC-123
        has_sku = bool(re.search(r'\b\d{5,}\b', query))  # 5+ digit numbers
        has_quotes = '"' in query  # Exact phrase match

        # Count technical terms (uppercase words, numbers)
        words = query.split()
        technical_count = sum(1 for w in words if w.isupper() or w.isdigit())
        technical_ratio = technical_count / len(words) if words else 0

        # Determine alpha based on characteristics
        if has_code or has_sku or has_quotes:
            # Strong keyword signal - favor sparse
            return 0.3
        elif technical_ratio > 0.4:
            # Many technical terms - balanced
            return 0.5
        else:
            # Natural language - favor dense
            return 0.7

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25 sparse retrieval.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of results with id, text, and score
        """
        if not self.bm25_index:
            return []

        # Tokenize query
        tokenized_query = word_tokenize(query.lower())

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx]),
                "metadata": self.documents[idx].get("metadata", {})
            })

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using dense vector embeddings.

        Args:
            query: Search query text
            top_k: Number of results to return
            metadata_filter: Optional Pinecone metadata filter

        Returns:
            List of results with id, text, and score
        """
        if not self.openai_client or not self.index:
            print("⚠ OpenAI or Pinecone not configured. Returning empty results.")
            return []

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Query Pinecone
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=metadata_filter
        )

        results = []
        for match in response.matches:
            results.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "score": float(match.score),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            })

        return results

    def hybrid_search_alpha(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using alpha weighting.

        Formula: combined_score = alpha * dense_score + (1 - alpha) * sparse_score

        Args:
            query: Search query text
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0.0=sparse only, 1.0=dense only)
                   If None, uses smart_alpha()
            metadata_filter: Optional Pinecone metadata filter

        Returns:
            List of results sorted by combined score
        """
        # Determine alpha
        if alpha is None:
            alpha = self.smart_alpha(query)

        # Get results from both systems (retrieve more candidates)
        dense_results = self.search_dense(query, top_k * 2, metadata_filter)
        sparse_results = self.search_bm25(query, top_k * 2)

        # Normalize scores to 0-1 range
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        # Combine scores
        combined = {}

        for doc_id, score in dense_scores.items():
            combined[doc_id] = alpha * score

        for doc_id, score in sparse_scores.items():
            if doc_id in combined:
                combined[doc_id] += (1 - alpha) * score
            else:
                combined[doc_id] = (1 - alpha) * score

        # Sort and return top k
        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        results = []
        doc_map = {doc["id"]: doc for doc in dense_results + sparse_results}

        for doc_id, score in sorted_ids:
            if doc_id in doc_map:
                result = doc_map[doc_id].copy()
                result["score"] = float(score)
                result["alpha_used"] = alpha
                results.append(result)

        return results

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 10,
        k: int = 60,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Formula: rrf_score = sum(1 / (k + rank + 1))

        Args:
            query: Search query text
            top_k: Number of results to return
            k: RRF constant (default 60)
            metadata_filter: Optional Pinecone metadata filter

        Returns:
            List of results sorted by RRF score
        """
        # Get results from both systems
        dense_results = self.search_dense(query, top_k * 2, metadata_filter)
        sparse_results = self.search_bm25(query, top_k * 2)

        # Calculate RRF scores
        rrf_scores = {}

        # Add dense ranks
        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = 1.0 / (k + rank + 1)

        # Add sparse ranks
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            else:
                rrf_scores[doc_id] = 1.0 / (k + rank + 1)

        # Sort and return top k
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        results = []
        doc_map = {doc["id"]: doc for doc in dense_results + sparse_results}

        for doc_id, score in sorted_ids:
            if doc_id in doc_map:
                result = doc_map[doc_id].copy()
                result["score"] = float(score)
                result["merge_method"] = "RRF"
                results.append(result)

        return results

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Normalize scores to 0-1 range.

        Args:
            results: List of search results with scores

        Returns:
            Dict mapping doc_id to normalized score
        """
        if not results:
            return {}

        scores = [r["score"] for r in results]
        max_score = max(scores) if scores else 1.0

        # Avoid division by zero
        if max_score == 0:
            max_score = 1.0

        return {
            r["id"]: r["score"] / max_score
            for r in results
        }


def demo_stub():
    """Stub demo function when API keys are missing."""
    print("\n" + "="*60)
    print("STUB MODE: API keys not configured")
    print("="*60)

    # Create engine without API keys
    engine = HybridSearchEngine()

    # Add sample documents
    docs = [
        {"id": "doc1", "text": "Python is a programming language", "metadata": {"category": "tech"}},
        {"id": "doc2", "text": "Machine learning uses algorithms", "metadata": {"category": "ai"}},
        {"id": "doc3", "text": "Product SKU 12345 available", "metadata": {"category": "product"}}
    ]

    engine.add_documents(docs)

    # Test BM25 search
    print("\n🔍 BM25 Search for 'python programming':")
    results = engine.search_bm25("python programming", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['id']}: {r['text'][:50]}... (score={r['score']:.4f})")

    # Test smart alpha
    print("\n🧠 Smart Alpha Detection:")
    queries = [
        "python programming",
        "SKU 12345",
        "\"exact phrase match\""
    ]
    for q in queries:
        alpha = engine.smart_alpha(q)
        print(f"  '{q}' → alpha={alpha:.2f}")

    print("\n✓ Stub demo complete!")


if __name__ == "__main__":
    # Check if API keys are available
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key.startswith("sk-"):
        print("API keys detected. Use notebook for full demo.")
    else:
        demo_stub()
