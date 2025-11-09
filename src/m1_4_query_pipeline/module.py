"""
M1.4 — Query Pipeline & Response Generation
Complete 7-stage RAG pipeline: Query→Retrieval→Rerank→Context→LLM→Answer
"""
import logging
import time
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Six query types for classification and optimization."""
    FACTUAL = "factual"
    HOW_TO = "how-to"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    TROUBLESHOOTING = "troubleshooting"
    OPINION = "opinion"


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryProcessor:
    """
    Stage 1: Query Understanding
    - Classify query type
    - Expand query with alternatives (when LLM available)
    - Extract keywords
    """

    def __init__(self, openai_client=None):
        self.openai_client = openai_client

    def classify(self, query: str) -> QueryType:
        """
        Classify query into one of six types using heuristic rules.

        Args:
            query: User query string

        Returns:
            QueryType enum value
        """
        query_lower = query.lower()

        # How-to patterns (user wants procedural steps)
        if any(pattern in query_lower for pattern in ["how to", "how do i", "how can i", "steps to"]):
            return QueryType.HOW_TO

        # Comparison patterns (user contrasting options)
        if any(pattern in query_lower for pattern in ["vs", "versus", "compare", "difference between", "better than"]):
            return QueryType.COMPARISON

        # Definition patterns (user seeking concept explanation)
        if any(pattern in query_lower for pattern in ["what is", "what are", "define", "meaning of"]):
            return QueryType.DEFINITION

        # Troubleshooting patterns (user debugging issue)
        if any(pattern in query_lower for pattern in ["error", "fix", "not working", "failed", "issue", "problem"]):
            return QueryType.TROUBLESHOOTING

        # Opinion patterns (user seeking recommendation)
        if any(pattern in query_lower for pattern in ["should i", "recommend", "best", "opinion", "think about"]):
            return QueryType.OPINION

        # Default to factual
        return QueryType.FACTUAL

    def expand(self, query: str, num_expansions: int = 2) -> List[str]:
        """
        Generate query expansions using LLM (when available).

        Rationale: Alternative phrasings can retrieve documents missed by
        the original query due to vocabulary mismatch (15-25% recall gain).

        Args:
            query: Original query
            num_expansions: Number of alternative phrasings to generate

        Returns:
            List of expanded queries (includes original)
        """
        if not self.openai_client:
            logger.warning("⚠️ No OpenAI client available for query expansion")
            return [query]

        try:
            from src.m1_4_query_pipeline.config import OPENAI_MODEL

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Generate alternative phrasings for the user's query. Return only the alternatives, one per line."},
                    {"role": "user", "content": f"Query: {query}\n\nGenerate {num_expansions} alternative phrasings:"}
                ],
                temperature=0.7,
                max_tokens=150
            )

            expansions = [query]
            content = response.choices[0].message.content.strip()
            for line in content.split('\n'):
                line = line.strip()
                if line and len(expansions) < num_expansions + 1:
                    # Remove numbering if present (e.g., "1. " or "1) ")
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    expansions.append(line)

            return expansions[:num_expansions + 1]

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    def extract_keywords(self, query: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key terms from query for filtering.

        Args:
            query: User query
            max_keywords: Maximum keywords to extract

        Returns:
            List of keywords
        """
        # Remove common stop words (heuristic-based extraction)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords[:max_keywords]


class SmartRetriever:
    """
    Stage 2: Retrieval Strategy
    - Hybrid dense + sparse retrieval
    - Query-type specific alpha tuning
    - Metadata filtering
    """

    def __init__(self, openai_client=None, pinecone_client=None, index_name: str = "production-rag"):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index_name = index_name
        self.bm25_encoder = None
        self._fitted = False

    def _get_alpha_for_query_type(self, query_type: QueryType) -> float:
        """
        Get optimal alpha (dense weight) for query type.

        Rationale: Different query types benefit from different dense/sparse balances.
        Troubleshooting needs exact term matching (low alpha), factual needs semantics (high alpha).

        Args:
            query_type: Query type enum

        Returns:
            Alpha value between 0 and 1
        """
        alpha_map = {
            QueryType.FACTUAL: 0.7,          # Favor semantic understanding
            QueryType.HOW_TO: 0.5,           # Balanced approach
            QueryType.COMPARISON: 0.6,       # Moderate semantic bias
            QueryType.DEFINITION: 0.7,       # Favor semantic understanding
            QueryType.TROUBLESHOOTING: 0.3,  # Favor exact terms (error codes)
            QueryType.OPINION: 0.6           # Moderate semantic bias
        }
        return alpha_map.get(query_type, 0.5)

    def fit_bm25(self, documents: List[str]) -> None:
        """
        Fit BM25 encoder on document corpus.

        Args:
            documents: List of document texts
        """
        try:
            from pinecone_text.sparse import BM25Encoder

            self.bm25_encoder = BM25Encoder()
            self.bm25_encoder.fit(documents)
            self._fitted = True
            logger.info(f"BM25 encoder fitted on {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to fit BM25 encoder: {e}")

    def retrieve(
        self,
        query: str,
        query_type: QueryType,
        top_k: int = 5,
        namespace: str = "demo",
        metadata_filter: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with auto-tuned alpha.

        Args:
            query: Search query
            query_type: Query classification
            top_k: Number of results to retrieve
            namespace: Pinecone namespace
            metadata_filter: Optional metadata filters

        Returns:
            List of RetrievalResult objects
        """
        if not self.openai_client or not self.pinecone_client:
            logger.warning("⚠️ Skipping API calls (no keys found)")
            return self._mock_results(query, top_k)

        try:
            from src.m1_4_query_pipeline.config import EMBEDDING_MODEL

            # Get dense embedding
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            dense_vector = response.data[0].embedding

            # Get sparse encoding (if fitted)
            sparse_vector = None
            if self._fitted and self.bm25_encoder:
                sparse_vector = self.bm25_encoder.encode_queries(query)

            # Get alpha for query type (auto-tuning)
            alpha = self._get_alpha_for_query_type(query_type)

            # Query Pinecone
            index = self.pinecone_client.Index(self.index_name)

            query_params = {
                "vector": dense_vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True
            }

            if sparse_vector:
                query_params["sparse_vector"] = sparse_vector
                query_params["alpha"] = alpha

            if metadata_filter:
                query_params["filter"] = metadata_filter

            results = index.query(**query_params)

            # Convert to RetrievalResult objects
            retrieval_results = []
            for match in results.get("matches", []):
                retrieval_results.append(RetrievalResult(
                    id=match.get("id", ""),
                    score=match.get("score", 0.0),
                    text=match.get("metadata", {}).get("text", ""),
                    metadata=match.get("metadata", {}),
                    original_score=match.get("score", 0.0)
                ))

            return retrieval_results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return self._mock_results(query, top_k)

    def _mock_results(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Generate mock results when API unavailable (graceful degradation)."""
        return [
            RetrievalResult(
                id=f"mock-{i}",
                score=0.85 - (i * 0.05),
                text=f"Mock result {i+1} for query: {query[:50]}...",
                metadata={"source": "mock"},
                original_score=0.85 - (i * 0.05)
            )
            for i in range(min(top_k, 3))
        ]


class Reranker:
    """
    Stage 3: Reranking with Cross-Encoder
    - Apply ms-marco-MiniLM-L-6-v2 cross-encoder
    - Preserve original scores
    - Sort by rerank score

    Rationale: Cross-encoders provide 10-20% better relevance than bi-encoders
    by jointly encoding query+document, but at 50-100ms latency cost.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy load cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Loaded reranker: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Optional limit on returned results

        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return results

        self._load_model()

        if self.model is None:
            logger.warning("Reranker model not available, returning original results")
            return results

        try:
            # Prepare query-document pairs
            pairs = [(query, result.text) for result in results]

            # Get rerank scores
            rerank_scores = self.model.predict(pairs)

            # Update results with rerank scores (preserve original for comparison)
            for result, score in zip(results, rerank_scores):
                result.rerank_score = float(score)

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.rerank_score or 0, reverse=True)

            # Apply top_k if specified
            if top_k:
                reranked = reranked[:top_k]

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results


class ContextBuilder:
    """
    Stage 4: Context Preparation
    - Deduplicate results
    - Add source tags
    - Enforce max length
    """

    def __init__(self, max_length: int = 4000):
        self.max_length = max_length

    def build_context(self, results: List[RetrievalResult]) -> str:
        """
        Build context string from results.

        Args:
            results: List of retrieval results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        # Deduplicate by text content (use first 100 chars as dedup key)
        seen_texts = set()
        unique_results = []
        for result in results:
            text_key = result.text.strip()[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)

        # Build context with source tags (for attribution/transparency)
        context_parts = []
        current_length = 0

        for i, result in enumerate(unique_results, 1):
            source = result.metadata.get("source", result.id)
            chunk = f"[Source {i}: {source}]\n{result.text}\n"

            if current_length + len(chunk) > self.max_length:
                break

            context_parts.append(chunk)
            current_length += len(chunk)

        return "\n".join(context_parts)

    def context_with_scores(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Build context with metadata including scores.

        Args:
            results: List of retrieval results

        Returns:
            Dict with context, scores, and metadata
        """
        context = self.build_context(results)

        # Calculate metrics
        num_chunks = len(results)
        avg_score = np.mean([r.rerank_score or r.score for r in results]) if results else 0.0
        unique_sources = len(set(r.metadata.get("source", r.id) for r in results))

        return {
            "context": context,
            "num_chunks": num_chunks,
            "avg_score": float(avg_score),
            "unique_sources": unique_sources,
            "sources": [r.metadata.get("source", r.id) for r in results]
        }


class PromptBuilder:
    """
    Stage 5: Prompt Engineering
    - Query-type specific templates
    - Context injection
    - System prompts
    """

    TEMPLATES = {
        QueryType.FACTUAL: {
            "system": "You are a precise AI assistant. Provide factual answers based strictly on the given context. If information is not in the context, say so.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a factual answer based only on the context above."
        },
        QueryType.HOW_TO: {
            "system": "You are a helpful AI assistant. Provide clear, step-by-step instructions based on the given context.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide step-by-step instructions based on the context above."
        },
        QueryType.COMPARISON: {
            "system": "You are an analytical AI assistant. Compare and contrast options based on the given context.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a balanced comparison based on the context above."
        },
        QueryType.DEFINITION: {
            "system": "You are a knowledgeable AI assistant. Provide clear definitions based on the given context.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear definition based on the context above."
        },
        QueryType.TROUBLESHOOTING: {
            "system": "You are a technical AI assistant. Help diagnose and solve problems based on the given context.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide troubleshooting steps based on the context above."
        },
        QueryType.OPINION: {
            "system": "You are a thoughtful AI assistant. Provide balanced perspectives based on the given context.",
            "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a balanced perspective based on the context above."
        }
    }

    def build_prompt(
        self,
        query: str,
        context: str,
        query_type: QueryType
    ) -> Dict[str, str]:
        """
        Build prompt for given query type.

        Args:
            query: User query
            context: Retrieved context
            query_type: Query classification

        Returns:
            Dict with 'system' and 'user' messages
        """
        template = self.TEMPLATES.get(query_type, self.TEMPLATES[QueryType.FACTUAL])

        return {
            "system": template["system"],
            "user": template["user"].format(context=context, query=query)
        }


class ResponseGenerator:
    """
    Stage 6: LLM Response Generation
    - Non-streaming generation
    - Streaming generation
    """

    def __init__(self, openai_client=None):
        self.openai_client = openai_client

    def generate(
        self,
        prompt: Dict[str, str],
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        """
        Generate non-streaming response.

        Args:
            prompt: Dict with 'system' and 'user' messages
            temperature: Sampling temperature (0.1 for factual consistency)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        if not self.openai_client:
            logger.warning("⚠️ Skipping API calls (no keys found)")
            return "Mock response: API keys not configured"

        try:
            from src.m1_4_query_pipeline.config import OPENAI_MODEL

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    def stream(
        self,
        prompt: Dict[str, str],
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        """
        Generate streaming response.

        Args:
            prompt: Dict with 'system' and 'user' messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Response chunks
        """
        if not self.openai_client:
            logger.warning("⚠️ Skipping API calls (no keys found)")
            yield "Mock streaming response..."
            return

        try:
            from src.m1_4_query_pipeline.config import OPENAI_MODEL

            stream = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"


class ProductionRAG:
    """
    Stage 7: Complete Pipeline Integration
    - End-to-end query processing
    - Timing and metrics
    - Source attribution
    """

    def __init__(
        self,
        openai_client=None,
        pinecone_client=None,
        use_expansion: bool = False,
        use_reranking: bool = True
    ):
        self.query_processor = QueryProcessor(openai_client)
        self.retriever = SmartRetriever(openai_client, pinecone_client)
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.response_generator = ResponseGenerator(openai_client)

        self.use_expansion = use_expansion
        self.use_reranking = use_reranking

    def query(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = 3,
        namespace: str = "demo",
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline.

        Args:
            query: User query
            top_k: Number of initial results
            rerank_top_k: Number after reranking
            namespace: Pinecone namespace
            temperature: Generation temperature

        Returns:
            Dict with answer, sources, and metrics

        Raises:
            None - gracefully handles errors with logging
        """
        start_time = time.time()
        metrics = {}

        # Stage 1: Query Understanding
        query_type = self.query_processor.classify(query)
        keywords = self.query_processor.extract_keywords(query)

        if self.use_expansion:
            expansions = self.query_processor.expand(query)
        else:
            expansions = [query]

        # Stage 2: Retrieval
        retrieval_start = time.time()
        results = self.retriever.retrieve(
            query=expansions[0],
            query_type=query_type,
            top_k=top_k,
            namespace=namespace
        )
        retrieval_time = time.time() - retrieval_start

        # Stage 3: Reranking (optional trade-off: +50-100ms latency, +10-20% relevance)
        rerank_start = time.time()
        if self.use_reranking and results:
            results = self.reranker.rerank(query, results, rerank_top_k)
        rerank_time = time.time() - rerank_start

        # Stage 4: Context Building
        context_data = self.context_builder.context_with_scores(results)

        # Stage 5: Prompt Building
        prompt = self.prompt_builder.build_prompt(
            query=query,
            context=context_data["context"],
            query_type=query_type
        )

        # Stage 6: Generation
        generation_start = time.time()
        answer = self.response_generator.generate(
            prompt=prompt,
            temperature=temperature
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "query_type": query_type.value,
            "keywords": keywords,
            "chunks_retrieved": len(results),
            "sources": context_data["sources"],
            "avg_score": context_data["avg_score"],
            "retrieval_time": round(retrieval_time, 3),
            "rerank_time": round(rerank_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3)
        }


def main():
    """CLI interface for query pipeline."""
    import argparse
    from src.m1_4_query_pipeline.config import get_clients

    parser = argparse.ArgumentParser(description="M1.4 Query Pipeline")
    parser.add_argument("--ask", type=str, help="Query to process")
    parser.add_argument("--stream", type=str, help="Query with streaming response")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--rerank", type=int, default=1, help="Enable reranking (1=yes, 0=no)")
    parser.add_argument("--expand", type=int, default=0, help="Enable query expansion (1=yes, 0=no)")
    parser.add_argument("--namespace", type=str, default="demo", help="Pinecone namespace")

    args = parser.parse_args()

    if not args.ask and not args.stream:
        parser.print_help()
        return

    # Get clients
    openai_client, pinecone_client = get_clients()

    # Initialize pipeline
    rag = ProductionRAG(
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        use_expansion=bool(args.expand),
        use_reranking=bool(args.rerank)
    )

    # Process query
    if args.ask:
        print(f"\n🔍 Query: {args.ask}\n")
        result = rag.query(
            query=args.ask,
            top_k=args.top_k,
            rerank_top_k=3,
            namespace=args.namespace
        )

        print(f"📊 Type: {result['query_type']}")
        print(f"⏱️  Times: retrieval={result['retrieval_time']}s, generation={result['generation_time']}s")
        print(f"📄 Retrieved: {result['chunks_retrieved']} chunks")
        print(f"📚 Sources: {', '.join(result['sources'][:3])}")
        print(f"\n💬 Answer:\n{result['answer']}\n")

    elif args.stream:
        print(f"\n🔍 Query (streaming): {args.stream}\n")

        # Get context first
        query_processor = QueryProcessor(openai_client)
        retriever = SmartRetriever(openai_client, pinecone_client)
        reranker = Reranker()
        context_builder = ContextBuilder()
        prompt_builder = PromptBuilder()
        response_gen = ResponseGenerator(openai_client)

        query_type = query_processor.classify(args.stream)
        results = retriever.retrieve(args.stream, query_type, args.top_k, args.namespace)

        if args.rerank:
            results = reranker.rerank(args.stream, results, 3)

        context_data = context_builder.context_with_scores(results)
        prompt = prompt_builder.build_prompt(args.stream, context_data["context"], query_type)

        print("💬 Answer: ", end="", flush=True)
        for chunk in response_gen.stream(prompt):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
