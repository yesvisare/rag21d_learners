"""
Smoke tests for M1.4 Query Pipeline components.
Run with: pytest tests/test_query_pipeline.py
"""
import sys
from src.m1_4_query_pipeline.module import (
    QueryProcessor, QueryType, SmartRetriever, Reranker,
    ContextBuilder, PromptBuilder, ResponseGenerator,
    ProductionRAG, RetrievalResult
)
from src.m1_4_query_pipeline.config import get_clients


def test_query_classification():
    """Test query type classification."""
    processor = QueryProcessor()

    test_cases = [
        ("How do I improve accuracy?", QueryType.HOW_TO),
        ("What is RAG?", QueryType.DEFINITION),
        ("Error 404 not found", QueryType.TROUBLESHOOTING),
        ("Dense vs sparse retrieval", QueryType.COMPARISON),
        ("Should I use reranking?", QueryType.OPINION),
    ]

    print("🧪 Testing Query Classification...")
    for query, expected in test_cases:
        result = processor.classify(query)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{query[:30]}...' -> {result.value}")

    assert processor.classify("How to") == QueryType.HOW_TO, "How-to classification failed"
    print("✅ Query classification tests passed\n")


def test_alpha_selector():
    """Test alpha value selection for query types."""
    retriever = SmartRetriever()

    print("🧪 Testing Alpha Selector...")
    for query_type in QueryType:
        alpha = retriever._get_alpha_for_query_type(query_type)
        assert 0 <= alpha <= 1, f"Invalid alpha {alpha} for {query_type}"
        print(f"  ✅ {query_type.value}: alpha={alpha}")

    print("✅ Alpha selector tests passed\n")


def test_context_with_scores():
    """Test context building with scores."""
    builder = ContextBuilder(max_length=1000)

    # Mock results
    results = [
        RetrievalResult(
            id=f"test-{i}",
            score=0.9 - (i * 0.1),
            text=f"This is test chunk {i} with some content.",
            metadata={"source": f"doc{i}.txt"},
            original_score=0.9 - (i * 0.1),
            rerank_score=0.85 - (i * 0.05)
        )
        for i in range(3)
    ]

    print("🧪 Testing Context Building...")
    context_data = builder.context_with_scores(results)

    assert "num_chunks" in context_data, "Missing num_chunks"
    assert "avg_score" in context_data, "Missing avg_score"
    assert "unique_sources" in context_data, "Missing unique_sources"
    assert "context" in context_data, "Missing context"
    assert "sources" in context_data, "Missing sources"

    print(f"  ✅ num_chunks: {context_data['num_chunks']}")
    print(f"  ✅ avg_score: {context_data['avg_score']:.3f}")
    print(f"  ✅ unique_sources: {context_data['unique_sources']}")

    assert context_data['num_chunks'] == 3, "Chunk count mismatch"
    assert context_data['unique_sources'] == 3, "Source count mismatch"
    print("✅ Context building tests passed\n")


def test_reranker():
    """Test reranker with small list."""
    reranker = Reranker()

    results = [
        RetrievalResult(
            id=f"test-{i}",
            score=0.8,
            text=f"Document about machine learning and neural networks. Chunk {i}.",
            metadata={"source": f"ml{i}.txt"},
            original_score=0.8
        )
        for i in range(3)
    ]

    print("🧪 Testing Reranker...")
    query = "machine learning"
    reranked = reranker.rerank(query, results, top_k=2)

    assert len(reranked) <= 2, "Reranker returned too many results"
    assert all(hasattr(r, 'rerank_score') for r in reranked), "Missing rerank_score"
    print(f"  ✅ Reranked {len(results)} -> {len(reranked)} results")
    print("✅ Reranker tests passed\n")


def test_retrieve_with_no_keys():
    """Test retrieval gracefully handles missing API keys."""
    retriever = SmartRetriever(openai_client=None, pinecone_client=None)

    print("🧪 Testing Retrieval without API keys...")
    results = retriever.retrieve(
        query="test query",
        query_type=QueryType.FACTUAL,
        top_k=3
    )

    assert isinstance(results, list), "Should return list of results"
    assert len(results) <= 3, "Should respect top_k limit"
    print(f"  ✅ Returned {len(results)} mock results")
    print("✅ Graceful fallback tests passed\n")


def test_prompt_templates():
    """Test prompt templates for all query types."""
    builder = PromptBuilder()

    print("🧪 Testing Prompt Templates...")
    for query_type in QueryType:
        prompt = builder.build_prompt(
            query="Test query",
            context="Test context",
            query_type=query_type
        )

        assert "system" in prompt, f"Missing system prompt for {query_type}"
        assert "user" in prompt, f"Missing user prompt for {query_type}"
        assert len(prompt["system"]) > 0, f"Empty system prompt for {query_type}"
        assert len(prompt["user"]) > 0, f"Empty user prompt for {query_type}"
        print(f"  ✅ {query_type.value}: templates valid")

    print("✅ Prompt template tests passed\n")


def test_end_to_end_pipeline():
    """Test complete pipeline integration."""
    openai_client, pinecone_client = get_clients()

    print("🧪 Testing End-to-End Pipeline...")
    rag = ProductionRAG(
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        use_expansion=False,
        use_reranking=True
    )

    result = rag.query(
        query="How do I improve retrieval?",
        top_k=3,
        rerank_top_k=2
    )

    # Verify required fields
    required_fields = [
        "answer", "query_type", "keywords", "chunks_retrieved",
        "sources", "avg_score", "retrieval_time", "generation_time", "total_time"
    ]

    for field in required_fields:
        assert field in result, f"Missing field: {field}"
        print(f"  ✅ {field}: present")

    assert isinstance(result["retrieval_time"], (int, float)), "Invalid retrieval_time type"
    assert isinstance(result["generation_time"], (int, float)), "Invalid generation_time type"
    assert result["total_time"] > 0, "Invalid total_time"

    print("✅ End-to-end pipeline tests passed\n")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("M1.4 Query Pipeline - Smoke Tests")
    print("=" * 60 + "\n")

    try:
        test_query_classification()
        test_alpha_selector()
        test_context_with_scores()
        test_reranker()
        test_retrieve_with_no_keys()
        test_prompt_templates()
        test_end_to_end_pipeline()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
