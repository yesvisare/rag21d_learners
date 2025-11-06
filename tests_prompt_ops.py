"""
Smoke tests for M2.2 Prompt Optimization module

Run with: python tests_prompt_ops.py
"""

import json
from m2_2_prompt_ops import (
    RAGPromptLibrary,
    TokenEstimator,
    ModelRouter,
    ModelTier,
    PromptTester,
    format_context_optimally,
)


def test_token_estimator():
    """Test token counting and cost estimation."""
    print("Testing TokenEstimator...")

    estimator = TokenEstimator("gpt-3.5-turbo")

    # Test token counting
    text = "Hello, world! This is a test."
    tokens = estimator.count_tokens(text)
    assert tokens > 0, "Token count should be positive"
    assert tokens < 100, "Token count should be reasonable"

    # Test cost estimation
    cost = estimator.estimate_cost(input_tokens=100, output_tokens=50)
    assert cost > 0, "Cost should be positive"
    assert cost < 1.0, "Cost should be reasonable for small token counts"

    # Test monthly projection
    projection = estimator.project_monthly_cost(
        avg_input_tokens=300,
        avg_output_tokens=150,
        queries_per_day=1000
    )
    assert "monthly_cost" in projection
    assert projection["monthly_cost"] > 0

    print("  ✓ Token counting works")
    print("  ✓ Cost estimation works")
    print("  ✓ Monthly projection works")
    print()


def test_prompt_library():
    """Test prompt template library."""
    print("Testing RAGPromptLibrary...")

    # Test all templates exist
    templates = [
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
        RAGPromptLibrary.STRUCTURED_RAG,
        RAGPromptLibrary.JSON_RAG,
        RAGPromptLibrary.SUPPORT_RAG,
    ]

    for template in templates:
        assert template.system_prompt, "System prompt should not be empty"
        assert template.user_template, "User template should not be empty"
        assert template.tokens_estimate > 0, "Token estimate should be positive"
        assert template.use_case, "Use case should not be empty"

    # Test token estimates are ordered correctly
    assert RAGPromptLibrary.BASIC_RAG.tokens_estimate > RAGPromptLibrary.CONCISE_RAG.tokens_estimate
    assert RAGPromptLibrary.CONCISE_RAG.tokens_estimate > RAGPromptLibrary.JSON_RAG.tokens_estimate

    # Test get_template_by_name
    template = RAGPromptLibrary.get_template_by_name("cost_optimization")
    assert template == RAGPromptLibrary.CONCISE_RAG

    print("  ✓ All templates valid")
    print("  ✓ Token estimates ordered correctly")
    print("  ✓ Template lookup works")
    print()


def test_model_router():
    """Test intelligent model routing."""
    print("Testing ModelRouter...")

    router = ModelRouter()

    # Test simple query
    simple_query = "What is your return policy?"
    decision = router.select_model(simple_query, context="")
    assert decision["model"] in ["gpt-3.5-turbo", "gpt-4o-mini"]
    assert decision["complexity_score"] < 5, "Simple query should have low complexity"

    # Test complex query
    complex_query = "Compare and analyze the performance differences between Q3 and Q4, explaining the key factors driving these changes and their long-term impact."
    decision = router.select_model(complex_query, context="")
    assert decision["complexity_score"] >= 3, "Complex query should have higher complexity"
    assert "factors" in decision["complexity_factors"] or decision["complexity_score"] > 0

    # Test forced tier
    decision = router.select_model(
        "Test query",
        context="",
        force_tier=ModelTier.PREMIUM
    )
    assert decision["model"] == ModelTier.PREMIUM.value
    assert decision["reason"] == "forced_selection"

    # Test cost budget constraint
    decision = router.select_model(
        "Test query",
        context="Some context",
        cost_budget=0.0001  # Very low budget
    )
    assert decision["model"] == "gpt-3.5-turbo", "Should downgrade to fast model with low budget"

    print("  ✓ Simple queries route to fast models")
    print("  ✓ Complex queries route to better models")
    print("  ✓ Force tier override works")
    print("  ✓ Cost budget constraint works")
    print()


def test_context_formatting():
    """Test context formatting and truncation."""
    print("Testing format_context_optimally...")

    # Load example data
    with open("example_data.json", "r") as f:
        data = json.load(f)

    documents = data["documents"]
    estimator = TokenEstimator()

    # Test basic formatting
    formatted = format_context_optimally(
        documents,
        max_tokens=500,
        include_metadata=False,
        estimator=estimator
    )
    assert formatted, "Formatted context should not be empty"

    # Check token limit respected
    tokens = estimator.count_tokens(formatted)
    assert tokens <= 520, f"Context should be ~500 tokens, got {tokens}"  # Allow small overage

    # Test with metadata
    formatted_with_meta = format_context_optimally(
        documents,
        max_tokens=500,
        include_metadata=True,
        estimator=estimator
    )
    assert "Score:" in formatted_with_meta, "Metadata should be included"

    # Test aggressive truncation
    formatted_small = format_context_optimally(
        documents,
        max_tokens=100,
        include_metadata=False,
        estimator=estimator
    )
    tokens_small = estimator.count_tokens(formatted_small)
    assert tokens_small < tokens, "Aggressive truncation should use fewer tokens"
    assert "[truncated]" in formatted_small or tokens_small < 120, "Should indicate truncation"

    print("  ✓ Basic formatting works")
    print("  ✓ Token limits respected")
    print("  ✓ Metadata inclusion works")
    print("  ✓ Aggressive truncation works")
    print()


def test_prompt_tester_dry_run():
    """Test PromptTester in dry run mode (no API key needed)."""
    print("Testing PromptTester (dry run mode)...")

    # Load example data
    with open("example_data.json", "r") as f:
        data = json.load(f)

    # Create tester in dry run mode
    tester = PromptTester(
        openai_client=None,
        model="gpt-3.5-turbo",
        dry_run=True
    )

    # Test single template
    result = tester.test_prompt_template(
        RAGPromptLibrary.CONCISE_RAG,
        data["test_queries"][:2],
        data["documents"]
    )

    assert result.template_name == "cost_optimization"
    assert result.avg_input_tokens > 0
    assert result.avg_output_tokens > 0
    assert result.avg_cost_per_query > 0
    assert result.queries_tested == 2

    # Test comparison (dry run shouldn't crash)
    templates = [
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
    ]

    results = tester.compare_templates(
        templates,
        data["test_queries"][:2],
        data["documents"],
        output_format="table"
    )

    assert len(results) == 2, "Should return results for both templates"
    assert results[0].avg_cost_per_query <= results[-1].avg_cost_per_query, "Should be sorted by cost"

    print("  ✓ Single template testing works")
    print("  ✓ Template comparison works")
    print("  ✓ Dry run mode handles missing API key")
    print()


def test_integration():
    """Test integration of multiple components."""
    print("Testing integration...")

    # Load data
    with open("example_data.json", "r") as f:
        data = json.load(f)

    # Initialize components
    router = ModelRouter()
    estimator = TokenEstimator()

    # Simulate a query pipeline
    query = data["test_queries"][0]["question"]
    documents = data["documents"]

    # 1. Route to appropriate model
    decision = router.select_model(query, context="")
    assert decision["model"], "Should select a model"

    # 2. Format context optimally
    formatted_context = format_context_optimally(
        documents,
        max_tokens=500,
        estimator=estimator
    )
    assert formatted_context, "Should format context"

    # 3. Select appropriate template
    template = RAGPromptLibrary.CONCISE_RAG

    # 4. Estimate tokens and cost
    system_tokens = estimator.count_tokens(template.system_prompt)
    user_tokens = estimator.count_tokens(
        template.user_template.format(
            context=formatted_context,
            question=query
        )
    )
    total_input_tokens = system_tokens + user_tokens

    cost = estimator.estimate_cost(
        input_tokens=total_input_tokens,
        output_tokens=150,
        model=decision["model"]
    )

    assert total_input_tokens > 0
    assert cost > 0

    print("  ✓ Query pipeline integration works")
    print(f"    Query: {query[:50]}...")
    print(f"    Model: {decision['model']}")
    print(f"    Input tokens: {total_input_tokens}")
    print(f"    Estimated cost: ${cost:.6f}")
    print()


def run_all_tests():
    """Run all smoke tests."""
    print("="*60)
    print("M2.2 Prompt Optimization - Smoke Tests")
    print("="*60)
    print()

    try:
        test_token_estimator()
        test_prompt_library()
        test_model_router()
        test_context_formatting()
        test_prompt_tester_dry_run()
        test_integration()

        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        print()
        print("Next steps:")
        print("1. Run the Jupyter notebook: jupyter notebook M2_2_Prompt_Optimization_and_Model_Selection.ipynb")
        print("2. Add your OPENAI_API_KEY to .env for live testing")
        print("3. Replace example_data.json with your real data")

        return True

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
