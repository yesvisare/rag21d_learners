"""
Smoke tests for M4.2 cost models.
Tests that estimators return sane numbers and provider tables build correctly.
"""

import sys
from m4_2_cost_models import PineconeCostEstimator, VectorDBComparison


def test_pinecone_free_tier():
    """Test free tier calculation."""
    estimator = PineconeCostEstimator(vectors=50_000)
    result = estimator.estimate_monthly_cost()

    assert result['total_monthly'] == 0, "Free tier should be $0"
    assert result['tier'] == 'Free', "Should be Free tier"
    print("✓ Free tier calculation correct")


def test_pinecone_starter_tier():
    """Test Starter tier calculation."""
    estimator = PineconeCostEstimator(vectors=500_000)
    result = estimator.estimate_monthly_cost()

    assert result['total_monthly'] == 70, f"Starter tier should be $70, got ${result['total_monthly']}"
    assert result['tier'] == 'Starter', "Should be Starter tier"
    print("✓ Starter tier calculation correct")


def test_pinecone_standard_tier():
    """Test Standard tier calculation."""
    estimator = PineconeCostEstimator(vectors=5_000_000)
    result = estimator.estimate_monthly_cost()

    # 5M vectors = 5 pods * $280 = $1400
    expected = 1400
    assert result['total_monthly'] == expected, f"Expected ${expected}, got ${result['total_monthly']}"
    assert result['tier'] == 'Standard', "Should be Standard tier"
    print("✓ Standard tier calculation correct")


def test_pinecone_replicas():
    """Test replica cost calculation."""
    estimator = PineconeCostEstimator(vectors=500_000, replicas=2)
    result = estimator.estimate_monthly_cost()

    # Base $70 + 1 replica ($70) = $140
    expected = 140
    assert result['total_monthly'] == expected, f"Expected ${expected} with 2 replicas, got ${result['total_monthly']}"
    assert result['replica_cost'] == 70, "Replica cost should be $70"
    print("✓ Replica costs correct")


def test_break_even_analysis():
    """Test break-even calculation."""
    estimator = PineconeCostEstimator(vectors=1_000_000)
    break_even = estimator.calculate_break_even(alternative_cost=100)

    # Break-even should be somewhere between free tier and 1M vectors
    assert 100_000 < break_even < 10_000_000, f"Break-even {break_even:,} seems unreasonable"
    print("✓ Break-even analysis functional")


def test_cost_per_vector():
    """Test cost per vector calculation."""
    estimator = PineconeCostEstimator(vectors=500_000)
    cost_per_vector = estimator.cost_per_vector()

    assert cost_per_vector > 0, "Cost per vector should be positive"
    assert cost_per_vector < 1, "Cost per vector should be less than $1"
    print("✓ Cost per vector calculation correct")


def test_annual_projection():
    """Test annual cost projection."""
    estimator = PineconeCostEstimator(vectors=500_000)
    annual = estimator.annual_projection()
    monthly = estimator.estimate_monthly_cost()['total_monthly']

    assert annual == monthly * 12, "Annual should be monthly * 12"
    print("✓ Annual projection correct")


def test_provider_features_table():
    """Test provider feature comparison table generation."""
    features = VectorDBComparison.get_provider_features()

    assert len(features) == 4, "Should have 4 providers"
    assert 'Pinecone' in features['Provider'].values, "Should include Pinecone"
    assert 'Weaviate' in features['Provider'].values, "Should include Weaviate"
    assert 'Qdrant' in features['Provider'].values, "Should include Qdrant"
    assert 'Elasticsearch' in features['Provider'].values, "Should include Elasticsearch"
    print("✓ Provider features table generated")


def test_weaviate_cost_estimate():
    """Test Weaviate cost estimation."""
    result = VectorDBComparison.estimate_weaviate_cost(500_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] > 0, "Cost should be positive"
    print("✓ Weaviate cost estimation works")


def test_qdrant_cost_estimate():
    """Test Qdrant cost estimation."""
    result = VectorDBComparison.estimate_qdrant_cost(50_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] == 0, "Should be free tier"

    result = VectorDBComparison.estimate_qdrant_cost(500_000, 100_000)
    assert result['monthly_cost'] > 0, "Paid tier should have cost"
    print("✓ Qdrant cost estimation works")


def test_elasticsearch_cost_estimate():
    """Test Elasticsearch cost estimation."""
    result = VectorDBComparison.estimate_elasticsearch_cost(100_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] >= 95, "ES typically starts at $95"
    print("✓ Elasticsearch cost estimation works")


def test_self_host_estimate():
    """Test self-hosting infrastructure estimate."""
    result = VectorDBComparison.self_host_infrastructure_estimate(1_000_000)

    assert 'compute_monthly' in result, "Should include compute costs"
    assert 'storage_monthly' in result, "Should include storage costs"
    assert 'total_infrastructure' in result, "Should include total"
    assert result['total_infrastructure'] > 0, "Total should be positive"
    assert result['storage_gb'] > 0, "Storage GB should be positive"
    print("✓ Self-host estimates reasonable")


def test_comparison_table_generation():
    """Test cost comparison table generation."""
    scenarios = [100_000, 1_000_000]
    comparison = VectorDBComparison.generate_comparison_table(scenarios)

    assert len(comparison) == 2, "Should have 2 rows for 2 scenarios"
    assert 'Pinecone' in comparison.columns, "Should include Pinecone"
    assert 'Weaviate' in comparison.columns, "Should include Weaviate"
    assert 'Qdrant' in comparison.columns, "Should include Qdrant"
    print("✓ Cost comparison table generated")


def test_query_cost_calculation():
    """Test query cost calculation for high volume."""
    # High query volume (exceeds 10M included)
    estimator = PineconeCostEstimator(vectors=500_000, monthly_queries=15_000_000)
    result = estimator.estimate_monthly_cost()

    # Should have query costs for 5M excess queries
    assert result['query_cost'] > 0, "Should have query costs for excess queries"
    print("✓ Query cost calculation works")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 50)
    print("Running M4.2 Cost Models Smoke Tests")
    print("=" * 50)

    print("\nTesting PineconeCostEstimator...")
    test_pinecone_free_tier()
    test_pinecone_starter_tier()
    test_pinecone_standard_tier()
    test_pinecone_replicas()
    test_cost_per_vector()
    test_annual_projection()
    test_break_even_analysis()
    test_query_cost_calculation()

    print("\nTesting VectorDBComparison...")
    test_provider_features_table()
    test_weaviate_cost_estimate()
    test_qdrant_cost_estimate()
    test_elasticsearch_cost_estimate()
    test_self_host_estimate()
    test_comparison_table_generation()

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        run_all_tests()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
