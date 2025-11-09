"""
Smoke tests for M4.2 cost models.
Tests that estimators return sane numbers and provider tables build correctly.
"""

import pytest
from m4_2_cost_models import PineconeCostEstimator, VectorDBComparison


def test_pinecone_free_tier():
    """Test free tier calculation."""
    estimator = PineconeCostEstimator(vectors=50_000)
    result = estimator.estimate_monthly_cost()

    assert result['total_monthly'] == 0, "Free tier should be $0"
    assert result['tier'] == 'Free', "Should be Free tier"


def test_pinecone_starter_tier():
    """Test Starter tier calculation."""
    estimator = PineconeCostEstimator(vectors=500_000)
    result = estimator.estimate_monthly_cost()

    assert result['total_monthly'] == 70, f"Starter tier should be $70, got ${result['total_monthly']}"
    assert result['tier'] == 'Starter', "Should be Starter tier"


def test_pinecone_standard_tier():
    """Test Standard tier calculation."""
    estimator = PineconeCostEstimator(vectors=5_000_000)
    result = estimator.estimate_monthly_cost()

    # 5M vectors = 5 pods * $280 = $1400
    expected = 1400
    assert result['total_monthly'] == expected, f"Expected ${expected}, got ${result['total_monthly']}"
    assert result['tier'] == 'Standard', "Should be Standard tier"


def test_pinecone_replicas():
    """Test replica cost calculation."""
    estimator = PineconeCostEstimator(vectors=500_000, replicas=2)
    result = estimator.estimate_monthly_cost()

    # Base $70 + 1 replica ($70) = $140
    expected = 140
    assert result['total_monthly'] == expected, f"Expected ${expected} with 2 replicas, got ${result['total_monthly']}"
    assert result['replica_cost'] == 70, "Replica cost should be $70"


def test_break_even_analysis():
    """Test break-even calculation."""
    estimator = PineconeCostEstimator(vectors=1_000_000)
    break_even = estimator.calculate_break_even(alternative_cost=100)

    # Break-even should be somewhere between free tier and 1M vectors
    assert 100_000 < break_even < 10_000_000, f"Break-even {break_even:,} seems unreasonable"


def test_cost_per_vector():
    """Test cost per vector calculation."""
    estimator = PineconeCostEstimator(vectors=500_000)
    cost_per_vector = estimator.cost_per_vector()

    assert cost_per_vector > 0, "Cost per vector should be positive"
    assert cost_per_vector < 1, "Cost per vector should be less than $1"


def test_annual_projection():
    """Test annual cost projection."""
    estimator = PineconeCostEstimator(vectors=500_000)
    annual = estimator.annual_projection()
    monthly = estimator.estimate_monthly_cost()['total_monthly']

    assert annual == monthly * 12, "Annual should be monthly * 12"


def test_provider_features_table():
    """Test provider feature comparison table generation."""
    features = VectorDBComparison.get_provider_features()

    assert len(features) == 4, "Should have 4 providers"
    assert 'Pinecone' in features['Provider'].values, "Should include Pinecone"
    assert 'Weaviate' in features['Provider'].values, "Should include Weaviate"
    assert 'Qdrant' in features['Provider'].values, "Should include Qdrant"
    assert 'Elasticsearch' in features['Provider'].values, "Should include Elasticsearch"


def test_weaviate_cost_estimate():
    """Test Weaviate cost estimation."""
    result = VectorDBComparison.estimate_weaviate_cost(500_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] > 0, "Cost should be positive"


def test_qdrant_cost_estimate():
    """Test Qdrant cost estimation."""
    result = VectorDBComparison.estimate_qdrant_cost(50_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] == 0, "Should be free tier"

    result = VectorDBComparison.estimate_qdrant_cost(500_000, 100_000)
    assert result['monthly_cost'] > 0, "Paid tier should have cost"


def test_elasticsearch_cost_estimate():
    """Test Elasticsearch cost estimation."""
    result = VectorDBComparison.estimate_elasticsearch_cost(100_000, 100_000)

    assert 'monthly_cost' in result, "Should return monthly_cost"
    assert result['monthly_cost'] >= 95, "ES typically starts at $95"


def test_self_host_estimate():
    """Test self-hosting infrastructure estimate."""
    result = VectorDBComparison.self_host_infrastructure_estimate(1_000_000)

    assert 'compute_monthly' in result, "Should include compute costs"
    assert 'storage_monthly' in result, "Should include storage costs"
    assert 'total_infrastructure' in result, "Should include total"
    assert result['total_infrastructure'] > 0, "Total should be positive"
    assert result['storage_gb'] > 0, "Storage GB should be positive"


def test_comparison_table_generation():
    """Test cost comparison table generation."""
    scenarios = [100_000, 1_000_000]
    comparison = VectorDBComparison.generate_comparison_table(scenarios)

    assert len(comparison) == 2, "Should have 2 rows for 2 scenarios"
    assert 'Pinecone' in comparison.columns, "Should include Pinecone"
    assert 'Weaviate' in comparison.columns, "Should include Weaviate"
    assert 'Qdrant' in comparison.columns, "Should include Qdrant"


def test_query_cost_calculation():
    """Test query cost calculation for high volume."""
    # High query volume (exceeds 10M included)
    estimator = PineconeCostEstimator(vectors=500_000, monthly_queries=15_000_000)
    result = estimator.estimate_monthly_cost()

    # Should have query costs for 5M excess queries
    assert result['query_cost'] > 0, "Should have query costs for excess queries"


# Parametrized tests for tier boundaries
@pytest.mark.parametrize("vectors,expected_tier,expected_cost", [
    (50_000, 'Free', 0),
    (100_000, 'Free', 0),
    (100_001, 'Starter', 70),
    (500_000, 'Starter', 70),
    (1_000_000, 'Starter', 70),
    (1_000_001, 'Standard', 560),  # Rounds up to 2 pods
    (2_000_000, 'Standard', 560),
])
def test_tier_boundaries(vectors, expected_tier, expected_cost):
    """Test tier boundary conditions."""
    estimator = PineconeCostEstimator(vectors=vectors)
    result = estimator.estimate_monthly_cost()
    assert result['tier'] == expected_tier
    assert result['total_monthly'] == expected_cost


# Parametrized tests for rounding policy
@pytest.mark.parametrize("vectors,expected_pods", [
    (1_000_000, 1),
    (1_500_000, 2),  # Rounds up
    (2_000_000, 2),
    (2_000_001, 3),  # Rounds up
])
def test_rounding_policy(vectors, expected_pods):
    """Test standard mathematical rounding for pod calculations."""
    estimator = PineconeCostEstimator(vectors=vectors)
    result = estimator.estimate_monthly_cost()
    if result['tier'] == 'Standard':
        assert result['pods_needed'] == expected_pods


# Test monthly vs annual projection consistency
def test_annual_projection_consistency():
    """Test that annual projection is exactly 12x monthly cost."""
    for vectors in [500_000, 1_000_000, 5_000_000]:
        estimator = PineconeCostEstimator(vectors=vectors)
        monthly = estimator.estimate_monthly_cost()['total_monthly']
        annual = estimator.annual_projection()
        assert annual == monthly * 12, f"Annual should be 12x monthly for {vectors} vectors"
