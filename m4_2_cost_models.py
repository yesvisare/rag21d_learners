"""
M4.2 - Pinecone Cost Estimator & Vector DB Comparison
Illustrative pricing models for vector database cost analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PineconeCostEstimator:
    """
    Pinecone cost calculator based on vectors, replicas, and query volume.
    Pricing: Free tier = 100K vectors; Starter = $70/mo; Standard = $280/mo+
    """

    # Illustrative pricing constants (USD/month)
    FREE_TIER_VECTORS = 100_000
    STARTER_POD_COST = 70
    STARTER_POD_CAPACITY = 1_000_000  # Starter can handle up to 1M vectors
    STANDARD_POD_BASE = 280
    STANDARD_POD_CAPACITY = 1_000_000

    # Query pricing (illustrative - typically bundled with pod costs)
    QUERIES_INCLUDED = 10_000_000  # Monthly queries included
    COST_PER_MILLION_QUERIES = 5   # After included quota

    def __init__(self, vectors: int, dimensions: int = 1536, replicas: int = 1,
                 monthly_queries: int = 100_000):
        """
        Initialize cost estimator.

        Args:
            vectors: Number of vectors to store
            dimensions: Embedding dimension (default 1536 for OpenAI)
            replicas: Number of replicas for HA (default 1)
            monthly_queries: Expected monthly query volume
        """
        self.vectors = vectors
        self.dimensions = dimensions
        self.replicas = replicas
        self.monthly_queries = monthly_queries

    def estimate_monthly_cost(self) -> Dict[str, float]:
        """
        Calculate estimated monthly costs.

        Returns:
            Dictionary with cost breakdown
        """
        # Free tier check
        if self.vectors <= self.FREE_TIER_VECTORS and self.replicas == 1:
            return {
                'storage_cost': 0,
                'query_cost': 0,
                'replica_cost': 0,
                'total_monthly': 0,
                'tier': 'Free'
            }

        # Calculate pod requirements
        if self.vectors <= self.STARTER_POD_CAPACITY:
            base_cost = self.STARTER_POD_COST
            tier = 'Starter'
        else:
            # Standard tier - scale by pods needed
            pods_needed = np.ceil(self.vectors / self.STANDARD_POD_CAPACITY)
            base_cost = self.STANDARD_POD_BASE * pods_needed
            tier = 'Standard'

        # Replica costs (each replica = full pod cost)
        storage_cost = base_cost
        replica_cost = base_cost * (self.replicas - 1) if self.replicas > 1 else 0

        # Query costs (only if exceeding included quota)
        excess_queries = max(0, self.monthly_queries - self.QUERIES_INCLUDED)
        query_cost = (excess_queries / 1_000_000) * self.COST_PER_MILLION_QUERIES

        total = storage_cost + replica_cost + query_cost

        return {
            'storage_cost': storage_cost,
            'query_cost': query_cost,
            'replica_cost': replica_cost,
            'total_monthly': total,
            'tier': tier,
            'pods_needed': int(np.ceil(self.vectors / self.STANDARD_POD_CAPACITY)) if tier == 'Standard' else 1
        }

    def calculate_break_even(self, alternative_cost: float) -> int:
        """
        Calculate vector count where Pinecone cost equals alternative provider.

        Args:
            alternative_cost: Monthly cost of alternative solution

        Returns:
            Number of vectors at break-even point
        """
        # Binary search for break-even
        low, high = self.FREE_TIER_VECTORS, 100_000_000

        while low < high:
            mid = (low + high) // 2
            temp_estimator = PineconeCostEstimator(mid, self.dimensions,
                                                   self.replicas, self.monthly_queries)
            cost = temp_estimator.estimate_monthly_cost()['total_monthly']

            if cost < alternative_cost:
                low = mid + 1
            else:
                high = mid

        return low

    def cost_per_vector(self) -> float:
        """Calculate cost per vector (USD)."""
        total = self.estimate_monthly_cost()['total_monthly']
        return total / self.vectors if self.vectors > 0 else 0

    def annual_projection(self) -> float:
        """Project annual costs."""
        return self.estimate_monthly_cost()['total_monthly'] * 12


class VectorDBComparison:
    """
    Comparison framework for vector database alternatives.
    Providers: Pinecone, Weaviate, Qdrant, Elasticsearch
    """

    @staticmethod
    def get_provider_features() -> pd.DataFrame:
        """
        Get feature comparison table across providers.

        Returns:
            DataFrame with provider features
        """
        data = {
            'Provider': ['Pinecone', 'Weaviate', 'Qdrant', 'Elasticsearch'],
            'Open Source': ['No', 'Yes', 'Yes', 'Yes (Apache 2.0)'],
            'Managed Cloud': ['Yes', 'Yes', 'Yes', 'Yes'],
            'Self-Host': ['No', 'Yes', 'Yes', 'Yes'],
            'Free Tier': ['100K vectors', 'Sandbox', '1GB storage', 'Limited'],
            'Starting Price': ['$70/mo', '$25/mo', '$25/mo', '$95/mo'],
            'Hybrid Search': ['No', 'Native', 'Sparse+Dense', 'Native (BM25+KNN)'],
            'Language': ['Proprietary', 'Go', 'Rust', 'Java'],
            'Best For': ['Simplicity', 'Flexibility', 'Performance', 'Existing ES users']
        }

        return pd.DataFrame(data)

    @staticmethod
    def estimate_weaviate_cost(vectors: int, monthly_queries: int) -> Dict[str, float]:
        """
        Weaviate cost estimate (illustrative).

        Cloud pricing: ~$25/mo starter; scales with storage/compute
        Self-host: Infrastructure costs only (AWS/GCP/Azure)
        """
        if vectors <= 100_000:
            return {'monthly_cost': 25, 'tier': 'Sandbox/Starter'}
        elif vectors <= 1_000_000:
            return {'monthly_cost': 100, 'tier': 'Professional'}
        else:
            # Scale with vector count
            return {'monthly_cost': 100 + (vectors - 1_000_000) / 10_000 * 1, 'tier': 'Enterprise'}

    @staticmethod
    def estimate_qdrant_cost(vectors: int, monthly_queries: int) -> Dict[str, float]:
        """
        Qdrant cost estimate (illustrative).

        Free tier: 1GB storage
        Paid: ~$25/mo starter; very efficient resource usage (Rust)
        """
        if vectors <= 50_000:  # ~1GB at 1536 dimensions
            return {'monthly_cost': 0, 'tier': 'Free'}
        elif vectors <= 500_000:
            return {'monthly_cost': 25, 'tier': 'Starter'}
        else:
            return {'monthly_cost': 25 + (vectors - 500_000) / 10_000 * 0.5, 'tier': 'Scaled'}

    @staticmethod
    def estimate_elasticsearch_cost(vectors: int, monthly_queries: int) -> Dict[str, float]:
        """
        Elasticsearch cost estimate (illustrative).

        Elastic Cloud: $95/mo starter; scales with storage + compute
        Self-host: Infrastructure + ops overhead
        """
        if vectors <= 100_000:
            return {'monthly_cost': 95, 'tier': 'Standard'}
        else:
            return {'monthly_cost': 95 + (vectors - 100_000) / 10_000 * 2, 'tier': 'Enterprise'}

    @staticmethod
    def self_host_infrastructure_estimate(vectors: int, dimensions: int = 1536) -> Dict[str, float]:
        """
        Estimate self-hosting infrastructure costs (AWS illustrative).

        Factors: Compute (EC2), Storage (EBS), Network (data transfer)
        """
        # Storage estimate: vectors * dimensions * 4 bytes (float32)
        storage_gb = (vectors * dimensions * 4) / (1024 ** 3)

        # EBS storage: $0.10/GB-month
        storage_cost = storage_gb * 0.10

        # Compute: EC2 instance based on scale
        if vectors <= 100_000:
            compute_cost = 50  # t3.medium equivalent
        elif vectors <= 1_000_000:
            compute_cost = 150  # m5.xlarge equivalent
        else:
            compute_cost = 300  # m5.2xlarge or cluster

        # Network (minimal estimate)
        network_cost = 20

        return {
            'compute_monthly': compute_cost,
            'storage_monthly': storage_cost,
            'network_monthly': network_cost,
            'total_infrastructure': compute_cost + storage_cost + network_cost,
            'storage_gb': storage_gb
        }

    @staticmethod
    def generate_comparison_table(vector_scenarios: list) -> pd.DataFrame:
        """
        Generate cost comparison across providers for multiple scenarios.

        Args:
            vector_scenarios: List of vector counts to compare

        Returns:
            DataFrame with cost comparison
        """
        results = []

        for vectors in vector_scenarios:
            pinecone = PineconeCostEstimator(vectors).estimate_monthly_cost()
            weaviate = VectorDBComparison.estimate_weaviate_cost(vectors, 100_000)
            qdrant = VectorDBComparison.estimate_qdrant_cost(vectors, 100_000)
            elasticsearch = VectorDBComparison.estimate_elasticsearch_cost(vectors, 100_000)
            self_host = VectorDBComparison.self_host_infrastructure_estimate(vectors)

            results.append({
                'Vectors': f"{vectors:,}",
                'Pinecone': f"${pinecone['total_monthly']:.2f}",
                'Weaviate': f"${weaviate['monthly_cost']:.2f}",
                'Qdrant': f"${qdrant['monthly_cost']:.2f}",
                'Elasticsearch': f"${elasticsearch['monthly_cost']:.2f}",
                'Self-Host (AWS)': f"${self_host['total_infrastructure']:.2f}"
            })

        return pd.DataFrame(results)


def main():
    """Demo cost calculations."""
    print("=== Pinecone Cost Estimator Demo ===\n")

    # Scenario 1: Small app
    small = PineconeCostEstimator(vectors=500_000, monthly_queries=100_000)
    print(f"Small App (500K vectors):")
    print(f"  Monthly Cost: ${small.estimate_monthly_cost()['total_monthly']:.2f}")
    print(f"  Cost per Vector: ${small.cost_per_vector():.6f}\n")

    # Scenario 2: Medium app with replicas
    medium = PineconeCostEstimator(vectors=5_000_000, replicas=2)
    print(f"Medium App (5M vectors, 2 replicas):")
    print(f"  Monthly Cost: ${medium.estimate_monthly_cost()['total_monthly']:.2f}")
    print(f"  Annual Projection: ${medium.annual_projection():.2f}\n")

    # Provider comparison
    print("=== Provider Comparison ===\n")
    scenarios = [100_000, 500_000, 1_000_000, 5_000_000]
    comparison = VectorDBComparison.generate_comparison_table(scenarios)
    print(comparison.to_string(index=False))

    # Break-even analysis
    print("\n=== Break-Even Analysis ===")
    estimator = PineconeCostEstimator(vectors=1_000_000)
    qdrant_cost = VectorDBComparison.estimate_qdrant_cost(1_000_000, 100_000)['monthly_cost']
    break_even = estimator.calculate_break_even(qdrant_cost)
    print(f"Break-even vs Qdrant ($100/mo): ~{break_even:,} vectors")


if __name__ == "__main__":
    main()
