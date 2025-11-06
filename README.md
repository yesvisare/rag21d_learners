# M4.2 — Beyond Pinecone Free Tier

**Cost Analysis, Alternatives & Migration Strategy for Vector Databases**

This module provides cost estimation tools and decision frameworks for evaluating vector database providers beyond the Pinecone free tier.

## 📁 Contents

- **`m4_2_cost_models.py`** - Cost estimator functions for Pinecone, Weaviate, Qdrant, Elasticsearch
- **`M4_2_Beyond_Pinecone_Free_Tier.ipynb`** - Interactive walkthrough with 6 sections
- **`requirements.txt`** - Dependencies (numpy, pandas)
- **`tests_costs.py`** - Smoke tests for cost models
- **`README.md`** - This file

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Command-Line Cost Estimation

```bash
python m4_2_cost_models.py
```

**Expected Output:**
```
=== Pinecone Cost Estimator Demo ===

Small App (500K vectors):
  Monthly Cost: $70.00
  Cost per Vector: $0.000140

Medium App (5M vectors, 2 replicas):
  Monthly Cost: $2800.00
  Annual Projection: $33600.00

=== Provider Comparison ===
  Vectors  Pinecone  Weaviate  Qdrant  Elasticsearch  Self-Host (AWS)
  100,000     $0.00    $25.00   $0.00         $95.00           $70.00
  ...
```

#### 2. Interactive Notebook

Open `M4_2_Beyond_Pinecone_Free_Tier.ipynb` in Jupyter:

```bash
jupyter notebook M4_2_Beyond_Pinecone_Free_Tier.ipynb
```

**Sections:**
1. Pricing Reality Check - Understand cost drivers
2. Cost Estimator Walkthrough - Calculate your scenarios
3. Provider Comparison - Feature & cost matrices
4. Self-Host vs Managed - Decision framework
5. Decision Cards by Scale - Recommendations by vector count
6. Troubleshooting & Hidden Costs - Production failures & mitigations

#### 3. Programmatic Use

```python
from m4_2_cost_models import PineconeCostEstimator, VectorDBComparison

# Estimate Pinecone costs
estimator = PineconeCostEstimator(vectors=1_000_000, replicas=2)
result = estimator.estimate_monthly_cost()
print(f"Monthly cost: ${result['total_monthly']:.2f}")

# Compare providers
comparison = VectorDBComparison.generate_comparison_table([100_000, 1_000_000])
print(comparison)

# Break-even analysis
break_even = estimator.calculate_break_even(alternative_cost=100)
print(f"Break-even at {break_even:,} vectors")
```

## 📊 Example Scenarios

### Scenario 1: Early-Stage Startup
- **Vectors:** 200K documents
- **Query Volume:** 50K/month
- **Recommendation:** Pinecone Starter ($70/mo) or Qdrant Free
- **Why:** Simple managed service; defer vendor lock-in decisions

### Scenario 2: Growing SaaS
- **Vectors:** 3M documents
- **Query Volume:** 500K/month
- **Recommendation:** Weaviate Cloud ($150/mo) or Qdrant Cloud ($100/mo)
- **Why:** Open-source flexibility; lower cost than Pinecone ($840/mo)

### Scenario 3: Enterprise at Scale
- **Vectors:** 50M documents
- **Query Volume:** 10M/month
- **Recommendation:** Self-hosted Qdrant on AWS (~$800/mo infrastructure)
- **Why:** Pinecone would cost $14,000+/mo; self-hosting saves 94%

## ⚠️ Key Risks

### 1. Vendor Lock-In
**Risk:** Migration to another provider requires significant re-architecture.

**Mitigation:**
- Abstract vector DB behind an interface/adapter layer
- Use standardized query formats where possible
- Maintain data export capabilities

**Example Interface:**
```python
class VectorDBInterface:
    def upsert(self, vectors, metadata): ...
    def query(self, vector, top_k): ...
    def delete(self, ids): ...

# Implement for each provider
class PineconeAdapter(VectorDBInterface): ...
class QdrantAdapter(VectorDBInterface): ...
```

### 2. Latency Costs
**Risk:** Cross-region latency adds 50-200ms per query.

**Mitigation:**
- Co-locate vector DB with application (same region/AZ)
- Use regional deployments for global apps
- Cache frequent queries

### 3. Hidden TCO Elements
**Often Forgotten:**
- **Embedding costs:** $0.10 per 1M tokens (OpenAI)
- **Multiple environments:** Dev/test/staging multiply costs
- **Data transfer:** Egress charges for self-hosted
- **DevOps time:** 20-40 hrs/month for self-hosted ops

**Example TCO Calculation:**
```
Base Pinecone: $280/mo (1M vectors)
+ Embedding generation: $50/mo (new documents)
+ Staging environment: $70/mo (separate index)
+ Dev environment: $0/mo (free tier)
= Total: $400/mo (43% higher than base)
```

### 4. Scale Surprises
**Risk:** Costs scale non-linearly; 10x data ≠ 10x cost.

**Reality:**
- Pinecone: Linear scaling ($280 per 1M vectors)
- Qdrant: Sublinear (efficiency gains at scale)
- Self-hosted: Step functions (instance size jumps)

### 5. Query Volume Underestimation
**Risk:** Exceeding 10M queries/month triggers overage charges.

**Mitigation:**
- Monitor query metrics from day 1
- Implement caching for repeated queries
- Set up alerts at 80% of quota

## 🔑 Decision Framework

### When to Use Pinecone
- ✅ Speed to market critical (fastest setup)
- ✅ No DevOps team
- ✅ Budget <$1K/month
- ❌ High sensitivity to vendor lock-in
- ❌ Need hybrid search (keyword + semantic)

### When to Use Weaviate
- ✅ Need hybrid search (BM25 + semantic)
- ✅ Want open-source flexibility
- ✅ Plan to eventually self-host
- ✅ Multiple vector spaces in single DB
- ❌ Need absolute best performance

### When to Use Qdrant
- ✅ Cost optimization priority
- ✅ Need excellent filtering capabilities
- ✅ Plan to self-host (Rust efficiency)
- ✅ Batch operations important
- ❌ Need mature managed service ecosystem

### When to Self-Host
- ✅ >5M vectors
- ✅ DevOps capacity available
- ✅ Stable, predictable workload
- ✅ Data sovereignty requirements
- ❌ Variable/spiky traffic
- ❌ Small team (<5 engineers)

## 🧪 Testing

Run smoke tests:

```bash
python tests_costs.py
```

**Expected:**
```
Testing PineconeCostEstimator...
✓ Free tier calculation correct
✓ Starter tier calculation correct
✓ Standard tier calculation correct
✓ Replica costs correct
✓ Break-even analysis functional

Testing VectorDBComparison...
✓ Provider features table generated
✓ Cost comparison table generated
✓ Self-host estimates reasonable

All tests passed!
```

## 📈 Next Steps

1. **Assess Current State:**
   - Count your vectors (or project 6-12 months out)
   - Estimate query volume
   - Determine HA requirements (replicas)

2. **Run Cost Models:**
   ```python
   estimator = PineconeCostEstimator(
       vectors=YOUR_COUNT,
       replicas=YOUR_REPLICAS,
       monthly_queries=YOUR_QUERIES
   )
   print(estimator.estimate_monthly_cost())
   ```

3. **Compare Alternatives:**
   - Use notebook Section 3 for feature comparison
   - Generate cost tables for your scenarios
   - Consider total TCO (not just base costs)

4. **Prototype:**
   - Start with free tiers (Pinecone, Qdrant, Weaviate sandbox)
   - Test query performance with your data
   - Validate embedding quality

5. **Plan Migration:**
   - If scaling beyond free tier, decide provider BEFORE hitting limits
   - Implement abstraction layer to reduce lock-in
   - Budget for embedding generation + multiple environments

## 📚 References

- [Pinecone Pricing](https://www.pinecone.io/pricing/)
- [Weaviate Cloud Pricing](https://weaviate.io/pricing)
- [Qdrant Cloud Pricing](https://qdrant.tech/pricing/)
- [Elasticsearch Service Pricing](https://www.elastic.co/pricing/)

## ⚖️ License

Educational material for RAG21D learners. Pricing data is illustrative and should be verified with providers.

---

**Questions?** Open an issue or refer to `M4_2_Beyond_Pinecone_Free_Tier.ipynb` for detailed walkthroughs.
