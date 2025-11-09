# M4.2 — Beyond Pinecone Free Tier

**Cost Analysis, Alternatives & Migration Strategy for Vector Databases**

## Learning Arc

This module is part of the RAG21D learning track, positioning you to make informed decisions about vector database providers at production scale. After implementing RAG systems in M1-M3, you now need to understand **total cost of ownership**, **alternative providers**, and **scaling economics**.

**Prerequisites:** M1 (Vector Databases), M2 (Cost Optimization), M3 (Deployment)
**Duration:** 2-3 hours hands-on
**Outcome:** Cost models and decision frameworks for vector DB selection

## 📁 Project Structure

```
rag21d_learners/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Python/Jupyter ignores
├── requirements.txt            # Dependencies (numpy, pandas, PyYAML, pytest)
├── src/
│   └── m4_2_cost_models/
│       ├── __init__.py         # Public API exports
│       └── core/
│           └── estimator.py    # Cost estimation logic
├── configs/
│   └── pricing/
│       └── pinecone_2025-11.yaml  # Pricing configuration
├── notebooks/
│   └── M4_2_Beyond_Pinecone_Free_Tier.ipynb  # Interactive walkthrough
├── tests/
│   └── test_cost_models.py     # Pytest test suite
└── scripts/
    └── run_tests.ps1           # PowerShell test runner
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Launch Interactive Notebook

**Recommended** - Work through the 6-section guided notebook:

```bash
# Option 1: Using PYTHONPATH (preferred)
$env:PYTHONPATH = "$PWD"  # PowerShell
export PYTHONPATH=$PWD    # Bash

jupyter notebook notebooks/M4_2_Beyond_Pinecone_Free_Tier.ipynb

# Option 2: Notebook includes fallback sys.path adjustment
jupyter notebook notebooks/M4_2_Beyond_Pinecone_Free_Tier.ipynb
```

**Notebook Sections:**
1. Pricing Reality Check - Cost drivers and tier structure
2. Cost Estimator Walkthrough - Calculate your scenarios
3. Provider Comparison - Weaviate, Qdrant, Elasticsearch feature matrix
4. Self-Host vs Managed - Decision framework and break-even analysis
5. Decision Cards by Scale - Recommendations from prototype to enterprise
6. Troubleshooting & Hidden Costs - Production failures and mitigations

### Programmatic Usage

```python
from m4_2_cost_models import PineconeCostEstimator, VectorDBComparison

# Estimate Pinecone costs
estimator = PineconeCostEstimator(vectors=1_000_000, replicas=2)
result = estimator.estimate_monthly_cost()
print(f"Monthly cost: ${result['total_monthly']:.2f}")

# Compare providers across scales
comparison = VectorDBComparison.generate_comparison_table([100_000, 1_000_000, 5_000_000])
print(comparison)

# Break-even analysis
break_even = estimator.calculate_break_even(alternative_cost=100)
print(f"Break-even at {break_even:,} vectors")
```

## Pricing Model

**Version:** `pinecone_2025-11`
**Effective Date:** 2025-11-01
**Configuration:** `configs/pricing/pinecone_2025-11.yaml`

Pricing data is loaded from YAML at runtime, with safe fallbacks if the file is missing. This allows:
- **Version locking** for reproducible cost analysis
- **Easy updates** when provider pricing changes
- **Comparison** across historical pricing models

**Current Tiers (Illustrative):**
- **Free:** 100K vectors, $0/mo
- **Starter:** Up to 1M vectors, $70/mo
- **Standard:** $280 per 1M vectors (multi-pod)
- **Query Overage:** $5 per 1M queries beyond 10M included

⚠️ **Note:** These are illustrative prices for educational purposes. Always verify current pricing at [pinecone.io/pricing](https://www.pinecone.io/pricing/).

### Rounding & Currency

- **Currency:** USD (United States Dollars)
- **Rounding Policy:** Standard mathematical rounding (0.5 rounds up)
- **Pod Calculations:** `ceil(vectors / capacity)` for Standard tier
- **Precision:** Costs reported to 2 decimal places; per-vector costs to 6 decimals

## 🧪 Testing

Run the full test suite using the PowerShell script:

```powershell
./scripts/run_tests.ps1
```

**Or manually with pytest:**

```bash
# Set PYTHONPATH
$env:PYTHONPATH = "$PWD"  # PowerShell
export PYTHONPATH=$PWD    # Bash

# Run pytest
pytest -q
pytest -v                  # Verbose output
pytest tests/test_cost_models.py::test_tier_boundaries -v  # Specific test
```

**Test Coverage:**
- Tier boundary conditions (Free/Starter/Standard transitions)
- Replica cost calculations
- Query overage pricing
- Break-even analysis logic
- Provider comparison table generation
- Rounding policy verification
- Annual vs monthly projection consistency

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

MIT License - See [LICENSE](LICENSE) file for details.

Educational material for RAG21D learners. Pricing data is illustrative and should be verified with providers.

---

**Questions?** Open an issue or refer to `notebooks/M4_2_Beyond_Pinecone_Free_Tier.ipynb` for detailed walkthroughs.
