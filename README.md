# M4.1 — Hybrid Search Implementation

Combines BM25 sparse retrieval with dense vector embeddings using alpha weighting and Reciprocal Rank Fusion (RRF).

## Learning Arc

### Purpose
This module teaches you to build production-grade hybrid search systems that combine the precision of keyword matching (BM25) with the semantic understanding of dense embeddings. You'll learn when hybrid search justifies its complexity overhead and when simpler approaches suffice.

### Concepts Covered
- **BM25 Sparse Retrieval**: Term-frequency-based ranking for exact matches
- **Dense Vector Embeddings**: Neural network-powered semantic search
- **Alpha Weighting**: Balancing sparse and dense scores with tunable weights
- **Reciprocal Rank Fusion (RRF)**: Rank-based merging without score normalization
- **Smart Query Analysis**: Auto-detecting query types to adjust search strategy
- **Cost-Benefit Analysis**: Understanding latency, complexity, and accuracy trade-offs

### After Completing
You will be able to:
- ✅ Implement hybrid search with configurable merge strategies
- ✅ Tune alpha weights based on query characteristics
- ✅ Decide when hybrid search adds value vs overhead
- ✅ Debug common issues (score normalization, namespace mismatches, etc.)
- ✅ Estimate costs and latency for production deployments
- ✅ Choose between alpha weighting and RRF for your use case

### Context in Track
- **Prerequisite**: M1 (Vector Databases), M2 (Cost Optimization)
- **Builds on**: Dense embeddings from M1.4, cost awareness from M2.1
- **Prepares for**: M4.2 (Beyond Free Tier), M4.3 (Portfolio Projects)
- **Real-world application**: E-commerce search, technical documentation, mixed-content platforms

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
    ┌───────▼────────┐       ┌───────▼────────┐
    │  BM25 (Sparse) │       │ Dense (Pinecone)│
    │  In-memory     │       │ OpenAI Embeddings│
    │  <1ms latency  │       │ 50-100ms latency│
    └───────┬────────┘       └───────┬─────────┘
            │                        │
            │  Top K*2 results       │  Top K*2 results
            │                        │
            └────────┬───────────────┘
                     │
            ┌────────▼────────┐
            │  Merge Strategy │
            │  • Alpha Weight │
            │  • RRF          │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │  Top K Results  │
            └─────────────────┘
```

## Components

### 1. BM25 Sparse Retrieval
- **Algorithm**: Okapi BM25 with NLTK tokenization
- **Strengths**: Exact matches, technical terms, SKUs, IDs
- **Weaknesses**: No semantic understanding, synonym gaps
- **Storage**: In-memory (viable for <100K docs)

### 2. Dense Vector Embeddings
- **Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Strengths**: Semantic understanding, synonyms, context
- **Weaknesses**: May miss exact terms, higher latency
- **Storage**: Pinecone vector database

### 3. Merge Strategies

#### Alpha Weighting
```python
combined_score = alpha * dense_score + (1 - alpha) * sparse_score
```

**Alpha values:**
- `1.0`: Pure dense (semantic only)
- `0.7`: Favor dense (recommended for natural language)
- `0.5`: Equal weighting (balanced)
- `0.3`: Favor sparse (recommended for technical/SKU queries)
- `0.0`: Pure sparse (BM25 only)

#### Reciprocal Rank Fusion (RRF)
```python
rrf_score = sum(1 / (k + rank + 1))  # k typically 60
```

**Advantages:**
- No score normalization needed
- More robust to scale differences
- Less tuning required
- Documents in both lists get boosted

### 4. Smart Alpha Detection

The `smart_alpha()` function dynamically adjusts alpha based on query characteristics:

```python
# Detects:
- SKU patterns (ABC-123, etc.) → alpha = 0.3 (favor sparse)
- Long numeric codes → alpha = 0.3 (favor sparse)
- Exact phrase quotes → alpha = 0.3 (favor sparse)
- High technical term ratio → alpha = 0.5 (balanced)
- Natural language → alpha = 0.7 (favor dense)
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file (see `.env.example`):

```bash
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_INDEX_NAME=hybrid-search
```

## Usage

### Basic Example

```python
from src.m4_1_hybrid_search import HybridSearchEngine

# Initialize
engine = HybridSearchEngine(
    openai_api_key="sk-...",
    pinecone_api_key="...",
    index_name="hybrid-search",
    namespace="default"
)

# Add documents
documents = [
    {
        "id": "doc1",
        "text": "Python is a programming language",
        "metadata": {"category": "tech"}
    },
    # ...
]
engine.add_documents(documents)
engine.upsert_to_pinecone()

# Search with auto-alpha
results = engine.hybrid_search_alpha("python programming", top_k=5)

# Search with fixed alpha
results = engine.hybrid_search_alpha("SKU-12345", top_k=5, alpha=0.3)

# Search with RRF
results = engine.hybrid_search_rrf("machine learning", top_k=5)
```

### With Metadata Filtering

```python
# Filter by category
results = engine.hybrid_search_alpha(
    query="python tutorials",
    top_k=5,
    metadata_filter={"category": "programming"}
)
```

### Using Namespaces

```python
# Separate indexes by tenant/use-case
engine_docs = HybridSearchEngine(..., namespace="documentation")
engine_products = HybridSearchEngine(..., namespace="products")
```

## When to Use Hybrid Search

### ✅ Use When:
- **Mixed query types**: Natural language + technical terms/codes
- **Product catalogs**: SKUs, model numbers, part IDs critical
- **Accuracy critical**: Need 40-60% improvement on exact matches
- **False positive reduction**: Pure semantic search too broad
- **Corpus size**: 1,000+ documents
- **Query diversity**: 40-60% split between keyword and semantic

### ❌ Avoid When:
- **Small corpus**: < 1,000 documents (overhead not justified)
- **Tight latency**: P99 < 50ms required (hybrid adds 80-120ms)
- **Pure conversational**: 90%+ natural language queries (dense-only better)
- **Limited resources**: Maintaining 2 indexes adds complexity
- **Simple use case**: Single search paradigm sufficient

## Performance Characteristics

### Latency Breakdown
```
Dense-only:     50-80ms
+ BM25:         +5-10ms
+ Merge:        +10-20ms
+ Network var:  +20-30ms
─────────────────────────
Total Hybrid:   85-140ms
```

### Cost (per 1,000 queries)
```
Embeddings:  1000 × $0.0001 = $0.10
Pinecone:    ~$0.05
BM25:        $0.00 (in-memory)
─────────────────────────────────
Total:       ~$0.15
```

### Scale Limits
- **In-memory BM25**: < 100K documents
- **Beyond 100K**: Requires Elasticsearch/OpenSearch ($150-500/month)
- **Pinecone**: Scales to millions (cost increases with usage)

## Alpha vs RRF Guidance

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Production system | RRF | Less tuning, more robust |
| Need interpretability | Alpha | Clear weight control |
| Query diversity high | RRF | Handles edge cases better |
| Specific use case | Alpha | Fine-tune per query type |
| Prototyping | Alpha 0.5 | Quick baseline |

## Common Issues

### BM25 scores dominating
- **Cause**: Score normalization issue
- **Fix**: Ensure `max(scores) > 0` check in normalization

### Dense search returns nothing
- **Cause**: Namespace mismatch or vectors not uploaded
- **Fix**: Check `index.describe_index_stats()` for namespace vectors

### RRF same as BM25
- **Cause**: Dense search failing silently
- **Fix**: Wrap dense search in try-catch, verify API keys

### High latency
- **Cause**: Sequential search execution
- **Fix**: Parallelize dense + sparse with `asyncio.gather()`

## Development Time Estimates

- **Basic implementation**: 4-6 hours
- **Production-ready**: 12-16 hours
- **With monitoring**: 20-24 hours
- **Scale optimization**: +8-12 hours

## Testing

Run smoke tests:
```bash
# On Windows PowerShell
powershell ./scripts/run_tests.ps1

# On Linux/Mac
PYTHONPATH=$PWD pytest
```

Run notebook demos:
```bash
jupyter notebook notebooks/M4_1_Hybrid_Search.ipynb
```

## Files

```
.
├── src/
│   └── m4_1_hybrid_search/
│       └── __init__.py          # HybridSearchEngine class & core logic
├── notebooks/
│   └── M4_1_Hybrid_Search.ipynb # Interactive tutorial (7 sections)
├── tests/
│   ├── __init__.py
│   └── test_hybrid_merge.py     # Smoke tests
├── scripts/
│   └── run_tests.ps1            # PowerShell test runner
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Limits & Costs

### OpenAI
- **Rate limit**: 3,000 RPM (tier 1)
- **Cost**: $0.0001 per 1K tokens (text-embedding-3-small)
- **Max input**: 8,191 tokens

### Pinecone
- **Free tier**: 1 index, 100K vectors
- **Starter**: $70/month (5M vectors)
- **Query cost**: Included in tier pricing

### BM25 (In-memory)
- **RAM usage**: ~100 bytes per doc (tokenized)
- **100K docs**: ~10 MB RAM
- **1M docs**: ~100 MB RAM (consider external BM25)

## References

- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Hybrid Search](https://docs.pinecone.io/guides/data/understanding-hybrid-search)

## License

MIT

## Contributing

Issues and PRs welcome! Please include tests for new features.
