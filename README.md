# M1.4 — Query Pipeline & Response Generation

**Complete 7-Stage RAG Pipeline:**
`Query → Retrieval → Rerank → Context → LLM → Answer`

This module implements a production-ready RAG query pipeline with hybrid search (dense + sparse), cross-encoder reranking, query-type specific optimization, and comprehensive metrics tracking.

---

## 🎯 Overview

Transform user queries into grounded, cited answers through seven sequential stages:

1. **Query Understanding** — Classify type, expand variants, extract keywords
2. **Retrieval Strategy** — Hybrid dense+sparse search with auto-tuned alpha
3. **Reranking** — Cross-encoder refinement for 10-20% better relevance
4. **Context Preparation** — Dedup, source attribution, length limits
5. **Prompt Engineering** — Query-type specific templates
6. **Response Generation** — LLM answer (streaming or batch)
7. **Metadata Collection** — Timings, sources, scores

**Key Benefits:**
- Reduces hallucination by 60-80%
- Handles 1,000+ documents efficiently
- Provides source attribution for trust
- Adapts to six query types automatically

---

## 📊 Pipeline Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Query Understanding │ ← Classify (6 types), Expand, Extract Keywords
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Hybrid Retrieval    │ ← Dense (semantic) + Sparse (BM25), Alpha auto-tune
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Cross-Encoder       │ ← Rerank with ms-marco-MiniLM-L-6-v2
│ Reranking           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Context Building    │ ← Dedup, Source tags, Max length (4000 chars)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Prompt Engineering  │ ← Type-specific templates (Factual, How-to, etc.)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ LLM Generation      │ ← GPT-4o-mini (streaming or batch)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Response + Metadata │ ← Answer, Sources, Timings, Scores
└─────────────────────┘
```

---

## 🚀 Quickstart

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   PINECONE_API_KEY=pc-...
#   PINECONE_REGION=us-east-1
```

### 2. Run Notebook

```bash
jupyter notebook M1_4_Query_Pipeline_and_Response.ipynb
```

The notebook walks through all 9 sections incrementally:
1. Reality Check (capabilities vs limitations)
2. Query Understanding (classification, expansion, keywords)
3. Retrieval Strategies (hybrid search, alpha tuning)
4. Reranking (cross-encoder scoring)
5. Context Preparation (dedup, sources, limits)
6. Prompt Engineering (type-specific templates)
7. Response Generation (streaming vs batch)
8. Complete Pipeline (end-to-end with metrics)
9. Failures & Decision Card (5 common issues + when to use RAG)

### 3. CLI Usage

```bash
# Basic query (non-streaming)
python m1_4_query_pipeline.py --ask "How do I improve RAG accuracy?" --top_k 5 --rerank 1

# Streaming query
python m1_4_query_pipeline.py --stream "Optimize Pinecone index" --top_k 5

# Without reranking (faster)
python m1_4_query_pipeline.py --ask "What is RAG?" --top_k 3 --rerank 0

# With query expansion (requires OpenAI)
python m1_4_query_pipeline.py --ask "Fix latency issues" --top_k 5 --expand 1 --rerank 1
```

**CLI Flags:**
- `--ask` — Query for non-streaming response
- `--stream` — Query for streaming response (token-by-token)
- `--top_k` — Number of initial retrieval results (default: 5)
- `--rerank` — Enable reranking: 1=yes, 0=no (default: 1)
- `--expand` — Enable query expansion: 1=yes, 0=no (default: 0)
- `--namespace` — Pinecone namespace (default: "demo")

### 4. Run Tests

```bash
python tests_query_pipeline.py
```

Smoke tests verify:
- Query classification returns valid enum
- Alpha selector returns floats in [0, 1]
- Context builder returns required fields
- Reranker processes results correctly
- Graceful fallbacks when API keys missing

---

## ⚙️ Reranking: With vs Without

**Without Reranking (Faster):**
- Latency: ~200-300ms
- Uses only initial bi-encoder scores (Pinecone + OpenAI embeddings)
- Good for: High-volume, latency-sensitive apps

**With Reranking (Better Quality):**
- Latency: ~300-400ms (+50-100ms)
- Applies cross-encoder for deeper query-document scoring
- Improves relevance by 10-20%
- Good for: Accuracy-critical apps, complex queries

**How to Toggle:**
```python
# Enable reranking
rag = ProductionRAG(use_reranking=True)

# Disable reranking
rag = ProductionRAG(use_reranking=False)
```

---

## 💰 Costs & Latency

### Cost Breakdown (per 1000 queries)

| Component | Cost | Notes |
|-----------|------|-------|
| Embeddings | $0.10 | OpenAI text-embedding-3-small (~1K tokens/query) |
| Pinecone | $0.00 | Included in monthly tier ($70/month for 100K vectors) |
| LLM Generation | $0.50 | GPT-4o-mini (~1K tokens/response) |
| Reranker | $0.00 | Local cross-encoder (free) |
| **Total** | **~$0.60** | **$0.0006 per query** |

### Latency Breakdown (typical)

| Stage | Time | Notes |
|-------|------|-------|
| Query Processing | 5-10ms | Classification, keywords |
| Retrieval | 50-150ms | Embedding + Pinecone query |
| Reranking | 50-100ms | Cross-encoder scoring |
| Context Building | 5-10ms | Dedup, formatting |
| LLM Generation | 500-2000ms | Depends on response length |
| **Total (no rerank)** | **~600-2200ms** | **0.6-2.2 seconds** |
| **Total (with rerank)** | **~650-2300ms** | **0.65-2.3 seconds** |

**Optimization Tips:**
- Cache frequent queries → Saves 100% of cost/latency
- Use smaller `top_k` (3 vs 5) → Saves ~20ms retrieval
- Disable query expansion → Saves ~200ms (1 LLM call)
- Lower `max_tokens` (300 vs 500) → Saves ~200ms generation

---

## 🔧 Common Failures & Fallbacks

### 1. Empty Retrieval Results
**Cause:** Query too specific, no matching content
**Fallback:** Return "No relevant information found" + suggest refinement
**Prevention:** Monitor hit rates, add default content

### 2. API Timeout (Pinecone/OpenAI)
**Cause:** Network issues, rate limits
**Fallback:** Retry with exponential backoff (3 attempts)
**Prevention:** Set timeouts (2s retrieval, 10s generation), circuit breakers

### 3. Context Overflow
**Cause:** Retrieved chunks exceed token budget
**Fallback:** Truncate to `max_length`, prioritize top scores
**Prevention:** Smart chunking (512 tokens/chunk), strict limits

### 4. LLM Hallucination Despite Context
**Cause:** Model extrapolates beyond information
**Fallback:** Stricter "context-only" prompt, lower temperature
**Prevention:** Temperature 0.1, post-filter responses

### 5. Reranker Model Load Failure
**Cause:** Missing model file, memory limits
**Fallback:** Skip reranking, return initial results
**Prevention:** Pre-load at startup, monitor memory

---

## 🎯 Decision Card: When to Use RAG

### ✅ Use RAG When:
- Content changes frequently (docs, FAQs, knowledge bases)
- Need source attribution for compliance/trust
- Query diversity is high (can't pre-generate answers)
- Domain knowledge exceeds LLM training cutoff
- Handling 100+ documents with evolving content

### ❌ Don't Use RAG When:
- Answers are static and finite → Use pre-generated cache
- Real-time latency critical (<100ms) → Use lookup table
- Content fits in single prompt (<4K tokens) → Direct LLM call
- Queries are highly repetitive → Use cache/CDN
- No infrastructure for vector DB + embeddings

### ⚖️ Latency Budget Check:
- **Acceptable:** 200-400ms for conversational AI, internal tools
- **Too Slow:** Real-time autocomplete, sub-100ms APIs
- **Solution if too slow:** Cache frequent queries, disable reranking

---

## 📁 Project Structure

```
rag21d_learners/
├── config.py                          # Environment config, client initialization
├── m1_4_query_pipeline.py             # All 7 pipeline components + CLI
├── M1_4_Query_Pipeline_and_Response.ipynb  # Interactive tutorial (9 sections)
├── requirements.txt                   # Dependencies (OpenAI, Pinecone, transformers)
├── tests_query_pipeline.py            # Smoke tests
├── example_data.txt                   # 20 sample docs for demos
├── .env.example                       # API key template
└── README.md                          # This file
```

---

## 🔗 Links

- **Previous Module:** [M1.3 — Indexing & Retrieval Strategies](../M1_3) (vector DB setup, hybrid search)
- **Reference Document:** [augmented_M1_VideoM1.4_QueryPipeline&ResponseGeneration.md](https://github.com/yesvisare/rag21d_learners/blob/main/augmented_M1_VideoM1.4_QueryPipeline%26ResponseGeneration.md)
- **Pinecone Docs:** [Hybrid Search](https://docs.pinecone.io/docs/hybrid-search)
- **OpenAI Embeddings:** [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- **Cross-Encoder:** [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

---

## 📝 Key Takeaways

1. **Hybrid Search > Dense-Only:** Combines semantic (dense) + keyword (sparse) for 15-25% better recall
2. **Query Type Matters:** Auto-tune alpha (0.3-0.8) based on query classification
3. **Reranking Pays Off:** 10-20% relevance gain for 50-100ms latency cost
4. **Source Attribution:** Builds trust; essential for compliance/verification
5. **Graceful Degradation:** Handle missing keys, empty results, API failures without crashes

**Trade-offs to Remember:**
- Accuracy vs Latency (reranking adds 50-100ms)
- Coverage vs Precision (more chunks = more noise)
- Cost vs Quality (query expansion adds 1 LLM call)

---

**Built with:** OpenAI, Pinecone, Sentence-Transformers
**License:** MIT (for educational purposes)
**Module:** M1.4 of RAG21D Learner Series
