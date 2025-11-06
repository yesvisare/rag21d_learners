# M2.2 — Prompt Optimization & Model Selection

Production-ready prompt optimization system for RAG applications. Reduce LLM costs by 30-50% through intelligent prompt engineering, token optimization, and model routing.

## Overview

This module teaches you how to optimize RAG prompts without sacrificing quality. You'll learn:

- **RAG-specific prompt templates** optimized for different use cases
- **Token reduction techniques** that preserve critical information
- **Intelligent model routing** to match query complexity with appropriate models
- **A/B testing framework** to measure real impact
- **Cost/quality trade-offs** and when NOT to optimize
- **Common failures** and how to fix them

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the Notebook

```bash
jupyter notebook M2_2_Prompt_Optimization_and_Model_Selection.ipynb
```

The notebook will automatically detect if you have an API key. Without one, it runs in **DRY RUN mode** using estimates only.

### 4. Test the Module

```bash
# Dry run (no API key needed)
python m2_2_prompt_ops.py

# Run tests
python tests_prompt_ops.py
```

## Components

### 1. Prompt Library (`RAGPromptLibrary`)

Five production-tested prompt templates:

- **BASIC_RAG** (350 tokens) - Baseline, comprehensive
- **CONCISE_RAG** (180 tokens) - 50% token reduction, good balance
- **STRUCTURED_RAG** (160 tokens) - Structured output format
- **JSON_RAG** (140 tokens) - JSON output for APIs (60% reduction)
- **SUPPORT_RAG** (170 tokens) - Domain-specific for customer support

```python
from m2_2_prompt_ops import RAGPromptLibrary

template = RAGPromptLibrary.CONCISE_RAG
print(template.system_prompt)
```

### 2. Token Estimator (`TokenEstimator`)

Accurate token counting and cost projection:

```python
from m2_2_prompt_ops import TokenEstimator

estimator = TokenEstimator("gpt-3.5-turbo")
tokens = estimator.count_tokens("Your text here")
cost = estimator.estimate_cost(input_tokens=500, output_tokens=200)

# Project monthly costs
projection = estimator.project_monthly_cost(
    avg_input_tokens=300,
    avg_output_tokens=150,
    queries_per_day=10_000
)
print(f"Monthly cost: ${projection['monthly_cost']:.2f}")
```

### 3. Model Router (`ModelRouter`)

Intelligent routing based on query complexity:

```python
from m2_2_prompt_ops import ModelRouter

router = ModelRouter()
decision = router.select_model(
    query="What is your return policy?",
    context=retrieved_docs
)

print(f"Selected: {decision['model']}")
print(f"Reason: {decision['reason']}")
# Output: Selected: gpt-3.5-turbo, Reason: Simple query - fast model sufficient
```

**Routing Logic:**
- Complexity score 0-2: Fast model (gpt-3.5-turbo)
- Complexity score 3-5: Balanced model (gpt-4o-mini)
- Complexity score 6+: Premium model (gpt-4o)

**Complexity Factors:**
- Query length + reasoning keywords
- Multiple questions
- Large context volume (>1000 words)
- Technical/code content

### 4. Context Formatting (`format_context_optimally`)

Smart document truncation that preserves critical information:

```python
from m2_2_prompt_ops import format_context_optimally

documents = [
    {"content": "Doc 1 content...", "score": 0.95},
    {"content": "Doc 2 content...", "score": 0.88},
]

formatted = format_context_optimally(
    documents,
    max_tokens=500,
    include_metadata=False
)
```

**Features:**
- Truncates at sentence boundaries (not mid-sentence)
- Preserves highest-scoring documents first
- Adds `[truncated]` indicators
- Respects token limits accurately

### 5. Prompt Tester (`PromptTester`)

A/B test framework for comparing templates:

```python
from m2_2_prompt_ops import PromptTester
from openai import OpenAI

client = OpenAI()  # Or None for dry run
tester = PromptTester(client, model="gpt-3.5-turbo")

results = tester.compare_templates(
    templates=[
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
    ],
    test_cases=[
        {"question": "What's the refund policy?", "expected_answer": "..."}
    ],
    context_docs=documents
)

# Export results
tester.export_results(results, "comparison.json")
```

**Metrics Tracked:**
- Average input/output tokens
- Cost per query
- Latency (milliseconds)
- Queries tested

## Decision Card: Should You Optimize?

### ✅ Use When:
- Query volume >1,000/day
- Monthly LLM costs >$100
- Acceptable to trade minor quality for cost savings
- Have monitoring infrastructure
- Team has bandwidth for ongoing tuning (2-4 hours/month)

### ❌ Avoid When:
- **Quality is non-negotiable** (medical, legal, financial)
- **Query volume <100/day** (overhead exceeds savings)
- **Query diversity >90%** (caching ineffective)
- **No monitoring** (can't measure impact)
- **Tight latency requirements <200ms** (optimization adds 50-100ms)

## Cost Projections

| Scale | Queries/Day | Baseline Cost | Optimized Cost | Savings |
|-------|-------------|---------------|----------------|---------|
| Startup | 100 | $6/mo | $4/mo | $2/mo (33%) |
| Growth | 1,000 | $60/mo | $39/mo | $21/mo (35%) |
| Production | 10,000 | $600/mo | $390/mo | $210/mo (35%) |
| Enterprise | 100,000 | $6,000/mo | $3,900/mo | $2,100/mo (35%) |

**ROI:**
- Implementation: 8 hours (~$800 cost)
- Break-even: 1,000 q/day = 3.8 months, 10,000 q/day = 0.4 months

## Common Failures & Fixes

### Failure #1: Token Limit Exceeded
**Cause:** Forgot prompt overhead + safety margin
**Fix:** `actual_limit = model_context - prompt_overhead - safety_margin`

### Failure #2: Wrong Model Selected
**Cause:** Over-weighting query length
**Fix:** Combine length with reasoning keyword detection

### Failure #3: Lost Critical Context
**Cause:** Truncating mid-sentence
**Fix:** Truncate at sentence boundaries, add `[truncated]`

### Failure #4: Cache Invalidation
**Cause:** Prompt hash changes invalidate all caches
**Fix:** Use semantic versioning (v1, v2), not exact hashes

### Failure #5: JSON Parsing Fails
**Cause:** Model ignores "return JSON only"
**Fix:** Use `response_format={"type": "json_object"}`, temperature=0.0

## Alternatives to Prompt Optimization

### When to Consider Other Approaches:

1. **Model Fine-Tuning** (10K+ q/day, domain-specific)
   - Upfront: $500-2000 training cost
   - Savings: 80% at scale
   - Use case: Legal document analysis

2. **Infrastructure Optimization** (Caching-first, M2.1)
   - Upfront: 8-12 hours
   - Savings: 40-50% with cache hits
   - Use case: Repetitive queries

3. **Hybrid Approach** (Recommended for production)
   - Combine caching + prompt optimization + routing
   - Savings: 50-70% total
   - Use case: Multi-tenant SaaS

## Troubleshooting

### Issue: "No module named 'm2_2_prompt_ops'"
**Solution:** Make sure you're running from the correct directory:
```bash
cd /path/to/rag21d_learners
python m2_2_prompt_ops.py
```

### Issue: "OpenAI API key not found"
**Solution:**
1. Copy `.env.example` to `.env`
2. Add your API key: `OPENAI_API_KEY=sk-...`
3. Or run in dry run mode (estimates only)

### Issue: Notebook costs too much
**Solution:** The notebook uses 3 test queries by default. Costs ~$0.01 per run with API calls. Run in dry run mode if concerned.

### Issue: Results seem wrong
**Solution:** In dry run mode, results are estimates only. Run with API key for accurate measurements.

## Files

- `M2_2_Prompt_Optimization_and_Model_Selection.ipynb` - Main interactive notebook
- `m2_2_prompt_ops.py` - Core module with all classes
- `config.py` - Configuration and pricing
- `example_data.json` - Sample documents and queries
- `tests_prompt_ops.py` - Smoke tests
- `requirements.txt` - Dependencies
- `.env.example` - Environment template

## Next Steps

1. **Run the notebook** - Work through all 7 sections
2. **Test with your data** - Replace `example_data.json` with real documents
3. **Measure impact** - Use `PromptTester` to compare templates
4. **Calculate ROI** - Use the decision card to determine if optimization is worth it
5. **Implement monitoring** - Track tokens, cost, and quality metrics
6. **Module M2.3** - Build production monitoring dashboard

## Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Token counting with tiktoken](https://github.com/openai/tiktoken)
- [RAG best practices](https://www.anthropic.com/index/retrieval-augmented-generation)

## License

Educational purposes only. Part of RAG 21-Day Learners course.

---

**Key Insight:** Prompt optimization is a tool, not a mandate. Let economics guide your engineering decisions.
