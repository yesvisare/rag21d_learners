# M2.2 — Prompt Optimization & Model Selection

Production-ready prompt optimization system for RAG applications. Reduce LLM costs by 30-50% through intelligent prompt engineering, token optimization, and model routing.

---

## Purpose

Learn to **reduce RAG LLM costs by 30-50%** through intelligent prompt engineering, token optimization, and model routing **without sacrificing quality**. This module teaches you when and how to optimize prompts, and critically, **when NOT to optimize**.

## Concepts Covered

- **RAG-specific prompt templates** (5 production-tested variants)
- **Token estimation and cost projection** across models
- **Intelligent model routing** based on query complexity
- **Context formatting** and smart document truncation
- **A/B testing framework** for prompt comparison
- **Cost/quality trade-offs** and decision frameworks
- **Common failure modes** and debugging strategies
- **ROI analysis** and break-even calculations

## After Completing

You will be able to:
- Design and test prompt variants that reduce token usage by 30-50%
- Route queries to appropriate models based on complexity and cost constraints
- Measure and project costs at different scales (100 to 100K queries/day)
- Identify when prompt optimization is counterproductive
- Debug the 5 most common prompt optimization failures
- Make data-driven decisions using ROI and decision frameworks

## Context in Track

This is **Module 2.2** in the RAG Production Engineering track:
- M1.x: Built foundational RAG system with vector search and generation
- M2.1: Implemented caching strategies for cost reduction
- **M2.2: Optimize prompts and route models intelligently** ← YOU ARE HERE
- M2.3: Build production monitoring dashboards
- M2.4: Implement error handling and reliability patterns

**Prerequisites:** M2.1 (Caching), working RAG system, OpenAI API access (optional for testing)
**Estimated time:** 60-90 minutes for implementation + practice

---

## Project Structure

```
rag21d_learners/
├── app.py                          # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # MIT License
│
├── src/m2_2_prompt_optimization/   # Core package
│   ├── __init__.py                 # Package exports & learning arc
│   ├── config.py                   # Configuration & pricing
│   ├── module.py                   # Core logic (templates, routing, testing)
│   └── router.py                   # FastAPI routes
│
├── notebooks/                      # Interactive learning
│   └── M2_2_Prompt_Optimization_and_Model_Selection.ipynb
│
├── tests/                          # Test suite
│   ├── test_smoke.py               # FastAPI endpoint tests
│   └── test_prompt_ops.py          # Module unit tests
│
├── data/example/                   # Example data
│   └── example_data.json           # Sample documents & queries
│
└── scripts/                        # Utility scripts
    └── run_local.ps1               # Windows server launcher
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository (if needed)
git clone <repo-url>
cd rag21d_learners

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OPENAI_API_KEY (optional)
# Without API key, everything runs in dry-run mode with estimates
```

### 3. Run the API

**Windows (PowerShell):**
```powershell
.\scripts\run_local.ps1
```

**macOS/Linux or Windows (alternative):**
```bash
# Set PYTHONPATH
export PYTHONPATH=$PWD

# Run server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Windows (CMD - one-liner):**
```cmd
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
```

API will be available at:
- **Swagger docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Root:** http://localhost:8000

### 4. Run the Notebook

```bash
# From project root
jupyter notebook notebooks/M2_2_Prompt_Optimization_and_Model_Selection.ipynb
```

The notebook will automatically detect if you have an API key. Without one, it runs in **DRY RUN mode** using estimates only.

### 5. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_smoke.py      # FastAPI endpoints
pytest tests/test_prompt_ops.py  # Module tests

# Or run module directly
python -m src.m2_2_prompt_optimization.module
```

---

## API Usage

### Example: Route a Query

```bash
curl -X POST "http://localhost:8000/m2_2_prompt_optimization/route" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your return policy?",
    "context": "We allow 30-day returns with receipt.",
    "cost_budget": 0.001
  }'
```

**Response:**
```json
{
  "model": "gpt-3.5-turbo",
  "tier": "FAST",
  "complexity_score": 0,
  "complexity_factors": {"simple_pattern": true},
  "reason": "Simple query - fast model sufficient"
}
```

### Example: Compare Prompt Templates

```bash
curl -X POST "http://localhost:8000/m2_2_prompt_optimization/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "templates": ["baseline_comparison", "cost_optimization"],
    "test_cases": [
      {"question": "What is your return policy?"}
    ],
    "documents": [
      {"content": "Our return policy allows returns within 30 days.", "score": 0.95}
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "template_name": "cost_optimization",
      "avg_input_tokens": 180,
      "avg_output_tokens": 150,
      "avg_total_tokens": 330,
      "avg_cost_per_query": 0.000347
    },
    {
      "template_name": "baseline_comparison",
      "avg_input_tokens": 350,
      "avg_output_tokens": 200,
      "avg_total_tokens": 550,
      "avg_cost_per_query": 0.000495
    }
  ],
  "summary": {
    "baseline_template": "baseline_comparison",
    "best_template": "cost_optimization",
    "max_savings_pct": 30.0
  }
}
```

---

## Components

### 1. RAGPromptLibrary

Five production-tested prompt templates:

- **BASIC_RAG** (350 tokens) - Baseline, comprehensive
- **CONCISE_RAG** (180 tokens) - 50% token reduction, good balance
- **STRUCTURED_RAG** (160 tokens) - Structured output format
- **JSON_RAG** (140 tokens) - JSON output for APIs (60% reduction)
- **SUPPORT_RAG** (170 tokens) - Domain-specific for customer support

```python
from src.m2_2_prompt_optimization import RAGPromptLibrary

template = RAGPromptLibrary.CONCISE_RAG
print(template.system_prompt)
```

### 2. TokenEstimator

Accurate token counting and cost projection:

```python
from src.m2_2_prompt_optimization import TokenEstimator

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

### 3. ModelRouter

Intelligent routing based on query complexity:

```python
from src.m2_2_prompt_optimization import ModelRouter

router = ModelRouter()
decision = router.select_model(
    query="What is your return policy?",
    context=retrieved_docs
)

print(f"Selected: {decision['model']}")
print(f"Reason: {decision['reason']}")
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

### 4. format_context_optimally

Smart document truncation that preserves critical information:

```python
from src.m2_2_prompt_optimization import format_context_optimally

formatted = format_context_optimally(
    documents=[
        {"content": "Doc 1 content...", "score": 0.95},
        {"content": "Doc 2 content...", "score": 0.88},
    ],
    max_tokens=500,
    include_metadata=False
)
```

**Features:**
- Truncates at sentence boundaries (not mid-sentence)
- Preserves highest-scoring documents first
- Adds `[truncated]` indicators
- Respects token limits accurately

### 5. PromptTester

A/B test framework for comparing templates:

```python
from src.m2_2_prompt_optimization import PromptTester, RAGPromptLibrary

tester = PromptTester(dry_run=True)  # No API calls
results = tester.compare_templates(
    templates=[
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
    ],
    test_cases=[{"question": "What's the refund policy?"}],
    context_docs=documents
)
```

---

## CLI Usage

The module can be run directly for quick testing:

```bash
# Run module demo (loads example data and shows all features)
python -m src.m2_2_prompt_optimization.module
```

**Expected output:**
```
M2.2 Prompt Optimization Module
================================================================================

Token counting example:
Text: Our return policy allows customers to return items...
Tokens: 67

Query: What is your return policy?
Selected: gpt-3.5-turbo (score: 0)
Reason: Simple query - fast model sufficient

Running template comparison (DRY RUN)...
[Table showing token/cost comparison]
```

---

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

---

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

---

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

---

## Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/m2_2_prompt_optimization --cov-report=html

# Specific tests
pytest tests/test_smoke.py -v        # FastAPI endpoints
pytest tests/test_prompt_ops.py -v   # Module functionality
```

**Test coverage includes:**
- ✓ Token estimation and cost calculation
- ✓ Prompt template library
- ✓ Model routing logic
- ✓ Context formatting and truncation
- ✓ A/B testing framework (dry-run mode)
- ✓ FastAPI endpoints (/health, /route, /compare)
- ✓ Error handling and edge cases

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution:** Set PYTHONPATH:
```bash
# macOS/Linux
export PYTHONPATH=$PWD

# Windows PowerShell
$env:PYTHONPATH = "$PWD"

# Windows CMD
set PYTHONPATH=%CD%
```

### Issue: "OpenAI API key not found"
**Solution:** Either:
1. Add API key to `.env` file
2. Or run in dry-run mode (uses estimates only)

### Issue: Notebook import errors
**Solution:** Ensure you're running from project root:
```bash
cd /path/to/rag21d_learners
jupyter notebook notebooks/M2_2_Prompt_Optimization_and_Model_Selection.ipynb
```

### Issue: Tests fail with path errors
**Solution:** Run tests from project root:
```bash
cd /path/to/rag21d_learners
pytest tests/
```

---

## Development

### Project slug
Module slug: `m2_2_prompt_optimization`

### Adding new prompt templates

Edit `src/m2_2_prompt_optimization/module.py`:

```python
class RAGPromptLibrary:
    MY_CUSTOM_RAG = PromptTemplate(
        system_prompt="Your system prompt...",
        user_template="Your user template with {context} and {question}",
        tokens_estimate=200,
        use_case="my_custom_use_case"
    )
```

### Extending the API

Add new endpoints in `src/m2_2_prompt_optimization/router.py`:

```python
@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    # Your logic here
    pass
```

---

## Next Steps

1. **Run the notebook** - Work through all 7 sections
2. **Test with your data** - Replace `data/example/example_data.json` with real documents
3. **Measure impact** - Use `PromptTester` to compare templates
4. **Calculate ROI** - Use the decision card to determine if optimization is worth it
5. **Implement monitoring** - Track tokens, cost, and quality metrics
6. **Module M2.3** - Build production monitoring dashboard

---

## Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Token counting with tiktoken](https://github.com/openai/tiktoken)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RAG best practices](https://www.anthropic.com/index/retrieval-augmented-generation)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Key Insight:** Prompt optimization is a tool, not a mandate. Let economics guide your engineering decisions.
