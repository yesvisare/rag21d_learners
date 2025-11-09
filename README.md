# M2.3 — Production Monitoring Dashboard

## Purpose

Implement production-grade monitoring for RAG (Retrieval-Augmented Generation) systems using Prometheus and Grafana. Learn to track performance metrics (latency, token usage, costs), emit structured logs, build dashboards, and set up intelligent alerting for real-world RAG deployments.

## Concepts Covered

- **Prometheus metric types** (Counter, Histogram, Gauge, Summary)
- **Time-series data collection** and percentile calculations (p50/p95/p99)
- **Structured JSON logging** for cloud environments (CloudWatch, Stackdriver, Datadog)
- **Cost tracking and budget alerting** for LLM APIs
- **Cache hit rate optimization** and performance tuning
- **Quality metrics** (relevance scores, context precision)
- **Grafana dashboard design** and PromQL queries
- **Alert rule configuration** and avoiding alert fatigue
- **Safe label cardinality management** to prevent metric explosions
- **Trade-offs**: self-hosted vs managed APM solutions

## After Completing

You will be able to:

- ✅ Instrument RAG pipelines with comprehensive metrics
- ✅ Design effective Grafana dashboards for production monitoring
- ✅ Set up proactive alerts for latency, errors, costs, and quality degradation
- ✅ Emit structured logs for debugging and audit trails
- ✅ Calculate and track per-query costs across multiple LLM models
- ✅ Optimize caching strategies using hit rate metrics
- ✅ Identify and prevent common monitoring pitfalls
- ✅ Decide when self-hosted monitoring is appropriate vs managed alternatives

## Context in Track

This module builds on **M2.1 (Semantic Caching)** and **M2.2 (Prompt Compression)** by adding observability to production RAG systems. It provides the foundation for **M2.4 (A/B Testing)** and **M2.5 (Cost Optimization)** by establishing metrics collection infrastructure. The monitoring patterns learned here apply to any production ML/LLM system, not just RAG.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (metrics port, service name, cost rates)
```

### 3. Start the FastAPI Application

**Using PowerShell script:**
```powershell
.\scripts\run_local.ps1
```

**Or manually:**
```bash
export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### 4. Access the Services

- **API Documentation**: http://localhost:8001/docs
- **API Root**: http://localhost:8001
- **Prometheus Metrics**: http://localhost:8000/metrics

### 5. Start Monitoring Stack (Prometheus + Grafana)

```bash
cd docker
docker compose -f docker-compose.monitoring.yml up -d
```

Services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 6. Import Grafana Dashboard

1. Navigate to http://localhost:3000
2. Log in (admin/admin)
3. Click **+** → **Import dashboard**
4. Upload `grafana/grafana_dash.json`
5. Select Prometheus as the data source
6. Click **Import**

---

## Project Structure

```
.
├── README.md                      # This file
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore patterns
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
├── app.py                         # FastAPI application entry point
│
├── src/
│   └── m2_3_monitoring/          # Main monitoring module
│       ├── __init__.py           # Module exports & learning arc
│       ├── config.py             # Configuration settings
│       ├── module.py             # Core metrics & logging classes
│       └── router.py             # FastAPI routes
│
├── tests/
│   ├── __init__.py
│   ├── test_monitoring.py        # Core module tests
│   └── test_smoke.py             # FastAPI endpoint tests
│
├── notebooks/
│   └── M2_3_Production_Monitoring_Dashboard.ipynb  # Interactive tutorial
│
├── docker/
│   └── docker-compose.monitoring.yml  # Prometheus + Grafana stack
│
├── infra/
│   └── prometheus.yml            # Prometheus scrape configuration
│
├── grafana/
│   └── grafana_dash.json         # Pre-built dashboard (10 panels)
│
└── scripts/
    └── run_local.ps1             # PowerShell dev script
```

---

## API Endpoints

### Health & Info

- **`GET /`** - Root endpoint with service information
- **`GET /health`** - Simple health check
- **`GET /monitoring/health`** - Monitoring module health check

### Cost Estimation

- **`POST /monitoring/cost/estimate`** - Estimate query costs

```json
{
  "input_tokens": 1000,
  "output_tokens": 500,
  "model": "gpt-4"
}
```

Response includes cost breakdown and per-1K-token rates.

### Demo Simulation

- **`POST /monitoring/simulate`** - Generate sample metrics for testing

```json
{
  "count": 5,
  "operation": "demo_api",
  "model": "gpt-4"
}
```

Simulates RAG queries with realistic latencies, token usage, and cache hits.

### Metrics Info

- **`GET /monitoring/metrics-info`** - List all registered Prometheus metrics

---

## Usage Examples

### Basic Instrumentation

Use the `@monitored_query` decorator to automatically track metrics:

```python
from src.m2_3_monitoring import monitored_query

@monitored_query(operation="rag_search", model="gpt-4")
def my_rag_function(query: str):
    # Your RAG logic here
    return {
        'answer': 'Generated answer',
        'input_tokens': 500,
        'output_tokens': 150,
        'relevance_score': 0.85
    }
```

### Manual Metric Recording

```python
from src.m2_3_monitoring import metrics, track_cache_operation

# Record latency
with metrics.query_latency.labels(operation="search").time():
    # Your code here
    pass

# Track cache operations
track_cache_operation(hit=True, cache_type="semantic")

# Calculate costs
cost = metrics.calculate_cost(
    input_tokens=500,
    output_tokens=150,
    model="gpt-4"
)
```

### Structured Logging

```python
from src.m2_3_monitoring import StructuredLogger

logger = StructuredLogger(__name__)

# Log request
logger.log_request(
    query="What are the sales figures?",
    user_id="alice",
    session_id="sess_123"
)

# Log response
logger.log_response(
    duration_ms=1250.5,
    tokens={'input': 450, 'output': 180},
    cost=0.0234,
    success=True
)

# Log error
try:
    raise ValueError("API error")
except Exception as e:
    logger.log_error(e, context={'endpoint': '/api/query'})
```

### Running as Module

```bash
python -m src.m2_3_monitoring.module
```

This starts a demo that generates sample metrics.

---

## Dashboard Panels

The Grafana dashboard includes **10 visualization panels**:

1. **Query Latency (p50/p95/p99)** - Track performance trends
2. **Request Rate & Error Rate** - Monitor traffic and failures
3. **Token Usage (Input vs Output)** - Understand token consumption
4. **Cost per Query & Total Spend** - Control spending
5. **Cache Hit Rate %** - Optimize caching strategy
6. **Error Rate %** - Set SLA alerts
7. **Active Requests** - Detect traffic spikes
8. **Rate Limit Remaining** - Prevent rate limit hits
9. **Response Relevance Score** - Monitor answer quality
10. **Error Breakdown by Type** - Debug error patterns

---

## Alert Templates

Set up these **5 critical alerts** in Prometheus Alertmanager or Grafana:

### 1. High Latency
```yaml
alert: HighQueryLatency
expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 2
for: 5m
annotations:
  summary: "p95 latency exceeds 2 seconds"
```

### 2. High Error Rate
```yaml
alert: HighErrorRate
expr: rate(rag_requests_total{status="error"}[5m]) / rate(rag_requests_total[5m]) > 0.05
for: 2m
annotations:
  summary: "Error rate above 5%"
```

### 3. Low Cache Hit Rate
```yaml
alert: LowCacheHitRate
expr: rag_cache_hit_rate < 0.30
for: 10m
annotations:
  summary: "Cache hit rate below 30%"
```

### 4. High Cost Burn
```yaml
alert: HighCostBurn
expr: sum(rate(rag_total_cost_usd[1h])) * 24 > 100
for: 15m
annotations:
  summary: "Daily spend projected to exceed $100"
```

### 5. Rate Limit Warning
```yaml
alert: RateLimitWarning
expr: rag_rate_limit_remaining < 100
for: 1m
annotations:
  summary: "Less than 100 API calls remaining"
```

---

## Recording Rules (Performance Optimization)

To reduce dashboard query load, create **recording rules** in Prometheus:

```yaml
# prometheus-rules.yml
groups:
  - name: rag_recording_rules
    interval: 30s
    rules:
      # Pre-compute p95 latency
      - record: rag:query_latency:p95
        expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))

      # Pre-compute error rate
      - record: rag:error_rate:5m
        expr: rate(rag_requests_total{status="error"}[5m]) / rate(rag_requests_total[5m])

      # Pre-compute hourly cost
      - record: rag:cost_per_hour:sum
        expr: sum(rate(rag_total_cost_usd[1h])) by (model) * 3600
```

Reference these in Grafana dashboards instead of complex queries.

---

## Safe Labels Checklist

**❌ NEVER use as labels** (unbounded cardinality):
- `user_id` or `session_id`
- Timestamps or request IDs
- Full query text or response text
- IP addresses

**✅ ALWAYS use as labels** (bounded cardinality):
- `operation` (e.g., "search", "summarize")
- `model` (e.g., "gpt-4", "claude-2")
- `error_type` (e.g., "ValueError", "TimeoutError")
- `cache_type` (e.g., "semantic", "exact")
- `status` (e.g., "success", "error")

**Rule of thumb**: Keep unique label combinations < 1,000 per metric.

---

## Common Failure Modes

### 1. Metric Cardinality Explosion
**Problem:** Using user IDs or query text as labels creates millions of unique metrics
**Solution:** Use bounded labels and aggregate high-cardinality data in logs

### 2. Dashboard Query Timeouts
**Problem:** Complex PromQL queries on large time ranges
**Solution:** Use recording rules to pre-compute expensive aggregations

### 3. Memory Leaks from Unbounded Labels
**Problem:** Labels with infinite possible values (timestamps, UUIDs)
**Solution:** Validate label cardinality < 1,000 per metric

### 4. Alert Fatigue
**Problem:** Too many alerts or poorly calibrated thresholds
**Solution:** Start with 3-5 critical alerts, tune thresholds over 1 week of real traffic

### 5. Metric Gaps After Deployment
**Problem:** Metrics server restarts lose in-memory state
**Solution:** Use Prometheus federation or remote storage for persistence

---

## Testing

Run unit tests:
```bash
pytest tests/test_monitoring.py -v
```

Run FastAPI endpoint tests:
```bash
pytest tests/test_smoke.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

---

## When NOT to Use This

Self-hosted Prometheus/Grafana is **overkill** if:

- ❌ Processing < 500 queries/day → Use CloudWatch/Stackdriver logs
- ❌ Team has no DevOps experience → Use Datadog/New Relic
- ❌ Need results in < 2 hours → Start with basic logging first

**Cost comparison:**

| Approach | Setup Time | Monthly Cost | Best For |
|----------|------------|--------------|----------|
| Self-Hosted (Prometheus) | 8-12 hours | $50-200 | >1K queries/day with technical teams |
| Managed APM (Datadog) | 30 minutes | $15-31/host | Non-technical teams, instant dashboards |
| Native Cloud Logging | 1 hour | $5-20 | <500 queries/day, minimal setup |

---

## Configuration

Edit `config.py` or `.env` to customize:

- **METRICS_PORT**: Port for Prometheus metrics endpoint (default: 8000)
- **SERVICE_NAME**: Name for your service (default: rag-service)
- **ENV**: Environment (development/staging/production)
- **COST_PER_1K_INPUT_TOKENS**: Cost per 1K input tokens (default: $0.003)
- **COST_PER_1K_OUTPUT_TOKENS**: Cost per 1K output tokens (default: $0.015)
- **LATENCY_P95_THRESHOLD_MS**: Alert threshold for p95 latency (default: 2000ms)
- **ERROR_RATE_THRESHOLD**: Alert threshold for error rate (default: 0.05 = 5%)
- **CACHE_HIT_MIN_THRESHOLD**: Minimum acceptable cache hit rate (default: 0.30 = 30%)

---

## Resources

- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PromQL Guide**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Best Practices**: https://prometheus.io/docs/practices/naming/
- **Notebook Tutorial**: `notebooks/M2_3_Production_Monitoring_Dashboard.ipynb`

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Assumptions

- **Module slug**: `m2_3_monitoring`
- **FastAPI app port**: 8001 (configurable)
- **Prometheus metrics port**: 8000 (configurable via `METRICS_PORT`)
- **Python version**: 3.8+
- **Docker required**: Only for Prometheus/Grafana stack (optional)

---

**Setup time:** 8-12 hours initial + 2-4 hours/month maintenance
**Value:** Real-time visibility into performance, costs, and quality 🚀
