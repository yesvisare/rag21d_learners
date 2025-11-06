# M2.3 — Production Monitoring Dashboard

A complete production-grade monitoring solution for RAG (Retrieval-Augmented Generation) systems using Prometheus and Grafana.

## Overview

This workspace implements comprehensive monitoring for RAG pipelines with:

- **Performance metrics** (latency p50/p95/p99, retrieval time, LLM generation time)
- **Cost tracking** (per-query costs, daily spend by model)
- **Quality metrics** (relevance scores, context precision)
- **System health** (cache hit rates, error rates, rate limit headroom)
- **Structured logging** (JSON logs for cloud environments)
- **Grafana dashboards** (10 pre-built visualization panels)
- **Alert templates** (5 critical production alerts)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d

# Verify services are running
docker compose -f docker-compose.monitoring.yml ps
```

### 4. Start Metrics Endpoint

```python
from m2_3_monitoring import start_metrics_server

start_metrics_server(port=8000)
```

Or run the demo:

```bash
python m2_3_monitoring.py
```

### 5. Access Services

- **Metrics Endpoint**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 6. Import Grafana Dashboard

1. Navigate to http://localhost:3000
2. Log in (admin/admin)
3. Click **+** → **Import dashboard**
4. Upload `grafana_dash.json`
5. Select Prometheus as the data source
6. Click **Import**

## Usage

### Basic Instrumentation

Use the `@monitored_query` decorator to automatically track metrics:

```python
from m2_3_monitoring import monitored_query

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
from m2_3_monitoring import metrics, track_cache_operation

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
from m2_3_monitoring import StructuredLogger

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

## Dashboard Panels

The Grafana dashboard includes 10 panels:

1. **Query Latency (p50/p95/p99)** - Track performance trends
2. **Request Rate & Error Rate** - Monitor traffic and failures
3. **Token Usage** - Understand token consumption
4. **Cost per Query & Total Spend** - Control spending
5. **Cache Hit Rate %** - Optimize caching strategy
6. **Error Rate %** - Set SLA alerts
7. **Active Requests** - Detect traffic spikes
8. **Rate Limit Remaining** - Prevent rate limit hits
9. **Response Relevance Score** - Monitor answer quality
10. **Error Breakdown by Type** - Debug error patterns

## Alert Templates

Set up these critical alerts in Prometheus Alertmanager or Grafana:

1. **High Latency**: p95 > 2 seconds
2. **High Error Rate**: > 5%
3. **Low Cache Hit Rate**: < 30%
4. **High Cost Burn**: Daily spend > $100
5. **Rate Limit Warning**: < 100 requests remaining

See the notebook for full alert rule definitions.

## Testing

Run smoke tests to verify everything works:

```bash
python tests_monitoring.py
```

Or use pytest:

```bash
pytest tests_monitoring.py -v
```

## Architecture

```
┌─────────────────┐
│  Your RAG App   │
│  (instrumented) │
└────────┬────────┘
         │ exposes
         ▼
┌─────────────────┐
│ /metrics:8000   │ ◄─── scrapes ──┐
└─────────────────┘                 │
                            ┌───────┴────────┐
                            │  Prometheus    │
                            │  :9090         │
                            └───────┬────────┘
                                    │ queries
                                    ▼
                            ┌────────────────┐
                            │   Grafana      │
                            │   :3000        │
                            └────────────────┘
```

## Configuration

Edit `config.py` or `.env` to customize:

- **METRICS_PORT**: Port for metrics endpoint (default: 8000)
- **SERVICE_NAME**: Name for your service (default: rag-service)
- **ENV**: Environment (development/staging/production)
- **COST_PER_1K_INPUT_TOKENS**: Cost per 1K input tokens (default: $0.003)
- **COST_PER_1K_OUTPUT_TOKENS**: Cost per 1K output tokens (default: $0.015)

## When NOT to Use This

Self-hosted Prometheus/Grafana is **overkill** if:

- Processing < 500 queries/day → Use CloudWatch/Stackdriver logs
- Team has no DevOps experience → Use Datadog/New Relic
- Need results in < 2 hours → Start with basic logging first

**Cost comparison:**

| Approach | Setup Time | Monthly Cost | Best For |
|----------|------------|--------------|----------|
| Self-Hosted (Prometheus) | 8-12 hours | $50-200 | >1K queries/day |
| Managed APM | 30 minutes | $15-31/host | Non-technical teams |
| Native Cloud Logging | 1 hour | $5-20 | <500 queries/day |

## Common Failure Modes

Watch out for these issues:

1. **Metric Cardinality Explosion** - Don't use unbounded labels (user IDs, timestamps)
2. **Dashboard Query Timeouts** - Use recording rules for expensive queries
3. **Memory Leaks** - Keep label cardinality < 1000 per metric
4. **Alert Fatigue** - Start with 3-5 critical alerts, tune over time
5. **Metric Gaps** - Use remote storage for persistence across restarts

## Files

- `m2_3_monitoring.py` - Core monitoring library (metrics, logging, decorators)
- `config.py` - Configuration settings
- `.env.example` - Environment variable template
- `docker-compose.monitoring.yml` - Prometheus + Grafana containers
- `prometheus.yml` - Prometheus scrape configuration
- `grafana_dash.json` - Pre-built Grafana dashboard
- `M2_3_Production_Monitoring_Dashboard.ipynb` - Interactive tutorial
- `tests_monitoring.py` - Smoke tests
- `requirements.txt` - Python dependencies

## Resources

- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PromQL Guide**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Best Practices**: https://prometheus.io/docs/practices/naming/

## License

MIT

## Support

For issues or questions, refer to the notebook or source code comments.

---

**Setup time:** 8-12 hours initial + 2-4 hours/month maintenance
**Value:** Real-time visibility into performance, costs, and quality 🚀
