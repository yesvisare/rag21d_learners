"""
M2.3 — Production Monitoring Dashboard

**Purpose:**
Implement production-grade monitoring for RAG systems using Prometheus and Grafana.
Learn to track performance metrics (latency, token usage, costs), emit structured logs,
build dashboards, and set up intelligent alerting for real-world RAG deployments.

**Concepts Covered:**
- Prometheus metric types (Counter, Histogram, Gauge, Summary)
- Time-series data collection and percentile calculations (p50/p95/p99)
- Structured JSON logging for cloud environments (CloudWatch, Stackdriver, Datadog)
- Cost tracking and budget alerting for LLM APIs
- Cache hit rate optimization and performance tuning
- Quality metrics (relevance scores, context precision)
- Grafana dashboard design and PromQL queries
- Alert rule configuration and avoiding alert fatigue
- Safe label cardinality management
- Trade-offs: self-hosted vs managed APM solutions

**After Completing:**
You will be able to:
- Instrument RAG pipelines with comprehensive metrics
- Design effective Grafana dashboards for production monitoring
- Set up proactive alerts for latency, errors, costs, and quality degradation
- Emit structured logs for debugging and audit trails
- Calculate and track per-query costs across multiple LLM models
- Optimize caching strategies using hit rate metrics
- Identify and prevent common monitoring pitfalls (cardinality explosions, alert fatigue)
- Decide when self-hosted monitoring is appropriate vs managed alternatives

**Context in Track:**
This module builds on M2.1 (Semantic Caching) and M2.2 (Prompt Compression) by adding
observability to production RAG systems. It provides the foundation for M2.4 (A/B Testing)
and M2.5 (Cost Optimization) by establishing metrics collection infrastructure. The
monitoring patterns learned here apply to any production ML/LLM system, not just RAG.

**Trade-offs & Reality Check:**
Self-hosted Prometheus/Grafana requires 8-12 hours initial setup + 2-4 hours/month
maintenance. Monthly infrastructure costs: $50-200. Only justified for >1K queries/day
with technical teams. For smaller volumes (<500 queries/day) or non-technical teams,
use native cloud logging (CloudWatch, Stackdriver) or managed APM (Datadog, New Relic).
"""

from .module import (
    RAGMetrics,
    StructuredLogger,
    monitored_query,
    start_metrics_server,
    track_cache_operation,
    track_rate_limit,
    track_rate_limit_hit,
    metrics,
    logger,
)

from .config import (
    METRICS_PORT,
    SERVICE_NAME,
    ENVIRONMENT,
    LOG_LEVEL,
    LOG_FORMAT,
    COST_PER_1K_INPUT_TOKENS,
    COST_PER_1K_OUTPUT_TOKENS,
    LATENCY_P95_THRESHOLD_MS,
    ERROR_RATE_THRESHOLD,
    CACHE_HIT_MIN_THRESHOLD,
)

__all__ = [
    # Core classes
    'RAGMetrics',
    'StructuredLogger',

    # Decorators and helpers
    'monitored_query',
    'start_metrics_server',
    'track_cache_operation',
    'track_rate_limit',
    'track_rate_limit_hit',

    # Global instances
    'metrics',
    'logger',

    # Configuration
    'METRICS_PORT',
    'SERVICE_NAME',
    'ENVIRONMENT',
    'LOG_LEVEL',
    'LOG_FORMAT',
    'COST_PER_1K_INPUT_TOKENS',
    'COST_PER_1K_OUTPUT_TOKENS',
    'LATENCY_P95_THRESHOLD_MS',
    'ERROR_RATE_THRESHOLD',
    'CACHE_HIT_MIN_THRESHOLD',
]

__version__ = '1.0.0'
