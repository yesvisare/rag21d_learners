# Enhanced Video Script: M2.3 - Production Monitoring Dashboard

## Video M2.3: Production Monitoring Dashboard (38-40 min)
**Duration:** 38-40 min  |  **Audience:** intermediate  |  **Prereqs:** M2.2 completed, working RAG system with caching

---

## OBJECTIVES
By the end of this video, learners will be able to:
- Implement production-grade monitoring with Prometheus and Grafana
- Design comprehensive metrics collection for RAG systems
- Debug common monitoring infrastructure failures
- **Decide when NOT to use self-hosted monitoring solutions**

---

### [0:00] Introduction

[SLIDE: "M2.3: Production Monitoring Dashboard"]

Welcome to the video where we finally answer the question: "What's actually happening in production?"

You've optimized caching. You've tuned your prompts. But here's the thing: **you can't trust what you can't see**. Without proper monitoring, you're flying blind. You don't know if your optimizations are working, if costs are spiking, or if users are getting errors.

Today, we're building a complete observability stack for RAG systems using industry-standard tools: Prometheus for metrics, Grafana for visualization, and structured logging.

[SLIDE: "What We'll Monitor"]
```
1. Performance Metrics
   - Query latency (p50, p95, p99)
   - Cache hit rates
   - Token usage

2. Cost Metrics
   - Real-time spending
   - Cost per query
   - Model usage breakdown

3. Quality Metrics
   - Response relevance
   - Context retrieval accuracy
   - Error rates

4. System Health
   - API availability
   - Queue depths
   - Rate limit headroom
```

**Important:** We'll also cover when this monitoring approach is overkill and what simpler alternatives exist for different scales.

Let's build this.

---

<!-- NEW SECTION: THE PROBLEM - Added per TVH Framework v2.0 -->

### [1:30] THE PROBLEM: Flying Blind in Production

**[1:30] [SLIDE: "What Happens Without Monitoring"]**

Before we dive into solutions, let me show you what happens when you DON'T have proper monitoring.

**[DEMO] Real production incident:**

[SCREEN: Show a RAG application terminal with sporadic errors]

Here's a production RAG system I ran last month. Look at this: queries work fine for a while, then suddenly...

[TERMINAL]
```bash
# Everything looks normal
Query 1: Success (234ms)
Query 2: Success (189ms)
Query 3: Success (212ms)
...
# Then disaster strikes
Query 47: ERROR - Rate limit exceeded
Query 48: ERROR - Rate limit exceeded
Query 49: ERROR - Rate limit exceeded
```

**The problem?** I had no visibility into:
- How close I was to rate limits
- Which queries were causing the spike
- How much this was costing me
- When it started happening

I discovered this **two days later** when a user complained. By then, I'd burned through $300 in retries and failed requests.

**[PAUSE]**

**Why logs alone aren't enough:**

Sure, I had logs. But here's what I couldn't see:
1. **Trends over time** - Was latency slowly increasing?
2. **Cost accumulation** - How much had I spent today vs yesterday?
3. **Real-time alerts** - No notification when error rate spiked
4. **Correlation** - Which cache misses correlated with high costs?

[DIAGRAM: Before/After comparison]
```
WITHOUT MONITORING:          WITH MONITORING:
- Errors → User reports      - Errors → Instant alerts
- Cost → Monthly bill shock  - Cost → Real-time tracking
- Latency → Guesswork        - Latency → p95/p99 metrics
- Capacity → Crashes         - Capacity → Proactive scaling
```

This is exactly what monitoring solves. But—and this is important—it comes with trade-offs we need to discuss honestly.

---

<!-- NEW SECTION: REALITY CHECK - Added per TVH Framework v2.0 -->

### [3:30] REALITY CHECK: What Monitoring Actually Costs You

**[3:30] [SLIDE: "Reality Check - The True Cost of Observability"]**

Let's set expectations straight. I'm going to tell you the honest truth about production monitoring infrastructure.

**What it DOES well:**
- ✅ **Real-time visibility**: See problems as they happen, not days later
- ✅ **Historical analysis**: Trend analysis and capacity planning with queryable metrics
- ✅ **Proactive alerting**: Get notified before users complain (when configured correctly)
- ✅ **Industry standard**: Prometheus + Grafana is what most teams use—good for hiring

**What it DOESN'T do:**
- ❌ **Fix problems automatically**: Monitoring shows you the fire, doesn't put it out
- ❌ **Work out of the box**: Requires significant configuration and tuning (2-4 weeks to get alerts right)
- ❌ **Stay simple**: You're now monitoring the monitoring system—meta-complexity
- ❌ **Scale for free**: Storage grows 1-5GB/day depending on metric cardinality

**[EMPHASIS]** This is important: **Prometheus itself needs monitoring.** You're adding 3-4 new services (Prometheus, Grafana, Alertmanager, exporters) that can fail, need updates, and require expertise to maintain.

**The trade-offs you're making:**

- **You gain** real-time visibility **but lose** simplicity (4 new moving parts)
- **Works great** for systems with 1000+ queries/day **but overkill** for hobby projects
- **Cost structure**: $50-200/month in infrastructure + 8-12 hours setup + ongoing maintenance

Let me be specific about costs:
- **Infrastructure**: $50-100/month for basic setup (Prometheus storage + compute)
- **Storage growth**: 1-5GB/day means $20-100/month additional for retention
- **Time investment**: 8-12 hours initial setup, 2-4 hours/month ongoing tuning
- **Learning curve**: Team needs 1-2 weeks to become proficient with PromQL

**Alert fatigue is real:**
- Week 1: You'll get 50+ alerts (most false positives)
- Week 2-3: Tuning thresholds and duration
- Week 4+: Finally getting actionable alerts

We'll see these trade-offs in action throughout this video. But first, let's talk about whether you even need this approach.

---

<!-- NEW SECTION: ALTERNATIVE SOLUTIONS - Added per TVH Framework v2.0 -->

### [6:00] ALTERNATIVE SOLUTIONS: When to Choose What

**[6:00] [SLIDE: "Three Monitoring Approaches"]**

Before we build with Prometheus + Grafana, you should know there are other ways to monitor production systems. Let's compare three approaches honestly.

**Option 1: Self-Hosted (Prometheus + Grafana) - What we're teaching today**
- **Best for**: Teams with DevOps capacity, >1K queries/day, need full control
- **Key trade-off**: Maximum flexibility but highest operational burden
- **Cost**: $50-200/month infrastructure, $0 licensing, 8-12 hours setup
- **Example use case**: Series A startup with 10K users, technical founding team, need to control costs long-term

**Option 2: Managed APM (DataDog, New Relic, Honeycomb)**
- **Best for**: Teams without DevOps, need out-of-box dashboards, fast time-to-value
- **Key trade-off**: Easier setup but vendor lock-in and ongoing per-host costs
- **Cost**: $15-31/host/month (so $180-360/year minimum), no setup time, instant dashboards
- **Example use case**: Bootstrapped startup, non-technical founder, would rather pay than maintain infrastructure

**Option 3: Simple Logging + Native Tools (CloudWatch Insights, Cloud Logging)**
- **Best for**: Early-stage projects, <500 queries/day, prototyping phase
- **Key trade-off**: Minimal setup but limited historical analysis and no custom metrics
- **Cost**: $5-20/month (included in cloud provider bills), 30 minutes setup
- **Example use case**: MVP validation phase, personal project, not sure if RAG is the right approach yet

**[DIAGRAM: Decision Framework]**

[SLIDE: Decision tree diagram]
```
Start: What's your query volume?
├─ < 500/day → Option 3 (Simple logging)
│              └─ Spend time on features, not infrastructure
│
├─ 500-5K/day → Do you have DevOps resources?
│              ├─ Yes → Option 1 (Prometheus)
│              │        └─ Will save money vs managed at scale
│              └─ No  → Option 2 (DataDog/New Relic)
│                       └─ Time-to-value more important
│
└─ > 5K/day → Option 1 (Prometheus) almost always
               └─ Managed APM costs become prohibitive
                  ($500-1000/month vs $200/month self-hosted)
```

**For this video, we're using Prometheus + Grafana because:**
1. You're at the stage where monitoring matters (Module 2 = production-ready)
2. It's the industry standard—you'll see this in job interviews
3. It teaches you observability fundamentals that apply to any tool
4. At scale, it's the most cost-effective option

**But remember:** If you're building an MVP or don't have DevOps capacity, there's no shame in using DataDog. It's what I use for my side projects because my time is worth more than $30/month.

**[PAUSE]**

Now let's see how to actually implement this.

---

<!-- EXISTING CONTENT STARTS HERE - Timestamps shifted from [1:30] to [8:30] -->

### [8:30] Metrics Collection Setup

[SLIDE: "Prometheus + Grafana Architecture"]

[CODE: "metrics_collector.py"]
```python
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import start_http_server, REGISTRY
import time
from typing import Dict, Optional
from functools import wraps
import logging

class RAGMetrics:
    """
    Comprehensive metrics collection for RAG systems.
    """
    
    def __init__(self, service_name: str = "rag_service"):
        self.service_name = service_name
        
        # Request metrics
        self.query_total = Counter(
            'rag_queries_total',
            'Total number of queries processed',
            ['cache_status', 'model', 'status']
        )
        
        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'Query processing time in seconds',
            ['endpoint', 'cache_status'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        )
        
        # Token metrics
        self.tokens_used = Histogram(
            'rag_tokens_used',
            'Number of tokens used per query',
            ['model', 'token_type'],  # token_type: input/output
            buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000)
        )
        
        # Cost metrics
        self.cost_per_query = Histogram(
            'rag_cost_per_query_usd',
            'Cost per query in USD',
            ['model'],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
        )
        
        self.total_cost = Counter(
            'rag_total_cost_usd',
            'Total accumulated cost in USD',
            ['model']
        )
        
        # Cache metrics
        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type']
        )
        
        self.cache_size = Gauge(
            'rag_cache_size_entries',
            'Number of entries in cache',
            ['cache_type']
        )
        
        # Vector database metrics
        self.vector_search_duration = Histogram(
            'rag_vector_search_duration_seconds',
            'Vector search latency',
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
        )
        
        self.documents_retrieved = Histogram(
            'rag_documents_retrieved',
            'Number of documents retrieved per query',
            buckets=(1, 3, 5, 10, 20, 50)
        )
        
        # Quality metrics
        self.relevance_score = Histogram(
            'rag_relevance_score',
            'Average relevance score of retrieved documents',
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        )
        
        # Error metrics
        self.errors_total = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Rate limit metrics
        self.rate_limit_remaining = Gauge(
            'rag_rate_limit_remaining',
            'Remaining API calls before rate limit',
            ['provider']
        )
        
        # System info
        self.service_info = Info(
            'rag_service',
            'RAG service information'
        )
        self.service_info.info({
            'version': '1.0.0',
            'service': service_name
        })
    
    def record_query(
        self,
        duration: float,
        cache_status: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        status: str = 'success'
    ):
        """Record a complete query with all metrics."""
        # Request metrics
        self.query_total.labels(
            cache_status=cache_status,
            model=model,
            status=status
        ).inc()
        
        self.query_duration.labels(
            endpoint='query',
            cache_status=cache_status
        ).observe(duration)
        
        # Token metrics
        self.tokens_used.labels(
            model=model,
            token_type='input'
        ).observe(tokens_input)
        
        self.tokens_used.labels(
            model=model,
            token_type='output'
        ).observe(tokens_output)
        
        # Cost metrics
        self.cost_per_query.labels(model=model).observe(cost)
        self.total_cost.labels(model=model).inc(cost)
    
    def record_cache_stats(self, cache_type: str, hit_rate: float, size: int):
        """Update cache metrics."""
        self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    def record_vector_search(self, duration: float, num_docs: int, avg_score: float):
        """Record vector search metrics."""
        self.vector_search_duration.observe(duration)
        self.documents_retrieved.observe(num_docs)
        self.relevance_score.observe(avg_score)
    
    def record_error(self, error_type: str, component: str):
        """Record an error."""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def update_rate_limit(self, provider: str, remaining: int):
        """Update rate limit information."""
        self.rate_limit_remaining.labels(provider=provider).set(remaining)

def monitored_query(metrics: RAGMetrics):
    """
    Decorator to automatically collect metrics for query functions.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract metrics from result
                metrics.record_query(
                    duration=duration,
                    cache_status=result.get('cache_status', 'miss'),
                    model=result.get('model', 'unknown'),
                    tokens_input=result.get('tokens_input', 0),
                    tokens_output=result.get('tokens_output', 0),
                    cost=result.get('cost', 0.0),
                    status='success'
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__
                
                metrics.record_error(error_type, 'query_handler')
                
                # Still record the failed query
                metrics.query_total.labels(
                    cache_status='miss',
                    model='unknown',
                    status='error'
                ).inc()
                
                raise
        
        return wrapper
    return decorator

# Example integration
class MonitoredRAGPipeline:
    """
    RAG pipeline with comprehensive monitoring.
    """
    
    def __init__(self, openai_client, vector_db, cache):
        self.client = openai_client
        self.vector_db = vector_db
        self.cache = cache
        
        # Initialize metrics
        self.metrics = RAGMetrics(service_name="production_rag")
        
        # Start metrics server
        start_http_server(8000)  # Prometheus scrapes this endpoint
        print("Metrics server started on :8000")
    
    @monitored_query(metrics=None)  # Will use self.metrics
    def query(self, user_query: str) -> Dict:
        """
        Process query with full monitoring.
        """
        # Check cache
        cached = self.cache.get_cached_response(user_query)
        if cached:
            return {
                'response': cached,
                'cache_status': 'hit',
                'model': 'cached',
                'tokens_input': 0,
                'tokens_output': 0,
                'cost': 0.0
            }
        
        # Vector search with timing
        search_start = time.time()
        results = self.vector_db.search(user_query, top_k=5)
        search_duration = time.time() - search_start
        
        # Record vector search metrics
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        self.metrics.record_vector_search(
            duration=search_duration,
            num_docs=len(results),
            avg_score=avg_score
        )
        
        # Generate response
        context = self._format_context(results)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based on context."},
                {"role": "user", "content": f"Context: {context}\n\nQ: {user_query}"}
            ]
        )
        
        # Extract token usage
        usage = response.usage
        cost = (usage.prompt_tokens * 0.0000015 + 
                usage.completion_tokens * 0.000002)
        
        # Update rate limit info (from headers if available)
        # This would come from response headers in real implementation
        self.metrics.update_rate_limit('openai', 9500)  # Example
        
        answer = response.choices[0].message.content
        
        # Cache the result
        self.cache.cache_response(user_query, answer)
        
        return {
            'response': answer,
            'cache_status': 'miss',
            'model': 'gpt-3.5-turbo',
            'tokens_input': usage.prompt_tokens,
            'tokens_output': usage.completion_tokens,
            'cost': cost
        }
    
    def _format_context(self, results):
        return "\n".join([r['content'] for r in results])
```

[TERMINAL: Start the metrics server]
```bash
python monitored_rag_pipeline.py
# Metrics available at http://localhost:8000/metrics
```

[SCREEN: Show Prometheus metrics endpoint output]

### [13:00] Structured Logging

[SLIDE: "Logging Best Practices"]

Metrics tell you *what* is happening. Logs tell you *why*.

[CODE: "structured_logging.py"]
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any
import traceback
import sys

class StructuredLogger:
    """
    JSON structured logging for RAG systems.
    """
    
    def __init__(self, service_name: str, environment: str = "production"):
        self.service_name = service_name
        self.environment = environment
        
        # Configure logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove default handlers
        self.logger.handlers = []
        
        # Add JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._json_formatter())
        self.logger.addHandler(handler)
    
    def _json_formatter(self):
        """Create custom JSON formatter."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'logger': record.name,
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra_fields'):
                    log_data.update(record.extra_fields)
                
                # Add exception info if present
                if record.exc_info:
                    log_data['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': ''.join(traceback.format_tb(record.exc_info[2]))
                    }
                
                return json.dumps(log_data)
        
        return JSONFormatter()
    
    def log_query(
        self,
        query_id: str,
        user_query: str,
        cache_hit: bool,
        model: str,
        latency_ms: float,
        cost: float,
        **kwargs
    ):
        """Log a query with all relevant context."""
        extra = {
            'extra_fields': {
                'event_type': 'query',
                'query_id': query_id,
                'query_length': len(user_query),
                'cache_hit': cache_hit,
                'model': model,
                'latency_ms': latency_ms,
                'cost_usd': cost,
                'service': self.service_name,
                'environment': self.environment,
                **kwargs
            }
        }
        
        self.logger.info(
            f"Query processed: {query_id}",
            extra=extra
        )
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str = "error"
    ):
        """Log an error with full context."""
        extra = {
            'extra_fields': {
                'event_type': 'error',
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'service': self.service_name,
                'environment': self.environment
            }
        }
        
        if severity == "critical":
            self.logger.critical(
                f"Critical error: {str(error)}",
                extra=extra,
                exc_info=True
            )
        else:
            self.logger.error(
                f"Error: {str(error)}",
                extra=extra,
                exc_info=True
            )
    
    def log_cache_performance(
        self,
        cache_type: str,
        hit_rate: float,
        size: int,
        **kwargs
    ):
        """Log cache performance metrics."""
        extra = {
            'extra_fields': {
                'event_type': 'cache_stats',
                'cache_type': cache_type,
                'hit_rate': hit_rate,
                'size': size,
                'service': self.service_name,
                **kwargs
            }
        }
        
        self.logger.info(
            f"Cache stats - {cache_type}: hit_rate={hit_rate:.2%}",
            extra=extra
        )
    
    def log_model_performance(
        self,
        model: str,
        avg_latency_ms: float,
        avg_tokens: float,
        total_queries: int,
        **kwargs
    ):
        """Log model performance summary."""
        extra = {
            'extra_fields': {
                'event_type': 'model_stats',
                'model': model,
                'avg_latency_ms': avg_latency_ms,
                'avg_tokens': avg_tokens,
                'total_queries': total_queries,
                'service': self.service_name,
                **kwargs
            }
        }
        
        self.logger.info(
            f"Model stats - {model}",
            extra=extra
        )

# Integration example
class FullyMonitoredRAG:
    """
    RAG pipeline with metrics and structured logging.
    """
    
    def __init__(self):
        self.metrics = RAGMetrics()
        self.logger = StructuredLogger("production_rag", "prod")
        
        # Start metrics server
        start_http_server(8000)
    
    def query(self, user_query: str, query_id: str) -> Dict:
        """Process query with full observability."""
        start_time = time.time()
        
        try:
            # Your RAG logic here
            result = self._process_query(user_query)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics.record_query(
                duration=latency_ms / 1000,
                cache_status=result['cache_status'],
                model=result['model'],
                tokens_input=result['tokens_input'],
                tokens_output=result['tokens_output'],
                cost=result['cost']
            )
            
            # Log the query
            self.logger.log_query(
                query_id=query_id,
                user_query=user_query,
                cache_hit=result['cache_status'] == 'hit',
                model=result['model'],
                latency_ms=latency_ms,
                cost=result['cost'],
                tokens_total=result['tokens_input'] + result['tokens_output']
            )
            
            return result
            
        except Exception as e:
            # Log error with full context
            self.logger.log_error(
                error=e,
                context={
                    'query_id': query_id,
                    'query': user_query,
                    'latency_ms': (time.time() - start_time) * 1000
                }
            )
            
            # Record error metric
            self.metrics.record_error(
                error_type=type(e).__name__,
                component='query_handler'
            )
            
            raise
    
    def _process_query(self, query: str) -> Dict:
        """Mock query processing."""
        # Implement your actual RAG logic
        return {
            'response': 'Answer',
            'cache_status': 'miss',
            'model': 'gpt-3.5-turbo',
            'tokens_input': 150,
            'tokens_output': 100,
            'cost': 0.0005
        }
```

[TERMINAL: Show structured log output]
```json
{
  "timestamp": "2025-10-08T14:23:45.123Z",
  "level": "INFO",
  "message": "Query processed: q_12345",
  "event_type": "query",
  "query_id": "q_12345",
  "cache_hit": false,
  "model": "gpt-3.5-turbo",
  "latency_ms": 234.5,
  "cost_usd": 0.0005,
  "tokens_total": 250
}
```

### [16:00] Grafana Dashboard Setup

[SLIDE: "Building Your Dashboard"]

Now let's visualize everything. I'll show you the Grafana configuration.

[CODE: "grafana_dashboard.json"]
```json
{
  "dashboard": {
    "title": "RAG System - Production Dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Query Rate (queries/sec)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_queries_total[5m])"
          }
        ]
      },
      {
        "id": 2,
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rag_cache_hit_rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 30, "color": "yellow"},
                {"value": 50, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "id": 4,
        "title": "Cost per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_total_cost_usd[1h]) * 3600"
          }
        ]
      },
      {
        "id": 5,
        "title": "Token Usage by Model",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_tokens_used[5m]) by (model)"
          }
        ]
      },
      {
        "id": 6,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_errors_total[5m])"
          }
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "params": [0.01],
                "type": "gt"
              }
            }
          ]
        }
      }
    ]
  }
}
```

[SCREEN: Show completed Grafana dashboard with real-time data]

### [18:00] Alerting Setup

[SLIDE: "Proactive Monitoring with Alerts"]

Don't wait for problems to find you. Set up alerts.

[CODE: "alerting_rules.py"]
```python
from typing import List, Dict, Callable
import time
from dataclasses import dataclass

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: Callable
    threshold: float
    duration_seconds: int
    severity: str  # 'warning' or 'critical'
    notification_channels: List[str]

class AlertingSystem:
    """
    Simple alerting system for RAG metrics.
    """
    
    def __init__(self):
        self.rules = []
        self.alert_state = {}  # Track ongoing alerts
        self.notification_handlers = {
            'console': self._console_notification,
            'slack': self._slack_notification,
            'email': self._email_notification,
        }
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def check_alerts(self, metrics: Dict[str, float]):
        """
        Check all rules against current metrics.
        Call this periodically (e.g., every 30 seconds).
        """
        current_time = time.time()
        
        for rule in self.rules:
            try:
                # Evaluate condition
                triggered = rule.condition(metrics)
                
                alert_key = rule.name
                
                if triggered:
                    if alert_key not in self.alert_state:
                        # First time triggered
                        self.alert_state[alert_key] = {
                            'first_triggered': current_time,
                            'notified': False
                        }
                    
                    # Check if duration threshold met
                    state = self.alert_state[alert_key]
                    time_triggered = current_time - state['first_triggered']
                    
                    if time_triggered >= rule.duration_seconds and not state['notified']:
                        # Fire alert
                        self._fire_alert(rule, metrics)
                        state['notified'] = True
                else:
                    # Condition no longer met - resolve alert
                    if alert_key in self.alert_state:
                        if self.alert_state[alert_key]['notified']:
                            self._resolve_alert(rule)
                        del self.alert_state[alert_key]
            
            except Exception as e:
                print(f"Error checking alert rule {rule.name}: {e}")
    
    def _fire_alert(self, rule: AlertRule, metrics: Dict):
        """Send alert notifications."""
        message = f"🚨 [{rule.severity.upper()}] {rule.name}\n"
        message += f"Threshold: {rule.threshold}\n"
        message += f"Current metrics: {metrics}\n"
        message += f"Duration: {rule.duration_seconds}s"
        
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                self.notification_handlers[channel](rule, message)
    
    def _resolve_alert(self, rule: AlertRule):
        """Send alert resolution notification."""
        message = f"✅ RESOLVED: {rule.name}"
        
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                self.notification_handlers[channel](rule, message)
    
    def _console_notification(self, rule: AlertRule, message: str):
        """Print to console."""
        print(f"\n{'='*60}")
        print(message)
        print('='*60)
    
    def _slack_notification(self, rule: AlertRule, message: str):
        """Send to Slack (implement with slack_sdk)."""
        # from slack_sdk import WebClient
        # client = WebClient(token=SLACK_TOKEN)
        # client.chat_postMessage(channel='#alerts', text=message)
        print(f"[SLACK] {message}")
    
    def _email_notification(self, rule: AlertRule, message: str):
        """Send email alert (implement with SMTP)."""
        print(f"[EMAIL] {message}")

# Define common alert rules
def setup_production_alerts() -> AlertingSystem:
    """
    Configure standard production alerts.
    """
    alerting = AlertingSystem()
    
    # High error rate
    alerting.add_rule(AlertRule(
        name="High Error Rate",
        condition=lambda m: m.get('error_rate', 0) > 0.05,  # 5%
        threshold=0.05,
        duration_seconds=60,
        severity='critical',
        notification_channels=['console', 'slack']
    ))
    
    # Low cache hit rate
    alerting.add_rule(AlertRule(
        name="Low Cache Hit Rate",
        condition=lambda m: m.get('cache_hit_rate', 1) < 0.20,  # 20%
        threshold=0.20,
        duration_seconds=300,
        severity='warning',
        notification_channels=['console']
    ))
    
    # High latency
    alerting.add_rule(AlertRule(
        name="High P95 Latency",
        condition=lambda m: m.get('p95_latency_ms', 0) > 2000,  # 2s
        threshold=2000,
        duration_seconds=120,
        severity='warning',
        notification_channels=['console', 'slack']
    ))
    
    # Cost spike
    alerting.add_rule(AlertRule(
        name="Cost Spike",
        condition=lambda m: m.get('cost_per_hour', 0) > 10,  # $10/hour
        threshold=10,
        duration_seconds=300,
        severity='critical',
        notification_channels=['console', 'slack', 'email']
    ))
    
    # Rate limit approaching
    alerting.add_rule(AlertRule(
        name="API Rate Limit Warning",
        condition=lambda m: m.get('rate_limit_remaining', 10000) < 1000,
        threshold=1000,
        duration_seconds=60,
        severity='warning',
        notification_channels=['console']
    ))
    
    return alerting

# Example monitoring loop
def run_monitoring_loop():
    """
    Main monitoring loop - run this as a background service.
    """
    import time
    from prometheus_client import REGISTRY
    
    alerting = setup_production_alerts()
    
    print("Starting monitoring loop...")
    
    while True:
        try:
            # Collect current metrics
            # In production, query Prometheus API
            metrics = {
                'error_rate': 0.02,  # Example: get from Prometheus
                'cache_hit_rate': 0.45,
                'p95_latency_ms': 450,
                'cost_per_hour': 2.5,
                'rate_limit_remaining': 8500
            }
            
            # Check alerts
            alerting.check_alerts(metrics)
            
            # Sleep for check interval
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(30)
```

[SCREEN: Show alert firing in console]

### [21:00] Cost Tracking Dashboard

[SLIDE: "Real-Time Cost Monitoring"]

Let's build a specific cost tracking component.

[CODE: "cost_tracker.py"]
```python
from datetime import datetime, timedelta
from typing import Dict, List
import json

class CostTracker:
    """
    Track and analyze costs in real-time.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.COST_KEY_PREFIX = "costs:"
        
        # Model pricing (per 1M tokens)
        self.pricing = {
            'gpt-4o': {'input': 5.0, 'output': 15.0},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
            'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
            'text-embedding-3-small': {'input': 0.02, 'output': 0.0},
        }
    
    def record_cost(
        self,
        query_id: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        timestamp: datetime = None
    ):
        """Record cost for a single query."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Calculate cost
        if model not in self.pricing:
            print(f"Warning: Unknown model {model}")
            return
        
        prices = self.pricing[model]
        cost = (
            (tokens_input / 1_000_000 * prices['input']) +
            (tokens_output / 1_000_000 * prices['output'])
        )
        
        # Store cost record
        record = {
            'query_id': query_id,
            'model': model,
            'tokens_input': tokens_input,
            'tokens_output': tokens_output,
            'cost': cost,
            'timestamp': timestamp.isoformat()
        }
        
        # Store in time-series key (by hour)
        hour_key = f"{self.COST_KEY_PREFIX}{timestamp.strftime('%Y-%m-%d-%H')}"
        self.redis.lpush(hour_key, json.dumps(record))
        self.redis.expire(hour_key, 604800)  # Keep 7 days
        
        # Update running totals
        date_key = f"{self.COST_KEY_PREFIX}daily:{timestamp.strftime('%Y-%m-%d')}"
        self.redis.incrbyfloat(date_key, cost)
        self.redis.expire(date_key, 2592000)  # Keep 30 days
    
    def get_hourly_cost(self, hours_ago: int = 0) -> float:
        """Get cost for a specific hour."""
        target_time = datetime.utcnow() - timedelta(hours=hours_ago)
        hour_key = f"{self.COST_KEY_PREFIX}{target_time.strftime('%Y-%m-%d-%H')}"
        
        records = self.redis.lrange(hour_key, 0, -1)
        total = sum(json.loads(r)['cost'] for r in records)
        
        return total
    
    def get_daily_cost(self, days_ago: int = 0) -> float:
        """Get total cost for a specific day."""
        target_date = datetime.utcnow() - timedelta(days=days_ago)
        date_key = f"{self.COST_KEY_PREFIX}daily:{target_date.strftime('%Y-%m-%d')}"
        
        cost = self.redis.get(date_key)
        return float(cost) if cost else 0.0
    
    def get_cost_breakdown_by_model(self, hours: int = 24) -> Dict[str, float]:
        """Get cost breakdown by model for last N hours."""
        breakdown = {}
        
        for h in range(hours):
            target_time = datetime.utcnow() - timedelta(hours=h)
            hour_key = f"{self.COST_KEY_PREFIX}{target_time.strftime('%Y-%m-%d-%H')}"
            
            records = self.redis.lrange(hour_key, 0, -1)
            
            for record_json in records:
                record = json.loads(record_json)
                model = record['model']
                cost = record['cost']
                
                breakdown[model] = breakdown.get(model, 0) + cost
        
        return breakdown
    
    def get_cost_projection(self) -> Dict[str, float]:
        """
        Project monthly cost based on recent usage.
        """
        # Get last 24 hours
        last_24h_cost = sum(self.get_hourly_cost(h) for h in range(24))
        
        # Get last 7 days
        last_7d_cost = sum(self.get_daily_cost(d) for d in range(7))
        
        # Calculate averages
        avg_hourly = last_24h_cost / 24
        avg_daily = last_7d_cost / 7
        
        # Project
        return {
            'last_hour': self.get_hourly_cost(0),
            'last_24h': last_24h_cost,
            'last_7d': last_7d_cost,
            'projected_daily': avg_daily,
            'projected_monthly': avg_daily * 30,
            'projected_yearly': avg_daily * 365
        }
    
    def generate_cost_report(self) -> str:
        """Generate human-readable cost report."""
        projection = self.get_cost_projection()
        breakdown = self.get_cost_breakdown_by_model(hours=24)
        
        report = []
        report.append("\n" + "="*60)
        report.append("COST REPORT")
        report.append("="*60)
        report.append(f"\nLast Hour: ${projection['last_hour']:.4f}")
        report.append(f"Last 24 Hours: ${projection['last_24h']:.2f}")
        report.append(f"Last 7 Days: ${projection['last_7d']:.2f}")
        report.append(f"\nProjections:")
        report.append(f"  Daily: ${projection['projected_daily']:.2f}")
        report.append(f"  Monthly: ${projection['projected_monthly']:.2f}")
        report.append(f"  Yearly: ${projection['projected_yearly']:.2f}")
        
        report.append(f"\nCost by Model (24h):")
        for model, cost in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (cost / projection['last_24h'] * 100) if projection['last_24h'] > 0 else 0
            report.append(f"  {model}: ${cost:.4f} ({percentage:.1f}%)")
        
        return "\n".join(report)

# Example usage
def demonstrate_cost_tracking():
    """Show cost tracking in action."""
    import redis
    
    redis_client = redis.Redis()
    tracker = CostTracker(redis_client)
    
    # Simulate some queries
    models_to_test = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']
    
    for i in range(100):
        model = models_to_test[i % len(models_to_test)]
        tracker.record_cost(
            query_id=f"q_{i}",
            model=model,
            tokens_input=150,
            tokens_output=100
        )
    
    # Print report
    print(tracker.generate_cost_report())
```

[TERMINAL: Show cost report output]

### [23:30] Health Checks & Readiness Probes

[SLIDE: "Kubernetes-Ready Health Checks"]

If you're deploying to Kubernetes, you need proper health checks.

[CODE: "health_checks.py"]
```python
from fastapi import FastAPI, Response
from typing import Dict
import time
import redis
import json

app = FastAPI()

class HealthChecker:
    """
    Comprehensive health checking for RAG service.
    """
    
    def __init__(self, redis_client, vector_db, openai_client):
        self.redis = redis_client
        self.vector_db = vector_db
        self.openai = openai_client
        self.start_time = time.time()
    
    def check_redis(self) -> Dict[str, any]:
        """Check Redis connectivity."""
        try:
            self.redis.ping()
            return {'status': 'healthy', 'latency_ms': 0}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_vector_db(self) -> Dict[str, any]:
        """Check vector database connectivity."""
        try:
            start = time.time()
            # Do a simple query
            # self.vector_db.ping() or similar
            latency = (time.time() - start) * 1000
            return {'status': 'healthy', 'latency_ms': latency}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_openai(self) -> Dict[str, any]:
        """Check OpenAI API connectivity."""
        try:
            # Simple models list call
            self.openai.models.list()
            return {'status': 'healthy'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def full_health_check(self) -> Dict[str, any]:
        """Run all health checks."""
        checks = {
            'redis': self.check_redis(),
            'vector_db': self.check_vector_db(),
            'openai': self.check_openai()
        }
        
        # Overall status
        all_healthy = all(
            c['status'] == 'healthy' for c in checks.values()
        )
        
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'uptime_seconds': int(time.time() - self.start_time),
            'checks': checks
        }

# Initialize (do this once at startup)
health_checker = HealthChecker(
    redis_client=redis.Redis(),
    vector_db=None,  # Your vector DB client
    openai_client=None  # Your OpenAI client
)

@app.get("/health")
def health():
    """
    Liveness probe - is the service running?
    Returns 200 if service is alive.
    """
    return {"status": "ok"}

@app.get("/ready")
def readiness():
    """
    Readiness probe - is the service ready to handle traffic?
    Returns 200 only if all dependencies are healthy.
    """
    health = health_checker.full_health_check()
    
    if health['status'] == 'healthy':
        return health
    else:
        return Response(
            content=json.dumps(health),
            status_code=503,
            media_type="application/json"
        )

@app.get("/metrics")
def metrics_endpoint():
    """
    Expose Prometheus metrics.
    This is already handled by prometheus_client,
    but shown here for completeness.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

[SCREEN: Show Kubernetes deployment YAML with health checks]

---

<!-- NEW SECTION: WHEN THIS BREAKS - Added per TVH Framework v2.0 -->

### [25:00] WHEN THIS BREAKS: Common Monitoring Failures

**[25:00] [SLIDE: "When the Monitoring System Fails"]**

Now for the MOST important part: what to do when your monitoring infrastructure itself goes wrong. Ironically, monitoring systems fail too. Let me show you the 5 most common issues and how to debug them.

---

#### Failure #1: Cardinality Explosion (Prometheus Storage Full)

**[25:00] [TERMINAL] Let me reproduce this error:**

[CODE: "cardinality_explosion.py"]
```python
# BAD: Adding unbounded labels
from prometheus_client import Counter

# This will kill Prometheus!
query_counter = Counter(
    'queries_total',
    'Total queries',
    ['user_id', 'query_text']  # ❌ BAD: Unbounded cardinality
)

# With 10K users and unique queries, you get millions of time series
for user_id in range(10000):
    query_counter.labels(user_id=user_id, query_text=f"unique_query_{user_id}").inc()
```

**Error message you'll see:**
```
PANIC: Out of disk space
Prometheus storage full at /data/prometheus
Cardinality: 4.2M time series (limit: 1M)
```

**What this means:**
Each unique combination of label values creates a new time series. With user_id (10K values) and query_text (infinite values), you're creating millions of time series. Prometheus stores each series separately, filling disk in hours.

**How to fix it:**

[SCREEN] [CODE: "cardinality_fix.py"]
```python
# GOOD: Bounded labels only
query_counter = Counter(
    'queries_total',
    'Total queries',
    ['cache_status', 'model']  # ✅ GOOD: Bounded cardinality (2 * 5 = 10 series)
)

# Use logging for high-cardinality data
logger.info("Query processed", extra={
    'user_id': user_id,  # Put in logs, not metrics
    'query_text': query_text
})
```

**How to verify:**
```bash
# Check current cardinality
curl localhost:9090/api/v1/status/tsdb | jq '.data.numSeries'
# Should be < 10,000 for most apps
```

**How to prevent:**
Never use labels with >100 possible values. User IDs, query text, timestamps, UUIDs all belong in logs, not metrics.

---

#### Failure #2: Grafana Dashboard Query Timeout

**[26:00] [SCREEN] Let me show you a dashboard that brings Grafana to its knees:**

[CODE: "slow_dashboard_query.json"]
```json
{
  "expr": "rate(rag_queries_total[24h])"  // ❌ BAD: 24h window
}
```

**Error message you'll see:**
```
Query timeout after 60s
Expression: rate(rag_queries_total[24h])
Evaluation took too long
```

**What this means:**
You're asking Prometheus to load 24 hours of data (millions of data points) and calculate rate. The query evaluates every data point across the entire time range. With high-cardinality metrics, this overwhelms Prometheus.

**How to fix it:**

[CODE: "fast_dashboard_query.json"]
```json
{
  // GOOD: Use shorter windows with aggregation
  "expr": "rate(rag_queries_total[5m])"  // ✅ GOOD: 5m window
}
```

**Alternative for long-term trends:**
```promql
# Use recording rules for expensive queries
# prometheus.yml
groups:
  - name: rag_rules
    interval: 1m
    rules:
      - record: rag:query_rate:5m
        expr: rate(rag_queries_total[5m])
```

**How to verify:**
[TERMINAL]
```bash
# Test query performance
time curl -g 'http://localhost:9090/api/v1/query?query=rate(rag_queries_total[5m])'
# Should return < 1 second
```

**How to prevent:**
Keep dashboard queries to 5-15 minute windows. Use recording rules for anything computed over >1 hour.

---

#### Failure #3: Metrics Server Memory Leak (Unbounded Labels)

**[27:00] [TERMINAL] Watch this memory leak in action:**

[CODE: "memory_leak_demo.py"]
```python
from prometheus_client import Histogram
import time

# BAD: Creating infinite label combinations
latency = Histogram('query_latency', 'Latency', ['timestamp'])

# This leaks memory because each timestamp is unique
for i in range(100000):
    latency.labels(timestamp=str(time.time())).observe(0.1)
    # Memory grows: 100MB -> 200MB -> 500MB -> OOM

print(f"Created {len(latency._metrics)} time series")  # Thousands!
```

**Error message you'll see:**
```
Process killed: Out of Memory (OOM)
Python process using 2.4GB (limit: 2GB)
Last activity: Metrics collection
```

**What this means:**
Each call to `.labels()` with a unique timestamp creates a new time series that's stored in memory forever. Python's prometheus_client keeps all time series in RAM. With unbounded labels, memory grows until the process crashes.

**How to fix it:**

[CODE: "memory_leak_fix.py"]
```python
# GOOD: No timestamps in labels
latency = Histogram('query_latency', 'Latency', ['model'])

# Timestamps are implicit in the metric scrape time
for i in range(100000):
    latency.labels(model='gpt-3.5-turbo').observe(0.1)
    # Memory stays constant: one time series per model
```

**How to verify:**
```bash
# Monitor metrics endpoint memory usage
ps aux | grep "python.*metrics" | awk '{print $6/1024 " MB"}'

# Check metric cardinality
curl localhost:8000/metrics | grep -c "^rag_"
# Should be stable, not growing
```

**How to prevent:**
Review all metric labels. If a label value changes every request (timestamps, request IDs, user IDs), remove it.

---

#### Failure #4: Alert Fatigue (Poorly Tuned Thresholds)

**[28:00] [DEMO] Let me show you an alerting disaster:**

[SCREEN: Slack channel with hundreds of alerts]
```
[06:23] ⚠️ High Latency Alert
[06:24] ✅ Resolved: High Latency
[06:25] ⚠️ High Latency Alert
[06:26] ✅ Resolved: High Latency
... (50 more times) ...
```

**Error message you'll see:**
No error—but your team ignores all alerts because of noise. The real incident gets missed in the flood.

**What this means:**
Alert threshold is too sensitive (triggers on normal variance) or duration is too short (triggers before transient spikes resolve). Your latency naturally fluctuates 200-400ms, but you're alerting on anything >300ms for >10 seconds.

**How to fix it:**

[CODE: "alert_tuning.py"]
```python
# BAD: Too sensitive
AlertRule(
    name="High Latency",
    condition=lambda m: m['p95_latency_ms'] > 300,  # ❌ Too low
    duration_seconds=10,  # ❌ Too short
    severity='critical'
)

# GOOD: Tuned with buffer
AlertRule(
    name="High Latency",
    condition=lambda m: m['p95_latency_ms'] > 500,  # ✅ Above normal range
    duration_seconds=120,  # ✅ Sustained problem
    severity='warning'  # ✅ Not critical unless >1000ms
)
```

**How to verify:**
[TERMINAL]
```bash
# Review alert history
grep "Alert:" logs/monitoring.log | wc -l
# Should be < 5 per day

# Check false positive rate
grep "Resolved:" logs/monitoring.log | head -n 20
# If most alerts resolve in <5 min, thresholds too low
```

**How to prevent:**
- Run alerts in "dry run" mode for 1 week, log instead of notify
- Set thresholds at p95 of normal behavior + 50% buffer
- Require 2+ minutes sustained before alerting
- Alert on rate of change, not absolute values

---

#### Failure #5: Missing Metrics After Deployment

**[29:00] [TERMINAL] The deployment looks fine, but:**

[SCREEN: Grafana dashboard with flat lines]
```
All metrics showing "No data" after 14:30 (deployment time)
Last data point: 14:29
Current time: 15:45
```

**Error message you'll see:**
```bash
# Prometheus target page shows:
Target: http://app:8000/metrics
Status: DOWN
Error: Connection refused
```

**What this means:**
New code deployed without metrics instrumentation, or metrics server not started. The decorator or metrics initialization code was removed during refactoring. Prometheus can't scrape the endpoint.

**How to fix it:**

[CODE: "deployment_fix.py"]
```python
# Check if metrics server is running
from prometheus_client import start_http_server

# COMMON MISTAKE: Forgetting to start the server
def main():
    # ... your app code ...
    pass
    # ❌ Missing: start_http_server(8000)

# FIXED VERSION
def main():
    # Start metrics server FIRST
    start_http_server(8000)  # ✅ CRITICAL: Don't forget this!
    print("Metrics available at :8000/metrics")
    
    # Then start your app
    # ... your app code ...
```

**How to verify:**
[TERMINAL]
```bash
# Add to CI/CD pipeline
echo "Testing metrics endpoint..."
curl -f http://localhost:8000/metrics || exit 1

# Verify key metrics exist
curl http://localhost:8000/metrics | grep "rag_queries_total" || exit 1
```

**How to prevent:**
- Add metrics endpoint test to CI/CD pipeline
- Use Prometheus target monitoring (alert if target down >2 min)
- Include metrics in smoke tests post-deployment
- Document metrics initialization in deployment runbook

---

**[29:30] [SLIDE: Monitoring Failure Prevention Checklist]**

To avoid these errors:
- [ ] Limit metric labels to <10 possible values each
- [ ] Keep dashboard queries under 5-15 minute windows
- [ ] Review memory usage of metrics server weekly
- [ ] Tune alert thresholds using 1-week dry run
- [ ] Add metrics endpoint to deployment tests
- [ ] Monitor Prometheus itself (disk, memory, query latency)
- [ ] Set up alerts for monitoring system health

---

<!-- NEW SECTION: WHEN NOT TO USE - Added per TVH Framework v2.0 -->

### [30:00] WHEN NOT TO USE THIS: Monitoring Anti-Patterns

**[30:00] [SLIDE: "When Self-Hosted Monitoring is Wrong"]**

Let me be crystal clear about when you should NOT use Prometheus + Grafana.

**❌ Don't use this when:**

**1. Early-Stage / MVP Phase (<500 queries/day)**
   - **Why it's wrong:** You're spending 12 hours on monitoring infrastructure when you should be validating product-market fit. The complexity added (4 new services) is not worth it when you could use provider logs.
   - **Use instead:** CloudWatch Logs + simple log queries OR just check OpenAI dashboard for costs
   - **Example:** You're building a chatbot MVP for local restaurant. You have 50 users testing. Setting up Prometheus is massive overkill—just tail logs and check daily costs in OpenAI dashboard.
   - **Red flag:** If you're still figuring out what to build, monitoring infrastructure is premature optimization.

**2. No DevOps Resources (Solo developer or non-technical team)**
   - **Why it's wrong:** Prometheus requires ongoing maintenance: version updates, storage scaling, backup configuration, PromQL knowledge. You'll spend 2-4 hours/month maintaining it. That's $400-800 of your time at consulting rates.
   - **Use instead:** DataDog ($31/host/month) or New Relic—worth the cost to avoid ops burden
   - **Example:** You're a solo founder building a RAG app. You're strong at Python but weak at infrastructure. You keep getting alerts about Prometheus disk full at 3am. Pay $30/month for DataDog and sleep.
   - **Red flag:** If you're Googling "how to scale Prometheus" and you're not a platform engineer, you've chosen the wrong approach.

**3. Budget Constraints BUT Need Monitoring (Prototyping on tight budget)**
   - **Why it's wrong:** The $50-200/month infrastructure cost might seem fine, but when your project doesn't get traction, you're paying for monitoring unused infrastructure. Plus your time cost for setup (12 hours = $1,200+ opportunity cost).
   - **Use instead:** Start with free tier of provider logging (CloudWatch free tier, Vercel logs, etc.)
   - **Example:** You got a $5K grant to build an educational tool. Spending $200/month on monitoring infrastructure is 4% of your runway. Use free CloudWatch Insights until you have real users, then upgrade.
   - **Red flag:** If monitoring costs are >2% of your total monthly budget, defer it.

**[PAUSE]**

**Red flags that you've chosen the wrong approach:**
- 🚩 You're spending more time debugging Prometheus than building features
- 🚩 Your monitoring dashboard has zero viewers in the last 30 days
- 🚩 You don't have anyone on the team comfortable with PromQL
- 🚩 Storage costs are growing faster than query volume
- 🚩 You haven't looked at the dashboard in over a week

If you see these, **stop.** Simplify to logging or switch to managed APM. There's no shame in right-sizing your monitoring approach.

**The honest truth:** I've built dozens of RAG systems. Only 30% needed Prometheus. The rest were better served by simpler approaches until they reached serious scale.

---

<!-- NEW SECTION: DECISION CARD - Added per TVH Framework v2.0 -->

### [32:00] DECISION CARD

**[32:00] [SLIDE: "Decision Card - Production Monitoring with Prometheus + Grafana"]**

Let me summarize everything in one decision framework:

**✅ BENEFIT**
Real-time visibility into performance, costs, and errors with queryable historical data; enables proactive alerting (discover issues before users complain); industry-standard tooling (good for hiring, lots of community resources); cost-effective at scale (zero licensing fees, $200/month vs $500-1000 for managed APM at 5K+ queries/day).

**❌ LIMITATION**
Requires DevOps expertise (PromQL learning curve, storage management, backup strategy); adds operational complexity (now monitoring the monitoring system—4 new services that can fail); alert tuning takes 2-4 weeks to eliminate false positives; storage grows 1-5GB/day requiring capacity planning; not real-time tracing (metrics don't show request flows, only aggregates); memory leaks possible with unbounded label cardinality.

**💰 COST**
Initial: 8-12 hours setup + learning PromQL. Infrastructure: $50-100/month basic (compute), $20-100/month storage (depends on retention and cardinality), grows to $200-300/month at scale. Time ongoing: 2-4 hours/month for alert tuning, storage management, version updates. Maintenance: Quarterly Prometheus upgrades, weekly storage monitoring, dashboard upkeep.

**🤔 USE WHEN**
Query volume >1K/day (makes monitoring complexity worthwhile); team has DevOps capacity (someone comfortable with infrastructure and PromQL); multi-component system (RAG + cache + vector DB + LLM APIs all need monitoring); need historical trend analysis (capacity planning, cost optimization over time); can allocate 12+ hours for initial setup and 2-4 hours/month ongoing.

**🚫 AVOID WHEN**
Query volume <500/day → use simple CloudWatch logs or provider dashboards; no DevOps resources → use DataDog ($31/host) or New Relic instead; prototyping/MVP phase → defer monitoring until product validation; tight budget (<$10K/month runway) → use free provider logging; already using managed cloud (AWS/GCP/Azure) → their native monitoring is sufficient for most cases.

**[PAUSE]** Take a screenshot of this slide—refer back when making monitoring decisions for your next project.

---

<!-- NEW SECTION: PRODUCTION CONSIDERATIONS (Expanded) - Added per TVH Framework v2.0 -->

### [33:00] PRODUCTION CONSIDERATIONS: What Changes at Scale

**[33:00] [SLIDE: "Scaling Production Monitoring"]**

What we built today works great for development and moderate production. Here's what changes when you scale and what to watch out for.

**Scaling Prometheus itself:**

When you hit 10K+ queries/day, monitoring becomes a bottleneck:
- **Storage growth**: Expect 2-5GB/day with our metric set. At 90-day retention, that's 180-450GB. Solution: Use Prometheus remote write to long-term storage (Thanos, Cortex, or S3).
- **Query performance**: With >1M time series, dashboard queries slow down. Solution: Use recording rules to pre-aggregate expensive queries.
- **High availability**: Single Prometheus instance becomes single point of failure. Solution: Run 2+ Prometheus instances with federation.

**Cost at scale with actual numbers:**

Let me be specific about costs as you grow:
- **Development (100 queries/day)**: $50/month (minimal infrastructure)
- **Production (1K queries/day)**: $100-150/month (basic HA setup)
- **Scale (10K queries/day)**: $200-300/month (HA + long-term storage)
- **Enterprise (100K+ queries/day)**: $500-1000/month (clustered setup) BUT still cheaper than managed APM at this scale (which would be $2K-5K/month)

**Break-even point vs alternatives:**
- DataDog: $31/host × 3 hosts = $93/month (simpler but more expensive at scale)
- Prometheus: $200/month at scale but requires DevOps time
- **Break-even**: Around 5K queries/day, if you have DevOps capacity

**Monitoring requirements for the monitoring system:**

Yes, you need to monitor Prometheus itself:
- **Disk usage**: Alert when >70% full (expansion takes time)
- **Query latency**: P95 should be <1s (slow queries indicate cardinality issues)
- **Scrape failures**: Alert if targets unreachable >2 minutes
- **Cardinality growth**: Track time series count (should grow linearly with traffic, not exponentially)

**Storage retention strategy:**

Don't keep everything forever:
- **High-resolution**: 7-15 days (full fidelity for recent debugging)
- **Downsampled**: 90 days (hourly aggregates for trend analysis)
- **Long-term**: 1+ years (daily aggregates for capacity planning)

Use Thanos or Cortex for automatic downsampling and long-term S3 storage.

**We'll cover production deployment patterns, including monitoring infrastructure as code, in Module 3.**

---

### [35:00] RECAP & KEY TAKEAWAYS

**[35:00] [SLIDE: "Key Takeaways"]**

Let's recap what we covered:

**✅ What we learned:**
1. How to instrument RAG systems with comprehensive metrics (latency, cost, errors)
2. Building production dashboards with Prometheus + Grafana
3. Setting up proactive alerting with proper threshold tuning
4. Structured logging for debugging
5. **When NOT to use self-hosted monitoring** (MVP phase, no DevOps, budget constraints)
6. Alternative approaches (managed APM, simple logging)

**✅ What we built:**
Complete production monitoring stack with metrics collection, Grafana dashboards, cost tracking, alerting system, and health checks—ready for Kubernetes deployment.

**✅ What we debugged:**
5 common monitoring failures: cardinality explosion (Prometheus crashes), query timeouts (dashboard issues), memory leaks (unbounded labels), alert fatigue (poor tuning), and missing metrics after deployment.

**⚠️ Critical limitation to remember:**
Monitoring infrastructure itself requires monitoring. You're adding 3-4 new services that need maintenance. Only worthwhile at >1K queries/day with DevOps capacity—otherwise use managed alternatives.

**[36:30] Connecting to next video:**
In M2.4, we'll cover error handling and reliability patterns to make your RAG system bulletproof. This builds directly on monitoring by using the metrics and alerts we set up today to detect failures, plus we'll implement retry logic, circuit breakers, and graceful degradation.

---

### [37:00] CHALLENGES

**[37:00] [SLIDE: "Practice Challenges"]**

Time to practice! Here are three challenges at different levels.

### 🟢 **EASY Challenge** (30-45 minutes)
**Task:** Set up Prometheus and create a basic dashboard showing query count, latency (p95), and cache hit rate for your RAG system.

**Success criteria:**
- [ ] Prometheus scraping metrics from your app every 15s
- [ ] Grafana dashboard with 3 panels (query rate, p95 latency, cache hit %)
- [ ] Dashboard updates in real-time as you run queries

**Hint:** Start with prometheus.yml configuration—make sure your scrape interval matches the metric recording frequency.

---

### 🟡 **MEDIUM Challenge** (1-2 hours)
**Task:** Implement a custom "response quality" metric that evaluates RAG responses using semantic similarity (embedding-based) or another LLM, then graph quality over time and alert when quality drops below threshold.

**Success criteria:**
- [ ] Custom metric `rag_response_quality_score` tracked in Prometheus
- [ ] Grafana panel showing quality trend over last 24 hours
- [ ] Alert configured to fire when quality <0.7 for >5 minutes
- [ ] At least 100 test queries to populate data

**Hint:** Use a lightweight embedding model (all-MiniLM-L6-v2) to compare generated response to expected answer or retrieved context.

---

### 🔴 **HARD Challenge** (3-4 hours, portfolio-worthy)
**Task:** Build an anomaly detection system that uses statistical methods (z-score, moving average, etc.) to automatically detect unusual patterns in your metrics (latency spikes, cost anomalies, error rate changes) and create adaptive alerts that adjust thresholds based on historical patterns.

**Success criteria:**
- [ ] Statistical baseline calculated from last 7 days of data
- [ ] Anomaly detection running every minute (3+ sigma deviations flagged)
- [ ] Slack/Discord notification when anomaly detected with context
- [ ] Dashboard showing normal range vs actual values with anomaly markers
- [ ] False positive rate <10% tested over 24 hours

**This is portfolio-worthy!** Share your solution in Discord when complete—especially your anomaly detection algorithm and how you tuned it.

**No hints - figure it out!** (Solutions will be provided in 48 hours)

---

### [38:00] ACTION ITEMS

**[38:00] [SLIDE: "Before Next Video"]**

**Before moving to M2.4, complete these:**

**REQUIRED:**
1. [ ] Install Prometheus and Grafana locally (Docker Compose recommended)
2. [ ] Instrument your RAG pipeline with at least 5 metrics
3. [ ] Create dashboard with 4-6 key metrics (latency, cache hit rate, cost, errors)
4. [ ] Set up at least 2 alert rules and test by simulating failures
5. [ ] Complete Easy challenge

**RECOMMENDED:**
1. [ ] Read Prometheus best practices docs: https://prometheus.io/docs/practices/naming/
2. [ ] Set up structured logging in your RAG app (JSON format)
3. [ ] Add health check endpoints (/health, /ready) to your API
4. [ ] Experiment with recording rules for complex queries
5. [ ] Review your metric cardinality (keep it <10K time series)

**OPTIONAL:**
1. [ ] Research Thanos or Cortex for long-term storage
2. [ ] Build a custom exporter for your vector database metrics
3. [ ] Try DataDog free trial to compare to self-hosted Prometheus

**Estimated time investment:** 2-3 hours for required items, 4-5 hours with recommended

---

### [39:00] WRAP-UP

**[39:00] [SLIDE: "Thank You"]**

Great job making it through! This was a dense video covering production infrastructure—these patterns separate hobby projects from production systems.

**Remember:**
- Monitoring is powerful but adds complexity—choose the right approach for your scale
- Self-hosted Prometheus is best for >1K queries/day with DevOps capacity
- For MVPs and small projects, simpler alternatives (CloudWatch, DataDog) are often better
- Always monitor the monitoring system itself

**If you get stuck:**
1. Review the "When This Breaks" section (timestamp: 25:00)
2. Check Prometheus documentation for metric cardinality issues
3. Post in Discord with your Grafana dashboard screenshot and question
4. Attend office hours Thursdays at 3pm PT

**Next video: M2.4 - Error Handling & Reliability Patterns** where we'll make your RAG system bulletproof with retries, circuit breakers, and graceful degradation. This builds on the monitoring we set up today!

[SLIDE: End Card with Course Branding]

---

# PRODUCTION NOTES

## Pre-Recording Checklist
- [ ] Prometheus and Grafana running locally with test data
- [ ] Demo RAG system instrumented with metrics
- [ ] Can reproduce all 5 failures on demand
- [ ] Grafana dashboards pre-configured
- [ ] Alert rules configured and tested
- [ ] Terminal history cleared
- [ ] Example cardinality explosion script ready
- [ ] DataDog account for comparison screenshots

## During Recording
- Show actual Prometheus crashing from cardinality explosion (not just talking about it)
- Demonstrate alert storm in real Slack channel
- Zoom in on Grafana queries to show PromQL syntax clearly
- Pause after Decision Card slide for 5+ seconds (students need to screenshot)
- Be honest about time investment—don't minimize setup complexity

## Post-Recording
- Verify all 5 failure scenarios are clearly demonstrated
- Check that Decision Card is on screen long enough (5+ seconds)
- Ensure Alternative Solutions comparison table is readable
- Verify timestamps match (might need to adjust if recording ran long)

---

**ENHANCEMENT SUMMARY:**

Added 6 new sections totaling ~12 minutes:
1. **The Problem** (2 min) - Live incident without monitoring
2. **Reality Check** (2.5 min) - Honest infrastructure costs
3. **Alternative Solutions** (2.5 min) - 3 approaches with decision framework
4. **When This Breaks** (5 min) - 5 failure scenarios with fixes
5. **When NOT to Use** (2 min) - 3 anti-pattern scenarios
6. **Decision Card** (1 min) - Complete 5-field framework
7. **Expanded Production Considerations** (2 min) - Scaling specifics

**Total duration:** 38-40 minutes (was 20 min, now with all honest teaching framework)

**Compliance:** ✅ All 6 mandatory TVH v2.0 sections now present