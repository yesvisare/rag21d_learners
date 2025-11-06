# M3.4 — Load Testing & Scaling for RAG Systems

Complete workspace for learning and implementing load testing with Locust, identifying bottlenecks, and scaling strategies.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API endpoint
# TARGET_URL=http://localhost:8000
```

### 2. Run Your First Load Test

**Smoke test** (quick health check):
```bash
locust -f locustfile.py --host=http://localhost:8000 \
  --users 10 --spawn-rate 2 --run-time 2m --headless
```

**Load test** (realistic traffic):
```bash
locust -f locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 10m --headless
```

**Web UI mode** (interactive):
```bash
locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 in browser
```

### 3. Analyze Results

Locust outputs:
- **Console**: Real-time metrics (RPS, p50/p95/p99 latency, error rate)
- **CSV**: `--csv=results/test_name` exports detailed statistics
- **HTML**: `--html=results/report.html` generates visual report

---

## Test Scenarios

All scenarios are pre-configured in `locustfile.py`.

| Scenario | Users | Spawn Rate | Duration | Purpose |
|----------|-------|------------|----------|---------|
| **Smoke** | 10 | 2/sec | 2 min | Health check before deployment |
| **Load** | 100 | 10/sec | 10 min | Weekly baseline testing |
| **Stress** | 1000 | 50/sec | 15 min | Find breaking point |
| **Spike** | 500 | 500/sec | 5 min | Sudden traffic surge |
| **Soak** | 50 | 5/sec | 4 hours | Memory leak detection |

### Running Specific Scenarios

```bash
# Smoke test
locust -f locustfile.py --host=YOUR_API \
  --users 10 --spawn-rate 2 --run-time 2m --headless

# Load test
locust -f locustfile.py --host=YOUR_API \
  --users 100 --spawn-rate 10 --run-time 10m --headless

# Stress test
locust -f locustfile.py --host=YOUR_API \
  --users 1000 --spawn-rate 50 --run-time 15m --headless

# Spike test
locust -f locustfile.py --host=YOUR_API \
  --users 500 --spawn-rate 500 --run-time 5m --headless

# Soak test (4 hours)
locust -f locustfile.py --host=YOUR_API \
  --users 50 --spawn-rate 5 --run-time 4h --headless
```

---

## Interpreting Metrics

### Key Metrics

| Metric | What It Means | Target |
|--------|---------------|--------|
| **p50** | Median response time - typical user experience | <500ms |
| **p95** | 95% of requests faster - most users' experience | <2s |
| **p99** | 99% of requests faster - worst case | <5s |
| **RPS** | Requests per second - throughput | 20-100+ |
| **Error Rate** | % of failed requests | <1% |

### Reading Locust Output

```
Type  Name       # reqs  # fails  Avg    Min  Max   Median  p95   p99   req/s
POST  /query     1523    12       856    234  3421  780     1850  2340  12.3
```

**Interpretation**:
- **12 failures** out of 1523 = 0.8% error rate ✅ (acceptable)
- **p95: 1850ms** = 95% of users wait <1.85s ✅
- **p99: 2340ms** = worst 1% wait ~2.3s ⚠️ (acceptable but room for improvement)
- **12.3 req/s** = system handles 12 queries per second

### Common Patterns

**Pattern 1: Rate limiting**
```
# reqs   # fails   p99      Error
2000     0         1200ms   -
3000     500       8500ms   429 Too Many Requests
```
→ **Diagnosis**: External API rate limit hit

**Pattern 2: Memory leak (soak test)**
```
Time     p95      Memory
0-1h     800ms    1.2GB
2-3h     1800ms   2.5GB  ← Degradation
```
→ **Diagnosis**: Memory leak causing slowdown

---

## Cost vs Time Trade-offs

### Time Investment

| Activity | Initial | Maintenance |
|----------|---------|-------------|
| Setup locustfile | 4-6 hours | - |
| Configure CI/CD | 2-3 hours | 30 min/sprint |
| Baseline testing | 1 hour | 1 hour/week |
| Analyze results | 1-2 hours | 1 hour/week |
| **Total** | **8-12 hours** | **2-4 hours/sprint** |

### When It's Worth It

✅ **YES** - Load test when:
- User base growing >20% monthly
- SLA requirements in contracts (e.g., p95 <1s)
- Planning infrastructure upgrades (data-driven decisions)
- Pre-launch (marketing campaign, new feature)

❌ **NO** - Skip when:
- <100 daily users, no growth plans
- Staging environment doesn't match production
- MVP/early-stage (architecture unstable)
- Other testing types more urgent (security, integration)

### ROI Example

**Scenario**: Growing SaaS with 1000 daily users

**Investment**:
- Initial: 8 hours setup
- Ongoing: 2 hours/week

**Value**:
- Prevented 2 outages (saved $5000 in lost revenue)
- Identified caching opportunity (saved $150/month on API costs)
- Data-driven scaling (avoided over-provisioning by $100/month)

**Break-even**: 2 months

---

## Bottleneck Diagnosis

### Quick Checklist

Run load test, then check:

1. **High error rate (>5%)**
   → Check logs for error type (rate limit? connection pool? timeout?)

2. **High p99 (>5s)**
   → Check CPU/memory during test (infrastructure bottleneck?)

3. **Errors at specific user threshold**
   → Identify capacity limit (e.g., breaks at 125 users)

4. **Latency increases over time (soak test)**
   → Memory leak or connection pool leak

### Common Fixes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| 429 errors | OpenAI rate limit | Add caching, batch requests |
| QueuePool errors | DB connection pool exhausted | Increase `pool_size` |
| High CPU (>80%) | CPU bottleneck | Vertical scale or optimize code |
| High memory (>85%) | Memory leak | Profile code, fix leak |
| Slow DB queries | Missing indexes | Add indexes on frequent queries |

---

## Scaling Strategies

### The Hierarchy (cheapest to most expensive)

1. **Optimize code** (free, high impact)
   - Fix N+1 queries, add database indexes
   - Impact: 5-10x improvement

2. **Add caching** ($10-50/month)
   - Redis for query results, embeddings
   - Impact: 10-100x speedup, 80% cost savings

3. **Batch operations** (free)
   - Batch embeddings, database queries
   - Impact: 5-10x efficiency

4. **Vertical scaling** ($40-200/month)
   - Increase CPU, RAM on single instance
   - Impact: 2-4x capacity

5. **Horizontal scaling** ($80-500/month)
   - Multiple instances + load balancer
   - Impact: Unlimited capacity

6. **Auto-scaling** (varies)
   - Dynamic instance scaling based on metrics
   - Impact: Cost efficiency + elasticity

---

## Project Structure

```
.
├── locustfile.py              # Main load test implementation
├── scaling_notes.md           # Detailed scaling reference guide
├── M3_4_Load_Testing_and_Scaling.ipynb  # Interactive tutorial
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
└── tests_locustfile.py        # Basic tests for locustfile
```

---

## Troubleshooting

### Locust won't start

```bash
# Check Python version (requires 3.8+)
python --version

# Reinstall dependencies
pip install --upgrade locust==2.31.6
```

### Connection errors during test

```bash
# Check if target API is running
curl http://localhost:8000/health

# Verify .env TARGET_URL is correct
cat .env
```

### High error rate in test

- **Check API logs** during test (are there actual errors?)
- **Reduce spawn rate** (too many users too fast?)
- **Increase timeout** in locustfile.py if needed

### Locust crashes under high load

- **Use distributed mode** (run multiple Locust workers)
- **Reduce spawn rate** (slower ramp-up)
- **Increase system resources** for Locust itself

---

## Advanced Usage

### Export CSV for Analysis

```bash
locust -f locustfile.py --host=YOUR_API \
  --users 100 --spawn-rate 10 --run-time 10m \
  --csv=results/load_test --headless
```

Generates:
- `results/load_test_stats.csv` - Summary stats
- `results/load_test_stats_history.csv` - Time-series data
- `results/load_test_failures.csv` - Error details

### Generate HTML Report

```bash
locust -f locustfile.py --host=YOUR_API \
  --users 100 --spawn-rate 10 --run-time 10m \
  --html=results/report.html --headless
```

### Distributed Load Testing

For massive scale (1000+ users):

```bash
# Master node
locust -f locustfile.py --master

# Worker nodes (run on multiple machines)
locust -f locustfile.py --worker --master-host=<master-ip>
```

---

## Resources

- **Locust Documentation**: https://docs.locust.io
- **scaling_notes.md**: Comprehensive scaling strategies
- **M3_4_Load_Testing_and_Scaling.ipynb**: Interactive tutorial with examples

---

## Learning Path

1. **Start**: Read `M3_4_Load_Testing_and_Scaling.ipynb` sections 1-2
2. **Practice**: Run smoke test against your API
3. **Learn**: Read sections 3-4 (metrics, bottlenecks)
4. **Apply**: Run load test, identify bottleneck
5. **Scale**: Implement fixes from section 5
6. **Decide**: Use section 6 framework for future tests

---

## License

MIT License - See repository root for details.
