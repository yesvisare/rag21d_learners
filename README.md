# M3.4 — Load Testing & Scaling for RAG Systems

Complete workspace for learning and implementing load testing with Locust, identifying bottlenecks, and scaling strategies.

---

## Learning Arc

### Purpose
This module teaches you to measure, diagnose, and scale production RAG systems using load testing. You'll learn to identify performance bottlenecks, interpret latency metrics (p50/p95/p99), and apply cost-effective scaling strategies before throwing money at infrastructure.

### Concepts Covered
- **Load testing fundamentals**: Smoke, load, stress, spike, and soak tests
- **Locust framework**: Task-weighted user simulation, realistic traffic patterns
- **Performance metrics**: Throughput (RPS), latency percentiles (p50/p95/p99), error rates
- **Bottleneck diagnosis**: Application code, external APIs, infrastructure constraints
- **Scaling strategies**: Caching, batching, connection pooling, vertical/horizontal scaling, auto-scaling (HPA)
- **Decision frameworks**: When to load test vs. when to skip

### After Completing This Module
You will be able to:
1. Implement production-grade load tests with Locust for RAG systems
2. Interpret performance metrics to identify capacity limits and user experience
3. Diagnose bottlenecks across application code, external services, and infrastructure
4. Apply systematic scaling strategies (optimize → cache → batch → scale)
5. Make data-driven infrastructure decisions with ROI calculations
6. Determine when load testing adds value vs. when to use alternative approaches

### Context in Track
**Module 3 (Production RAG)** focuses on deploying RAG systems to production. Previous modules covered containerization (M3.1), cloud deployment (M3.2), and API security (M3.3). This module completes the production readiness arc by teaching you to validate capacity, identify breaking points, and scale efficiently—essential before launching to real users.

---

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
```powershell
powershell ./scripts/run_smoke_test.ps1
```

**Load test** (realistic traffic):
```powershell
powershell ./scripts/run_load_test.ps1
```

**Web UI mode** (interactive):
```powershell
powershell ./scripts/run_web_ui.ps1
# Open http://localhost:8089 in browser
```

### 3. Analyze Results

Locust outputs:
- **Console**: Real-time metrics (RPS, p50/p95/p99 latency, error rate)
- **CSV**: Exported to `results/` directory (e.g., `results/smoke_stats.csv`)
- **HTML**: Visual reports in `results/` (e.g., `results/smoke.html`)

---

## Configuration via .env

All test parameters can be configured via environment variables in `.env`:

```bash
# Target host (REQUIRED - verify before running!)
LOADTEST_HOST=http://localhost:8000

# Smoke Test (health check)
SMOKE_USERS=10
SMOKE_SPAWN_RATE=2
SMOKE_RUNTIME=2m

# Load Test (realistic traffic)
LOAD_USERS=100
LOAD_SPAWN_RATE=10
LOAD_RUNTIME=10m

# Stress Test (find breaking point)
STRESS_USERS=1000
STRESS_SPAWN_RATE=50
STRESS_RUNTIME=15m

# Spike Test (sudden surge)
SPIKE_USERS=500
SPIKE_SPAWN_RATE=500
SPIKE_RUNTIME=5m

# Soak Test (memory leak detection)
SOAK_USERS=50
SOAK_SPAWN_RATE=5
SOAK_RUNTIME=4h
```

**Override host per-test**:
```powershell
# Test against different host
powershell ./scripts/run_load_test.ps1 -HOST http://127.0.0.1:9000

# Test against staging
powershell ./scripts/run_smoke_test.ps1 -HOST https://staging-api.example.com
```

---

## Artifacts

All test scripts automatically generate artifacts in the `results/` directory:

| Test | CSV Files | HTML Report |
|------|-----------|-------------|
| Smoke | `results/smoke_stats.csv`, `results/smoke_stats_history.csv`, `results/smoke_failures.csv` | `results/smoke.html` |
| Load | `results/load_stats.csv`, `results/load_stats_history.csv`, `results/load_failures.csv` | `results/load.html` |
| Stress | `results/stress_stats.csv`, `results/stress_stats_history.csv`, `results/stress_failures.csv` | `results/stress.html` |
| Spike | `results/spike_stats.csv`, `results/spike_stats_history.csv`, `results/spike_failures.csv` | `results/spike.html` |
| Soak | `results/soak_stats.csv`, `results/soak_stats_history.csv`, `results/soak_failures.csv` | `results/soak.html` |

**CSV Files**:
- `*_stats.csv`: Aggregated statistics (avg, median, p95, p99, RPS)
- `*_stats_history.csv`: Time-series data for plotting trends
- `*_failures.csv`: Detailed error logs

**HTML Reports**:
- Interactive charts of RPS, response times, and error rates
- Downloadable for sharing with team

**Note**: The `results/` directory is in `.gitignore` to prevent committing test data.

---

## Safety & Guardrails

### ⚠️ CRITICAL: Verify Target Before Running

**NEVER load test production systems without explicit approval.**

Load testing generates significant traffic that can:
- Overwhelm production systems
- Trigger rate limits
- Incur unexpected costs (API usage, infrastructure scaling)
- Impact real users

### Pre-Flight Checklist

Before running ANY load test:

1. ✅ **Verify HOST**: Check that `LOADTEST_HOST` points to the correct environment
   ```bash
   # Windows
   echo $env:LOADTEST_HOST

   # Linux/Mac
   echo $LOADTEST_HOST
   ```

2. ✅ **Confirm environment**: Ensure you're targeting staging/test, NOT production

3. ✅ **Get approval**: For production testing, obtain written approval from:
   - Engineering lead
   - Infrastructure team
   - Product owner (if customer-facing)

4. ✅ **Start small**: Always run smoke test first before load/stress tests

5. ✅ **Monitor costs**: Watch for unexpected API charges or infrastructure scaling

### Safe Testing Practices

- **Local development**: Use `http://localhost:8000` for initial testing
- **Staging environment**: Use dedicated staging URL (e.g., `https://staging.example.com`)
- **Production**: Only with approval, during off-peak hours, with monitoring

### Emergency Stop

If test is causing issues:
1. Press `Ctrl+C` in terminal (kills Locust immediately)
2. Check application logs for errors
3. Verify system recovery before re-running

---

## Test Scenarios

All scenarios are pre-configured in `locustfile.py` with helper scripts in `scripts/`.

| Scenario | Users | Spawn Rate | Duration | Purpose | Script |
|----------|-------|------------|----------|---------|--------|
| **Smoke** | 10 | 2/sec | 2 min | Health check before deployment | `run_smoke_test.ps1` |
| **Load** | 100 | 10/sec | 10 min | Weekly baseline testing | `run_load_test.ps1` |
| **Stress** | 1000 | 50/sec | 15 min | Find breaking point | `run_stress_test.ps1` |
| **Spike** | 500 | 500/sec | 5 min | Sudden traffic surge | `run_spike_test.ps1` |
| **Soak** | 50 | 5/sec | 4 hours | Memory leak detection | `run_soak_test.ps1` |

### Running Specific Scenarios

**Default (uses .env configuration)**:
```powershell
# Smoke test
powershell ./scripts/run_smoke_test.ps1

# Load test
powershell ./scripts/run_load_test.ps1

# Stress test
powershell ./scripts/run_stress_test.ps1

# Spike test
powershell ./scripts/run_spike_test.ps1

# Soak test (4 hours)
powershell ./scripts/run_soak_test.ps1

# Web UI mode
powershell ./scripts/run_web_ui.ps1
```

**Override host per test**:
```powershell
# Test against different port
powershell ./scripts/run_smoke_test.ps1 -HOST http://127.0.0.1:9000

# Test against staging environment
powershell ./scripts/run_load_test.ps1 -HOST https://staging-api.example.com
```

**Note**: Scripts read from `.env` with fallback to `http://localhost:8000`. Use `-HOST` parameter to override for a specific test run.

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
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
├── locustfile.py              # Main load test implementation
├── notebooks/
│   ├── M3_4_Load_Testing_and_Scaling.ipynb  # Interactive tutorial
│   └── scaling_notes.md       # Detailed scaling reference guide
├── tests/
│   ├── __init__.py
│   └── test_locustfile.py     # Basic tests for locustfile
└── scripts/
    ├── run_smoke_test.ps1     # Smoke test (10 users, 2 min)
    ├── run_load_test.ps1      # Load test (100 users, 10 min)
    ├── run_stress_test.ps1    # Stress test (1000 users, 15 min)
    ├── run_spike_test.ps1     # Spike test (500 users, 5 min)
    ├── run_soak_test.ps1      # Soak test (50 users, 4 hours)
    └── run_web_ui.ps1         # Interactive web UI
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
- **notebooks/scaling_notes.md**: Comprehensive scaling strategies
- **notebooks/M3_4_Load_Testing_and_Scaling.ipynb**: Interactive tutorial with examples

---

## Learning Path

1. **Start**: Read `notebooks/M3_4_Load_Testing_and_Scaling.ipynb` sections 1-2
2. **Practice**: Run smoke test against your API (`powershell ./scripts/run_smoke_test.ps1`)
3. **Learn**: Read sections 3-4 (metrics, bottlenecks)
4. **Apply**: Run load test, identify bottleneck
5. **Scale**: Implement fixes from section 5
6. **Decide**: Use section 6 framework for future tests

---

## License

MIT License - See LICENSE file for details.
