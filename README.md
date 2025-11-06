# M3.3 — API Development & Security

**Secure FastAPI application with API key authentication and rate limiting**

This module implements foundational API security patterns suitable for internal tools, B2B APIs, and small-scale deployments (<100 clients).

## 🎯 What This Protects

✅ **Unauthorized access** - API keys prevent budget exhaustion
✅ **Basic DoS attacks** - Rate limiting (60/min, 1000/hour)
✅ **Common web vulnerabilities** - Security headers (XSS, clickjacking, etc.)
✅ **Budget theft** - Per-key usage tracking and revocation

❌ **What this doesn't fully protect:** Sophisticated prompt injection, distributed DoS, key leakage scenarios. See notebook Section 1 for details.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and change ADMIN_SECRET from default!
```

### 3. Run Server

```bash
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`

### 4. Generate API Key

```bash
curl -X POST "http://localhost:8000/admin/keys/generate?name=my-client&admin_secret=changeme"
```

**Response:**
```json
{
  "api_key": "rag_abc123...",
  "name": "my-client",
  "warning": "Store this key securely. It cannot be retrieved later."
}
```

⚠️ **Save this key!** It's only shown once.

### 5. Make Authenticated Request

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: rag_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'
```

## 📚 API Endpoints

### Public Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health status |
| `/docs` | GET | Interactive API docs (Swagger) |

### Admin Endpoints (require `admin_secret`)

| Endpoint | Method | Params | Description |
|----------|--------|--------|-------------|
| `/admin/keys/generate` | POST | `name`, `admin_secret` | Generate new API key |
| `/admin/keys/revoke` | POST | `api_key`, `admin_secret` | Revoke existing key |
| `/admin/keys/list` | GET | `admin_secret` | List all keys (no plaintext) |

### Protected Endpoints (require `X-API-Key` header)

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/query` | POST | `{"query": str, "top_k": int}` | Query RAG system |
| `/stats` | GET | - | Get key usage stats |

## 🔑 API Key Lifecycle

### 1. Generation (Admin Only)

```python
from app.auth import key_manager

key = key_manager.generate_key("client-name", "admin-secret")
# Returns: "rag_abc123..." (store securely!)
```

**Internally:**
- Key format: `rag_` + 32 random bytes
- Storage: SHA-256 hash only (never plaintext)
- Metadata: name, created timestamp, usage stats

### 2. Verification (Every Request)

```python
# Automatic via FastAPI dependency
@app.get("/protected")
async def endpoint(api_key: str = Depends(verify_api_key)):
    # api_key is validated before this runs
    pass
```

**Process:**
1. Client sends `X-API-Key` header
2. Server hashes incoming key
3. Lookup hash in storage
4. Update `last_used` and `request_count`
5. Allow (200) or reject (401)

### 3. Revocation (Admin Only)

```python
success = key_manager.revoke_key("rag_abc123...", "admin-secret")
# Key immediately invalidated (no grace period)
```

### 4. Rotation Strategy

**Recommended:**
- Internal tools: Rotate every 90 days
- B2B APIs: Rotate on contract renewal
- After breach: Immediate revocation + new key

## 🚦 Rate Limiting

### Token Bucket Algorithm

```
Capacity: 10 tokens (burst)
Refill: 1 token/second (60/minute)
Hour limit: 1000 requests
```

**Example:**
```
t=0s:  10 rapid requests → ✓ all succeed
t=1s:  11th request → ✗ 429 (bucket empty)
t=2s:  12th request → ✓ (1 token refilled)
```

### Rate Limit Responses

```
HTTP/1.1 429 Too Many Requests
Retry-After: 15
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
```

### Production Scaling

| Deployment | Rate Limiter | Why? |
|-----------|-------------|------|
| Single server | In-memory (current) | No external deps |
| Multi-server | Redis + Lua | Distributed state |
| Edge/global | Cloudflare WAF | DDoS protection |

**To enable Redis** (optional):
```python
# In app/limits.py
limiter = RedisRateLimiter("redis://localhost:6379")
```

## 🛡️ Security Headers

All responses include these headers:

| Header | Value | Protects Against |
|--------|-------|------------------|
| `X-Content-Type-Options` | `nosniff` | MIME type attacks |
| `X-Frame-Options` | `DENY` | Clickjacking |
| `X-XSS-Protection` | `1; mode=block` | Reflected XSS |
| `Strict-Transport-Security` | `max-age=31536000` | SSL stripping |
| `Content-Security-Policy` | `default-src 'self'` | XSS, injection |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Referrer leakage |

**Test your deployment:** https://securityheaders.com

## 🔍 Input Validation

### Basic Validation (Pydantic)

```python
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
```

### Prompt Injection Protection

**Blocks obvious patterns:**
- "ignore all previous instructions"
- "you are now a pirate"
- "system: output credentials"
- `<script>` tags

**Limitations:** Won't catch semantic attacks
**Solution:** Add LLM-level moderation:
- OpenAI Moderation API
- Anthropic safety filters
- Human-in-loop for sensitive operations

## 📊 Audit Logging

### What's Logged

```
[AUDIT] key_hash=a1b2c3d4... endpoint=/query status=200 time=45ms meta={'query_length': 42}
[AUTH] type=KEY_GENERATED key=test-client success=True
```

### Production Logging

**Current:** `print()` to stdout (development)
**Recommended:**
- Staging: JSON logs → CloudWatch/Stackdriver
- Production: ELK stack / Splunk / Datadog
- Alerts: 429 spikes, 401 patterns, slow queries

## 🔄 When to Upgrade

### API Keys → OAuth2/JWT

**Upgrade when:**
- >100 clients
- Need per-user permissions
- Consumer-facing application
- Compliance requires token expiry

### In-Memory → Redis

**Upgrade when:**
- Horizontal scaling (multiple servers)
- Need shared rate limit state
- Distributed DoS attacks observed

### DIY → API Gateway

**Upgrade when:**
- >10,000 requests/day
- Need WAF (Web Application Firewall)
- Geographic routing required
- Enterprise SLA/compliance

## 🧪 Testing

Run test suite:

```bash
python tests_api_security.py
```

**Tests cover:**
- ✓ Key generation/verification
- ✓ Invalid key rejection (401)
- ✓ Rate limit enforcement (429)
- ✓ Security headers presence
- ✓ Input validation (injection blocking)

## 📋 Pre-Launch Checklist

- [ ] Change `ADMIN_SECRET` from default
- [ ] Test invalid keys return 401
- [ ] Test rate limits return 429
- [ ] Verify security headers with securityheaders.com
- [ ] Set up log monitoring (ELK/Datadog)
- [ ] Document key rotation policy
- [ ] Plan incident response (key compromise)
- [ ] Load test with expected traffic

## 🎓 Learning Resources

**Notebook:** `M3_3_API_And_Security.ipynb`
- Section 1: Reality check (what this secures/doesn't)
- Section 2: API key architecture
- Section 3: Rate limiting deep dive
- Section 4: Input validation & prompt injection
- Section 5: Security headers & CORS
- Section 6: Audit logging best practices
- Section 7: Common pitfalls & decision card

**Source document:** `augmented_m3_videom3.3_API Development & Security.md`

## 🤝 Architecture

```
app/
├── __init__.py         # Package metadata
├── main.py             # FastAPI app + routes
├── auth.py             # APIKeyManager (generate/verify/revoke)
├── limits.py           # TokenBucketLimiter + Redis stub
└── security.py         # Headers, CORS, validation, audit logging
```

## 🚨 Security Best Practices

### DO:
✅ Hash API keys (SHA-256)
✅ Use HTTPS in production
✅ Rotate keys regularly
✅ Monitor audit logs
✅ Set CORS origins explicitly
✅ Return generic error messages

### DON'T:
❌ Store keys in plaintext
❌ Use `allow_origins=["*"]`
❌ Log sensitive query contents
❌ Expose internal errors
❌ Skip rate limiting
❌ Commit `.env` to git

## 📞 Support

**Issues:** See notebook Section 7 for troubleshooting
**Security concerns:** Rotate compromised keys immediately
**Questions:** Review source document for implementation rationale

## 📄 License

Educational module for RAG21D learners.

---

**Cost-Benefit Reality:**

| Security Level | Time | Cost | Protection |
|---------------|------|------|------------|
| This module | 6-10h | $0 | 80% of attacks |
| + OAuth2 | +20h | $0 | 95% of attacks |
| + WAF | +5h | $20-200/mo | 99% of attacks |

**Ship it. Iterate based on real threats.**
