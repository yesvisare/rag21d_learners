# M3.3 — API Development & Security

**A practical guide to securing RAG APIs with authentication, rate limiting, and production-ready patterns**

---

## 🎯 Purpose

This module teaches you how to build **production-grade API security** for RAG applications. You'll learn to protect against unauthorized access, budget exhaustion, DoS attacks, and prompt injection—all without over-engineering.

**Why this matters:**
A single unprotected API endpoint can drain your $1000/month LLM budget in hours. This module shows you how to implement security layers that stop 80% of real-world attacks with 6-10 hours of work.

## 📚 Concepts Covered

### Core Security Patterns
- **API Key Authentication** with SHA-256 hashing (never store plaintext)
- **Token Bucket Rate Limiting** (burst + sustained limits)
- **Input Validation** with Pydantic + prompt injection defense
- **Security Headers** (XSS, clickjacking, HSTS, CSP)
- **CORS Configuration** for cross-origin protection
- **Audit Logging** for forensics and compliance

### Real-World Trade-offs
- When to use API keys vs OAuth2/JWT
- In-memory vs Redis-backed rate limiting
- Pattern matching vs LLM-level moderation
- Development vs production security posture

### Decision Framework
- **"Is This Secure Enough?"** checklist
- Cost-benefit analysis (time, money, protection)
- Migration path: API keys → OAuth2 → API Gateway

## ✅ After Completing This Module

You will be able to:

✅ Implement secure API key generation and verification (SHA-256 hashing)
✅ Configure rate limiting to prevent budget exhaustion (60/min, 1000/hour)
✅ Validate inputs and defend against basic prompt injection
✅ Apply security headers to mitigate common web vulnerabilities
✅ Set up audit logging for compliance and incident response
✅ Make informed decisions about when to upgrade security layers

**Deliverable:** A FastAPI application with production-ready security suitable for internal tools and B2B APIs (<100 clients).

## 🔍 Context

### Where This Fits in the RAG Pipeline

```
User Request
    ↓
[API SECURITY LAYER] ← This module
    ├─ API Key Auth
    ├─ Rate Limiting
    ├─ Input Validation
    └─ Audit Logging
    ↓
RAG Application
    ├─ Document Retrieval
    ├─ LLM Generation
    └─ Response Formatting
    ↓
Secured Response
```

### Prerequisites
- Basic Python knowledge
- Understanding of HTTP/REST APIs
- Familiarity with FastAPI (or similar frameworks)

### What This Module Does NOT Cover
- OAuth2/JWT implementation (upgrade path explained)
- Database-backed key storage (in-memory for simplicity)
- Advanced prompt injection defenses (LLM-level moderation recommended)
- Distributed DoS protection (Cloudflare/WAF recommended)

---

## 📂 Files in This Module

```
M3_3_API_And_Security/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Python/IDE ignores
├── .env.example                 # Environment config template
├── requirements.txt             # Python dependencies
├── app.py                       # FastAPI application entry point
├── src/
│   └── m3_3_api_security/
│       ├── __init__.py          # Package metadata
│       ├── auth.py              # API key management (SHA-256)
│       ├── limits.py            # Token bucket rate limiter
│       └── security.py          # Headers, CORS, validation
├── notebooks/
│   └── M3_3_API_And_Security.ipynb  # Interactive learning (7 sections)
├── tests/
│   ├── __init__.py
│   └── test_api_security.py    # Pytest test suite (11 tests)
└── scripts/
    └── run_local.ps1            # PowerShell launcher
```

---

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

### 3. Run the API Server

**PowerShell (recommended):**
```powershell
powershell ./scripts/run_local.ps1
```

**Or manually:**
```bash
# Linux/Mac
export PYTHONPATH=$PWD
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Windows CMD
set PYTHONPATH=%CD%
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`

### 4. Generate an API Key

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

### 5. Make an Authenticated Request

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: rag_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "Mock response for: What is RAG?",
  "sources": [
    {"text": "Mock source 1", "score": 0.95},
    {"text": "Mock source 2", "score": 0.87}
  ],
  "rate_limit": {
    "tokens_remaining": 9,
    "hour_remaining": 999
  }
}
```

---

## 🧪 How to Test

Run the full test suite with pytest:

```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_api_security.py::test_api_key_generation PASSED
tests/test_api_security.py::test_api_key_verification PASSED
tests/test_api_security.py::test_key_usage_tracking PASSED
tests/test_api_security.py::test_key_revocation PASSED
tests/test_api_security.py::test_rate_limiting_burst PASSED
tests/test_api_security.py::test_rate_limiting_refill PASSED
tests/test_api_security.py::test_input_validation_valid PASSED
tests/test_api_security.py::test_input_validation_injection PASSED
tests/test_api_security.py::test_input_validation_limits PASSED
tests/test_api_security.py::test_security_headers PASSED
tests/test_api_security.py::test_key_isolation PASSED

================================ 11 passed in 2.34s ================================
```

### Running Individual Tests

```bash
# Test only authentication
pytest tests/test_api_security.py::test_api_key_generation -v

# Test rate limiting
pytest tests/test_api_security.py::test_rate_limiting_burst -v
```

---

## 📖 Learning Path

Work through the notebook incrementally for hands-on learning:

### Notebook: `notebooks/M3_3_API_And_Security.ipynb`

**Section 1: Reality Check**
Understand what this implementation secures (and doesn't). Learn when to upgrade.

**Section 2: API Key Authentication**
Generate, verify, and revoke keys. Never store plaintext (SHA-256 hashing).

**Section 3: Rate Limiting**
Token bucket algorithm for burst + sustained limits. Prevent budget exhaustion.

**Section 4: Input Validation**
Pydantic validation + prompt injection defense. Learn limitations.

**Section 5: Security Headers**
Apply 6 critical headers (XSS, clickjacking, CSP, HSTS). Configure CORS.

**Section 6: Audit Logging**
Track requests, auth events, and anomalies for forensics.

**Section 7: Common Pitfalls**
Avoid the top 5 mistakes. Decision card for choosing auth methods.

---

## 🔑 API Endpoints

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

---

## 🛡️ Security Features

### 1. API Key Authentication
- **Format**: `rag_` + 32 random bytes
- **Storage**: SHA-256 hash only (never plaintext)
- **Lifecycle**: Generate → Verify → Revoke
- **Metadata**: Tracks creation time, last used, request count

### 2. Rate Limiting (Token Bucket)
- **Burst**: 10 rapid requests allowed
- **Sustained**: 60 requests/minute (1/second)
- **Hourly Cap**: 1000 requests/hour
- **Response**: 429 with `Retry-After` header

### 3. Input Validation
- **Length**: 1-500 characters
- **Type**: Pydantic validation
- **Injection**: Blocks obvious patterns ("ignore all previous instructions")
- **Limitation**: Semantic attacks require LLM-level moderation

### 4. Security Headers (6 critical headers)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`
- `Referrer-Policy: strict-origin-when-cross-origin`

### 5. CORS Configuration
- **Default**: Localhost only
- **Production**: Whitelist exact origins (no wildcards)

### 6. Audit Logging
- **Request logs**: key_hash, endpoint, status, response_time
- **Auth events**: KEY_GENERATED, KEY_REVOKED
- **Output**: Stdout (development), structured JSON (production)

---

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

---

## 📋 Pre-Launch Checklist

- [ ] Change `ADMIN_SECRET` from default in `.env`
- [ ] Test invalid keys return 401
- [ ] Test rate limits return 429 after burst
- [ ] Verify security headers with https://securityheaders.com
- [ ] Set up log monitoring (ELK/Datadog)
- [ ] Document key rotation policy
- [ ] Plan incident response (key compromise)
- [ ] Load test with expected traffic

---

## 🧩 Architecture

### Request Flow

```
1. Client sends request with X-API-Key header
   ↓
2. FastAPI middleware adds security headers
   ↓
3. verify_api_key dependency:
   - Hashes incoming key (SHA-256)
   - Looks up hash in storage
   - Updates last_used + request_count
   - Returns 401 if invalid
   ↓
4. check_rate_limit_dependency:
   - Checks token bucket for key
   - Refills tokens based on elapsed time
   - Returns 429 if depleted
   ↓
5. Input validation (Pydantic):
   - Length, type, injection patterns
   - Returns 422 if invalid
   ↓
6. Endpoint logic executes
   ↓
7. Audit log records request
   ↓
8. Response returned with security headers
```

### Module Structure

```python
# app.py - FastAPI application
from src.m3_3_api_security.auth import verify_api_key, key_manager
from src.m3_3_api_security.limits import check_rate_limit_dependency
from src.m3_3_api_security.security import (
    add_security_headers,
    configure_cors,
    QueryRequest,
    AuditLogger
)

# src/m3_3_api_security/auth.py
class APIKeyManager:
    def generate_key(name, admin_secret) -> str
    def verify_key(raw_key) -> bool
    def revoke_key(raw_key, admin_secret) -> bool
    def list_keys(admin_secret) -> List[Dict]

# src/m3_3_api_security/limits.py
class TokenBucketLimiter:
    def check_rate_limit(key_identifier) -> Tuple[bool, Dict]

# src/m3_3_api_security/security.py
SECURITY_HEADERS: Dict[str, str]
class QueryRequest(BaseModel)
class AuditLogger
```

---

## ⚠️ Common Pitfalls

### 1. Storing Keys in Plaintext
**Mistake**: Database compromise = instant key theft
**Fix**: SHA-256 hashing (like passwords)

### 2. No Rate Limiting
**Mistake**: Single attacker can drain $1000/month budget in 1 hour
**Fix**: Token bucket (60/min, 1000/hour)

### 3. Leaking Errors
**Mistake**: `return {"error": str(e)}` exposes internal paths
**Fix**: Generic error messages in production

### 4. CORS Wildcards
**Mistake**: `allow_origins=["*"]` lets any site call your API
**Fix**: Whitelist exact origins

### 5. Ignoring Prompt Injection
**Mistake**: Trusting user input to LLM directly
**Fix**: Input validation + LLM moderation (recommended)

---

## 💡 Cost-Benefit Reality

| Security Level | Implementation Time | Cost/Month | Protects Against |
|---------------|---------------------|------------|------------------|
| **None** | 0 hours | $0 | Nothing (will be abused) |
| **This module** | 6-10 hours | $0 | 80% of attacks |
| **+ OAuth2** | +20 hours | $0 | 95% of attacks |
| **+ WAF** | +5 hours | $20-200 | 99% of attacks |
| **Enterprise** | +100 hours | $500+ | 99.9% + compliance |

**Recommendation:** Start with this module. Iterate based on actual threat data.

---

## 🔒 Security Best Practices

### DO:
✅ Hash API keys (SHA-256)
✅ Use HTTPS in production
✅ Rotate keys regularly (90 days)
✅ Monitor audit logs for anomalies
✅ Set CORS origins explicitly
✅ Return generic error messages

### DON'T:
❌ Store keys in plaintext
❌ Use `allow_origins=["*"]`
❌ Log sensitive query contents
❌ Expose internal errors to users
❌ Skip rate limiting "because it's internal"
❌ Commit `.env` to git

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

**Note:** This is an educational module for RAG21D learners. Assumes MIT license unless otherwise specified.

---

## 🤝 Contributing

This is a learning module. Feedback and improvements are welcome:

1. Test thoroughly before proposing changes
2. Update tests for new features
3. Follow existing code style
4. Document security trade-offs

---

## 📞 Support

**Issues:** See notebook Section 7 for troubleshooting
**Security concerns:** Rotate compromised keys immediately
**Questions:** Review source document for implementation rationale

---

## 🎓 Additional Resources

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Rate Limiting Algorithms](https://en.wikipedia.org/wiki/Token_bucket)
- [Prompt Injection Defenses](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)

---

**Final Thought:**
*Perfect is the enemy of good. This implementation won't stop nation-state actors, but it WILL stop script kiddies, budget thieves, and 80% of real-world attacks. Ship it. Iterate based on actual threats.*
