# M3.2 — Cloud Deployment Guide (Railway/Render)

**One-page deployment guide for getting FastAPI apps live on Railway or Render.**

---

## Quick Start

### Prerequisites
- GitHub account
- Railway or Render account (free tier)
- Python 3.11+
- Git repo with your code

### Files in This Module
```
m3_2_deploy.py              # FastAPI app with health endpoints
requirements.txt            # Dependencies
Procfile                    # Render start command
service.json                # Railway service config (optional)
render.yaml                 # Render infrastructure-as-code
railway.json                # Railway template (optional)
.env.example                # Environment variables template
tests_deploy_sanity.py      # Smoke tests
M3_2_Cloud_Deployment.ipynb # Full deployment walkthrough
```

---

## Option A: Deploy to Render (750 Free Hours/Month)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add M3.2 deployment files"
git push origin main
```

### Step 2: Create Web Service
1. Go to https://dashboard.render.com/
2. Click **New +** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name:** `m3-2-cloud-deploy`
   - **Region:** Oregon (or closest)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn m3_2_deploy:app --host=0.0.0.0 --port=$PORT`

### Step 3: Set Environment Variables
In Render dashboard → Environment:
- `PLATFORM=render`
- `ADMIN_SECRET=<generate-random-string>`

### Step 4: Deploy & Test
Click **Create Web Service**. Wait 2-3 minutes for build.

Test your deployment:
```bash
curl https://your-app.onrender.com/health
# Expected: {"status": "healthy", "service": "m3.2-deploy"}
```

---

## Option B: Deploy to Railway ($5 Monthly Credit)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add M3.2 deployment files"
git push origin main
```

### Step 2: Create Project
1. Go to https://railway.app/new
2. Click **Deploy from GitHub repo**
3. Select your repository
4. Railway auto-detects Python → Click **Deploy Now**

### Step 3: Configure Environment
1. Click your service → **Variables**
2. Add:
   - `PLATFORM=railway`
   - `ADMIN_SECRET=<generate-random-string>`
3. Click **Deploy** to restart

### Step 4: Generate Public Domain
1. Settings → Networking → **Generate Domain**
2. Test your deployment:
```bash
curl https://your-app.up.railway.app/health
# Expected: {"status": "healthy", "service": "m3.2-deploy"}
```

---

## Platform Comparison

| Feature | Railway | Render |
|---------|---------|--------|
| **Free Tier** | $5 credit (~140 hrs) | 750 hours/month |
| **Cold Start** | 15 min idle → 30-60s wake | 15 min idle → 30-60s wake |
| **Setup Speed** | ⚡ Fastest (auto-detect) | Fast (manual config) |
| **PostgreSQL** | ✅ Built-in | ✅ Free (90-day limit) |
| **Redis** | ✅ Built-in | ❌ Use Upstash |
| **Always-On Cost** | $5/month | $7/month |

---

## Cost Breakdown

### Free Tier Reality
- **Railway:** $5 monthly credit = ~140 hours runtime
- **Render:** 750 hours = one always-on app OR multiple apps with usage limits
- **Cold Starts:** Both platforms sleep after 15 minutes of inactivity

### Paid Tier Costs
| Configuration | Railway | Render |
|---------------|---------|--------|
| Basic web app | $5/mo | $7/mo |
| + PostgreSQL | +$5/mo | +$0 (included) |
| + Redis | +$5/mo | N/A (use Upstash) |
| **Total (typical)** | **$15/mo** | **$7/mo** |

### When to Upgrade
- **<5K requests/day:** Free tier acceptable
- **5-50K requests/day:** Upgrade to Starter ($7-15/mo)
- **50-500K requests/day:** Pro tier ($25-50/mo)
- **>500K requests/day:** Migrate to AWS/GCP

---

## Trade-offs (Honest Assessment)

### Railway Pros ✅
- Zero-config deployment (detects everything)
- Fast: GitHub push → live in <10 minutes
- Built-in databases (PostgreSQL, Redis, MySQL)
- Great for rapid prototyping

### Railway Cons ❌
- $5 credit runs out fast (~4-5 days always-on)
- No true free tier (just credit)
- Vendor lock-in (specific env var format)

### Render Pros ✅
- True free tier (750 hours/month)
- One always-on app possible on free tier
- Simple pricing (no surprise charges)
- Free PostgreSQL (with 90-day limit)

### Render Cons ❌
- Manual environment variable setup
- No free Redis (use Upstash instead)
- Slower cold starts than Railway
- Persistent storage limitations

---

## Alternatives Decision Flow

```
Do you need HIPAA/SOC2 compliance?
├─ YES → AWS/GCP (certified platforms)
└─ NO → Continue

Is your monthly budget under $10?
├─ YES → Railway/Render free tier (cold starts acceptable)
└─ NO → Continue

Is traffic < 50K requests/day?
├─ YES → Railway/Render Starter ($7-15/mo)
└─ NO → Continue

Do you have DevOps skills (3+/5)?
├─ YES → VPS (DigitalOcean, $5-20/mo, full control)
└─ NO → Railway/Render Pro ($25-50/mo)

Is traffic > 500K requests/day?
└─ YES → AWS/GCP (enterprise scale)
```

### Other Platform Options

| Platform | Best For | Starting Cost |
|----------|----------|---------------|
| **Fly.io** | Global edge deployment | $1.94/month |
| **DigitalOcean** | Full VPS control | $5/month |
| **Linode** | Predictable VPS pricing | $5/month |
| **Heroku** | Legacy apps (not recommended for new projects) | $7/month |
| **AWS Lightsail** | AWS ecosystem entry | $5/month |

---

## Testing Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
cp .env.example .env
# Edit .env and set ADMIN_SECRET
```

### 3. Run App
```bash
python m3_2_deploy.py
```

### 4. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Environment info
curl http://localhost:8000/env-check
```

### 5. Run Tests
```bash
python tests_deploy_sanity.py
```

**Expected output:**
```
🧪 Running M3.2 Deployment Sanity Tests

✅ Health check passed
✅ Root endpoint passed
✅ Readiness check passed
✅ Environment check passed
✅ ADMIN_SECRET is set
✅ PORT is valid: 8000

✅ All sanity tests passed!
```

---

## Troubleshooting

### Issue: App crashes on startup
**Check:**
1. Verify `requirements.txt` includes all imports
2. Ensure start command uses `$PORT` variable
3. App must listen on `0.0.0.0`, not `localhost`

**Fix:**
```python
# m3_2_deploy.py
port = int(os.getenv("PORT", "8000"))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### Issue: Health check fails
**Check platform logs:**
- Railway: Deployments → View Logs
- Render: Logs tab

**Common causes:**
- Wrong health check path (must be `/health`)
- Timeout too short (increase to 60s)
- App not returning 200 status

### Issue: 502 Bad Gateway
**Causes:**
1. Cold start (wait 30-60 seconds)
2. App crashed (check logs)
3. Wrong port binding

### Issue: Environment variables not loading
**Test endpoint:**
```bash
curl https://your-app.url/env-check
```

**If variables missing:**
- Railway: Add in Variables tab
- Render: Add in Environment tab
- Must restart/redeploy after adding

### Issue: Out of memory errors
**Free tier memory limits:**
- Railway: 512MB
- Render: 512MB-1GB

**Solutions:**
1. Upgrade to paid tier (more memory)
2. Optimize app (reduce dependencies)
3. Use external services (databases, Redis)

---

## Advanced: Custom Domains

### Railway
1. Settings → Networking → **Custom Domain**
2. Add CNAME record: `your-domain.com` → `your-app.up.railway.app`
3. Wait for DNS propagation (~5-60 minutes)

### Render
1. Settings → Custom Domain → **Add Custom Domain**
2. Add CNAME record: `your-domain.com` → `your-app.onrender.com`
3. SSL auto-configured (Let's Encrypt)

---

## Next Steps

1. **Add Database:** Follow platform docs for PostgreSQL
2. **Set Up CI/CD:** Both platforms auto-deploy from GitHub
3. **Monitor Logs:** Use platform dashboards or integrate Sentry
4. **Scale Up:** Upgrade to paid tier when traffic increases
5. **Migrate:** Plan migration path if outgrowing platform (see Alternatives)

---

## Resources

### Railway
- Docs: https://docs.railway.app
- Discord: https://discord.gg/railway
- Status: https://status.railway.app

### Render
- Docs: https://render.com/docs
- Community: https://community.render.com
- Status: https://status.render.com

### This Module
- Full walkthrough: `M3_2_Cloud_Deployment.ipynb`
- Tests: `tests_deploy_sanity.py`
- App code: `m3_2_deploy.py`

---

## License

MIT License - modify and use freely for your projects.

---

**Questions?** Open an issue or check the notebook for detailed troubleshooting steps.
