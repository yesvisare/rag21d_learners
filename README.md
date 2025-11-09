# M3.1 - Docker Containerization

Production-ready Docker setup for RAG applications with FastAPI.

---

## 📚 Learning Arc

### Purpose
Learn to containerize RAG applications using Docker with production-ready practices including multi-stage builds, health checks, and security hardening. This module bridges local development and cloud deployment by creating portable, reproducible environments.

### Concepts Covered
- **Docker Fundamentals**: Images, containers, volumes, and networks
- **Multi-stage Builds**: Optimize image size and security
- **Container Orchestration**: docker-compose for multi-service applications
- **Health Checks**: Automatic failure detection and recovery
- **Security Best Practices**: Non-root users, minimal images, secret management
- **Deployment Strategies**: When to use Docker vs VMs vs bare metal vs Kubernetes
- **Troubleshooting**: Common failure modes and debugging techniques

### After Completing This Module
You will be able to:
- ✅ Build production-ready Docker images with security hardening
- ✅ Orchestrate multi-container applications with docker-compose
- ✅ Implement health checks and automatic restart policies
- ✅ Debug common Docker networking and permission issues
- ✅ Make informed decisions about deployment strategies
- ✅ Deploy containerized applications to cloud platforms

### Context in Track
**Module 3: Production Deployment** - Video 3.1 of 4

This module is the foundation for cloud deployment. After containerizing your application here, you'll deploy to Railway/Render (M3.2), implement API security (M3.3), and perform load testing (M3.4).

---

## Why Docker?

Docker solves the "works on my machine" problem by packaging your application with all dependencies into portable containers. This provides:

- **Environment consistency** across dev/staging/production
- **Horizontal scaling** - spin up containers as needed
- **Simplified dependency management** - everything is version-controlled

### Trade-offs

**Benefits:**
- Genuine environment reproducibility (eliminates 60-80% of deployment issues)
- Enables cloud-native scaling
- Infrastructure as code

**Limitations:**
- Adds networking complexity
- 10-20% I/O performance overhead for intensive workloads
- Learning curve: 4-6 hours for competent developers
- Cost: $10-100/month for registries plus infrastructure

## Quick Start

### Prerequisites

- Docker Engine installed ([docs.docker.com](https://docs.docker.com))
- Docker Compose v3.8+
- Python 3.11+

### Build and Run

```bash
# From the docker/ directory
cd docker

# Build and start all services
docker compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker compose logs -f web

# Stop services
docker compose down
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with hot reload (PowerShell)
./scripts/run_local.ps1

# Or with Python directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_smoke.py

# Run with coverage
pytest --cov=app --cov-report=html
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
ENVIRONMENT=development
REDIS_HOST=redis
REDIS_PORT=6379
PORT=8000
```

## Architecture

### Services

1. **web** - FastAPI application (port 8000)
   - Multi-stage build for optimized image size
   - Non-root user (UID 1000)
   - Health checks enabled

2. **redis** - Cache layer (port 6379)
   - Persistent volume for data
   - Health checks with redis-cli

### Volumes

- `redis-data` - Persistent Redis data

### Networks

- `rag-network` - Bridge network for inter-container communication

## Project Structure

```
.
├── app.py                          # FastAPI application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── LICENSE                         # MIT License
├── .gitignore                      # Git exclusions
│
├── src/
│   └── m3_1_docker_containerization/
│       └── __init__.py            # Module initialization
│
├── docker/
│   ├── Dockerfile                 # Multi-stage production image
│   ├── docker-compose.yml         # Service orchestration
│   └── .dockerignore              # Build exclusions
│
├── notebooks/
│   └── M3_1_Containerization_with_Docker.ipynb
│
├── tests/
│   ├── test_smoke.py              # FastAPI TestClient tests
│   └── test_docker_sanity.py      # Docker setup tests
│
└── scripts/
    └── run_local.ps1              # Local development script
```

## Development Workflow

```bash
# Run tests
pytest

# Run sanity tests
python -m pytest tests/test_docker_sanity.py

# CLI usage
python app.py --serve              # Start server
python app.py --health             # Health check

# Docker commands (from docker/ directory)
cd docker
docker compose build --no-cache    # Rebuild from scratch
docker compose ps                  # Check status
docker compose exec web /bin/bash  # Shell into container
docker stats                       # Monitor resources
```

## Alternatives to Docker

### When NOT to use Docker:

1. **Solo developer, single server** → Use virtualenv + systemd
2. **Windows-specific dependencies** → Use VMs or native deployment
3. **Ultra-low latency (<5ms)** → Use bare metal

### Decision Framework:

- **2+ developers, multi-environment** → Docker
- **Legacy systems, OS dependencies** → Virtual Machines
- **Single server, max performance** → Bare metal + virtualenv
- **50+ services, enterprise scale** → Kubernetes

See `notebooks/M3_1_Containerization_with_Docker.ipynb` for detailed decision card.

## Troubleshooting

### Container Networking vs Localhost

**Important**: When services communicate inside Docker, use **service names**, not `localhost`:

```bash
# ✓ Correct (inside container)
REDIS_HOST=redis

# ✗ Wrong (inside container)
REDIS_HOST=localhost

# ✓ Correct (from host machine)
curl http://localhost:8000/health

# Note: "localhost" works from the host because of port mapping
```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8000

# Stop containers properly
docker compose down

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Map to different host port
```

### Volume Permission Errors

```bash
# Fix permissions
chmod 755 ./data
sudo chown -R 1000:1000 ./data

# Or use user matching
echo "USER_ID=$(id -u)" >> .env
echo "GROUP_ID=$(id -g)" >> .env
```

### Container Networking Issues

```bash
# Inspect network (from docker/ directory)
docker network inspect docker_rag-network

# Test connectivity between containers
docker compose exec web ping redis
docker compose exec web curl http://redis:6379

# View container logs
docker compose logs -f web
docker compose logs -f redis
```

### Stale Cache

```bash
# Force rebuild (from docker/ directory)
docker compose build --no-cache

# Clean everything
docker compose down -v
docker system prune -a
```

### Out of Memory (Exit 137)

```bash
# Monitor resources
docker stats

# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

## Docker Compose Commands Reference

```bash
# All commands should be run from the docker/ directory
cd docker

# Build and start
docker compose up -d

# View logs (follow mode)
docker compose logs -f

# View logs for specific service
docker compose logs -f web
docker compose logs -f redis

# Check service status
docker compose ps

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild and restart
docker compose up -d --build

# Execute command in running container
docker compose exec web /bin/bash
docker compose exec web python app.py --health
```

## Production Considerations

### Security

- ✅ Multi-stage build to minimize attack surface
- ✅ Non-root user (UID 1000)
- ✅ No secrets in image (use env vars)
- ✅ Minimal base image (python:3.11-slim)
- ✅ Health checks for automatic recovery

### Image Size

- Base: ~180MB
- With dependencies: ~200-250MB (multi-stage optimization)
- Previous single-stage: ~300MB

### Monitoring

Required metrics:
- Container restarts
- Memory/CPU usage
- Request latency
- Health check status

See `notebooks/M3_1_Containerization_with_Docker.ipynb` for scaling guide.

## Files

### Application
- `app.py` - FastAPI application with health endpoints
- `requirements.txt` - Python dependencies (FastAPI, Uvicorn, pytest, httpx)

### Docker Configuration
- `docker/Dockerfile` - Multi-stage production image
- `docker/docker-compose.yml` - Service orchestration with health checks
- `docker/.dockerignore` - Build exclusions

### Testing
- `tests/test_smoke.py` - FastAPI TestClient tests
- `tests/test_docker_sanity.py` - Docker setup verification

### Documentation
- `notebooks/M3_1_Containerization_with_Docker.ipynb` - Interactive learning notebook

### Scripts
- `scripts/run_local.ps1` - Local development without Docker

## Next Steps

1. Complete M3.1 notebook challenges in `notebooks/`
2. Deploy to Railway/Render (M3.2)
3. Implement API security (M3.3)
4. Load testing and scaling (M3.4)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Resources

- [Docker Documentation](https://docs.docker.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Docker Compose Reference](https://docs.docker.com/compose/)
