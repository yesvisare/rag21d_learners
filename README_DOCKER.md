# M3.1 - Docker Containerization

Production-ready Docker setup for RAG applications with FastAPI.

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

- Docker Engine installed (docs.docker.com)
- Docker Compose v3.8+

### Build and Run

```bash
# Build the image
docker build -t rag-api:latest .

# Run with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

### Environment Variables

Create a `.env` file:

```env
ENVIRONMENT=development
REDIS_HOST=redis
REDIS_PORT=6379
```

## Architecture

### Services

1. **web** - FastAPI application (port 8000)
2. **redis** - Cache layer (port 6379)

### Volumes

- `redis-data` - Persistent Redis data

### Networks

- `rag-network` - Bridge network for inter-container communication

## Development Workflow

```bash
# Run sanity tests
python tests_docker_sanity.py

# CLI usage
python m3_1_dockerize.py --serve        # Start server
python m3_1_dockerize.py --health       # Health check

# Docker commands
docker-compose build --no-cache         # Rebuild from scratch
docker-compose ps                       # Check status
docker stats                            # Monitor resources
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

See `M3_1_Containerization_with_Docker.ipynb` for detailed decision card.

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8000

# Stop containers properly
docker-compose down

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
# Inspect network
docker network inspect rag-network

# Test connectivity
docker-compose exec web ping redis
docker-compose exec web curl http://redis:6379

# Use service names, not localhost
REDIS_HOST=redis  # Correct
REDIS_HOST=localhost  # Wrong (inside container)
```

### Stale Cache

```bash
# Force rebuild
docker-compose build --no-cache

# Clean everything
docker-compose down -v
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

## Production Considerations

### Security

- Non-root user (UID 1000)
- No secrets in image (use env vars)
- Minimal base image (python:3.11-slim)

### Image Size

- Base: ~180MB
- With dependencies: ~250-300MB
- Use multi-stage builds for further reduction

### Monitoring

Required metrics:
- Container restarts
- Memory/CPU usage
- Request latency
- Health check status

See `M3_1_Containerization_with_Docker.ipynb` for scaling guide.

## Files

- `m3_1_dockerize.py` - FastAPI application
- `Dockerfile` - Image definition
- `docker-compose.yml` - Multi-container orchestration
- `.dockerignore` - Build exclusions
- `requirements.txt` - Python dependencies
- `tests_docker_sanity.py` - Smoke tests

## Next Steps

1. Complete M3_1 notebook challenges
2. Deploy to Railway/Render (M3.2)
3. Implement API security (M3.3)
4. Load testing and scaling (M3.4)
