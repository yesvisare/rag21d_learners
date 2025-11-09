"""
FastAPI router for M2.3 Production Monitoring Dashboard

Provides HTTP endpoints for:
- Health checks
- Demo metric simulation
- Cost estimation
"""
import random
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .module import metrics, track_cache_operation, monitored_query


router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)


# === Request/Response Models ===

class HealthResponse(BaseModel):
    status: str
    module: str
    version: str = "1.0.0"


class CostEstimateRequest(BaseModel):
    input_tokens: int = Field(..., ge=0, description="Number of input tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens")
    model: Optional[str] = Field("default", description="Model name for cost calculation")


class CostEstimateResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    breakdown: dict


class SimulateRequest(BaseModel):
    count: int = Field(5, ge=1, le=100, description="Number of demo queries to simulate")
    operation: str = Field("demo_api", description="Operation name for metrics")
    model: str = Field("gpt-4", description="Model name")


class SimulateResponse(BaseModel):
    message: str
    queries_simulated: int
    metrics_recorded: list


# === Endpoints ===

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring service availability.
    Returns basic service information.
    """
    return {
        "status": "ok",
        "module": "m2_3_monitoring",
        "version": "1.0.0"
    }


@router.post("/cost/estimate", response_model=CostEstimateResponse)
async def estimate_cost(request: CostEstimateRequest):
    """
    Estimate cost for a given token usage.

    This endpoint calculates the estimated cost based on input/output tokens
    and the specified model. Useful for budget planning and cost optimization.
    """
    try:
        cost = metrics.calculate_cost(
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            model=request.model
        )

        from .config import COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS

        return {
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "model": request.model,
            "cost_usd": round(cost, 6),
            "breakdown": {
                "input_cost_usd": round((request.input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS, 6),
                "output_cost_usd": round((request.output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS, 6),
                "rate_per_1k_input": COST_PER_1K_INPUT_TOKENS,
                "rate_per_1k_output": COST_PER_1K_OUTPUT_TOKENS,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost calculation failed: {str(e)}")


@router.post("/simulate", response_model=SimulateResponse)
async def simulate_queries(request: SimulateRequest):
    """
    Simulate RAG queries to generate demo metrics.

    This endpoint creates sample metrics data for testing dashboards and alerts.
    It simulates realistic query patterns including:
    - Variable latencies (retrieval + LLM generation)
    - Token usage variations
    - Cache hits/misses
    - Relevance scores
    """

    @monitored_query(operation=request.operation, model=request.model)
    def _simulate_single_query(query_id: int):
        """Simulate a single RAG query with realistic timing"""

        # Simulate cache check
        cache_hit = random.random() > 0.7  # 30% cache hit rate
        track_cache_operation(hit=cache_hit, cache_type="semantic")

        if cache_hit:
            # Fast path - cached result
            time.sleep(random.uniform(0.01, 0.05))
            return {
                'input_tokens': random.randint(400, 600),
                'output_tokens': random.randint(100, 200),
                'relevance_score': random.uniform(0.8, 0.95),
                'cached': True
            }

        # Simulate retrieval
        retrieval_time = random.uniform(0.05, 0.3)
        metrics.retrieval_latency.observe(retrieval_time)
        time.sleep(retrieval_time)

        # Simulate LLM generation
        llm_time = random.uniform(0.5, 2.0)
        metrics.llm_latency.observe(llm_time)
        time.sleep(llm_time)

        return {
            'input_tokens': random.randint(500, 1500),
            'output_tokens': random.randint(150, 500),
            'relevance_score': random.uniform(0.65, 0.92),
            'cached': False
        }

    # Run simulations
    results = []
    for i in range(request.count):
        try:
            result = _simulate_single_query(i)
            results.append({
                'query_id': i,
                'tokens': f"{result['input_tokens']}+{result['output_tokens']}",
                'relevance': round(result['relevance_score'], 2),
                'cached': result['cached']
            })
        except Exception as e:
            results.append({
                'query_id': i,
                'error': str(e)
            })

    return {
        "message": f"Simulated {request.count} queries successfully",
        "queries_simulated": request.count,
        "metrics_recorded": results
    }


# Additional utility endpoint for debugging
@router.get("/metrics-info")
async def metrics_info():
    """
    Get information about available metrics.
    Returns a list of all registered Prometheus metrics.
    """
    from prometheus_client import REGISTRY

    metric_families = []
    for metric_family in REGISTRY.collect():
        metric_families.append({
            'name': metric_family.name,
            'type': metric_family.type,
            'documentation': metric_family.documentation,
        })

    return {
        "total_metrics": len(metric_families),
        "metrics": metric_families
    }
