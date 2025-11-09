"""
FastAPI router for M2.2 Prompt Optimization endpoints.

Provides REST API access to prompt optimization, model routing, and testing functionality.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .module import (
    ModelRouter,
    ModelTier,
    PromptTester,
    RAGPromptLibrary,
    PromptTemplate,
)


# Request/Response models
class RouteRequest(BaseModel):
    """Request model for intelligent model routing."""
    query: str = Field(..., description="User query to analyze")
    context: Optional[str] = Field(None, description="Context that will be sent with query")
    force_tier: Optional[str] = Field(None, description="Force specific tier: FAST, BALANCED, PREMIUM")
    cost_budget: Optional[float] = Field(None, description="Maximum cost per query in dollars")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is your return policy?",
                "context": "Our policy allows returns within 30 days...",
                "cost_budget": 0.001
            }
        }


class RouteResponse(BaseModel):
    """Response model for routing decision."""
    model: str
    tier: str
    complexity_score: int
    complexity_factors: Dict[str, bool]
    reason: str
    estimated_cost: Optional[float] = None


class CompareRequest(BaseModel):
    """Request model for prompt template comparison."""
    templates: List[str] = Field(..., description="Template names to compare (e.g., ['baseline_comparison', 'cost_optimization'])")
    test_cases: List[Dict[str, str]] = Field(..., description="Test cases with 'question' and optional 'expected_answer'")
    documents: List[Dict[str, Any]] = Field(..., description="Documents with 'content' and optional 'score'")

    class Config:
        schema_extra = {
            "example": {
                "templates": ["baseline_comparison", "cost_optimization"],
                "test_cases": [
                    {"question": "What is your return policy?", "expected_answer": "30 days"}
                ],
                "documents": [
                    {"content": "Our return policy allows returns within 30 days.", "score": 0.95}
                ]
            }
        }


class CompareResponse(BaseModel):
    """Response model for template comparison."""
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


# Initialize router
router = APIRouter(
    prefix="/m2_2_prompt_optimization",
    tags=["M2.2 Prompt Optimization"],
)

# Initialize shared instances
_model_router = ModelRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Status information for the module.
    """
    return {
        "status": "ok",
        "module": "m2_2_prompt_optimization",
        "version": "2.2.0"
    }


@router.post("/route", response_model=RouteResponse)
async def route_query(request: RouteRequest) -> RouteResponse:
    """
    Route a query to the appropriate model based on complexity.

    Analyzes query complexity using heuristics (length, reasoning keywords,
    technical content) and selects the optimal model tier. Supports cost
    budget constraints and forced tier selection.

    Args:
        request: RouteRequest with query, optional context, tier, and cost budget

    Returns:
        RouteResponse with selected model, tier, complexity analysis, and reasoning

    Raises:
        HTTPException: If tier is invalid or routing fails
    """
    try:
        # Parse force_tier if provided
        force_tier = None
        if request.force_tier:
            try:
                force_tier = ModelTier[request.force_tier.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tier: {request.force_tier}. Must be FAST, BALANCED, or PREMIUM"
                )

        # Route the query
        decision = _model_router.select_model(
            query=request.query,
            context=request.context or "",
            force_tier=force_tier,
            cost_budget=request.cost_budget
        )

        return RouteResponse(**decision)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@router.post("/compare", response_model=CompareResponse)
async def compare_templates(request: CompareRequest) -> CompareResponse:
    """
    Compare multiple prompt templates using A/B testing.

    Runs test cases through different prompt templates and compares token usage,
    cost, and latency. Always runs in dry-run mode (no actual API calls) to avoid
    unexpected costs.

    Args:
        request: CompareRequest with template names, test cases, and documents

    Returns:
        CompareResponse with comparison results and summary

    Raises:
        HTTPException: If templates are invalid or comparison fails
    """
    try:
        # Resolve template names to PromptTemplate objects
        templates = []
        for name in request.templates:
            template = RAGPromptLibrary.get_template_by_name(name)
            if template is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown template: {name}"
                )
            templates.append(template)

        # Initialize tester in dry-run mode (no API calls)
        tester = PromptTester(
            openai_client=None,
            model="gpt-3.5-turbo",
            dry_run=True
        )

        # Run comparison
        results = tester.compare_templates(
            templates=templates,
            test_cases=request.test_cases,
            context_docs=request.documents,
            output_format="json"  # Suppress table output
        )

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "template_name": result.template_name,
                "avg_input_tokens": result.avg_input_tokens,
                "avg_output_tokens": result.avg_output_tokens,
                "avg_total_tokens": result.avg_total_tokens,
                "avg_latency_ms": result.avg_latency_ms,
                "avg_cost_per_query": result.avg_cost_per_query,
                "queries_tested": result.queries_tested,
            })

        # Generate summary
        if results:
            baseline = results_dict[-1]  # Most expensive
            best = results_dict[0]  # Cheapest
            savings_pct = ((baseline["avg_cost_per_query"] - best["avg_cost_per_query"]) /
                          baseline["avg_cost_per_query"] * 100) if baseline["avg_cost_per_query"] > 0 else 0

            summary = {
                "baseline_template": baseline["template_name"],
                "best_template": best["template_name"],
                "max_savings_pct": round(savings_pct, 1),
                "baseline_cost_per_query": baseline["avg_cost_per_query"],
                "best_cost_per_query": best["avg_cost_per_query"],
            }
        else:
            summary = {}

        return CompareResponse(results=results_dict, summary=summary)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get basic usage metrics.

    Returns:
        Dictionary with metric counters (placeholder for now)
    """
    return {
        "routes_processed": 0,
        "comparisons_run": 0,
        "note": "Metrics tracking not yet implemented - placeholder endpoint"
    }
