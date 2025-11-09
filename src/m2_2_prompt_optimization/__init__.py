"""
M2.2 — Prompt Optimization & Model Selection

**Purpose:**
Learn to reduce RAG LLM costs by 30-50% through intelligent prompt engineering,
token optimization, and model routing without sacrificing quality. This module
teaches you when and how to optimize prompts, and critically, when NOT to optimize.

**Concepts Covered:**
- RAG-specific prompt templates (5 production-tested variants)
- Token estimation and cost projection across models
- Intelligent model routing based on query complexity
- Context formatting and smart document truncation
- A/B testing framework for prompt comparison
- Cost/quality trade-offs and decision frameworks
- Common failure modes and debugging strategies
- ROI analysis and break-even calculations

**After Completing:**
You will be able to:
- Design and test prompt variants that reduce token usage by 30-50%
- Route queries to appropriate models based on complexity and cost constraints
- Measure and project costs at different scales (100 to 100K queries/day)
- Identify when prompt optimization is counterproductive
- Debug the 5 most common prompt optimization failures
- Make data-driven decisions using ROI and decision frameworks

**Context in Track:**
This is Module 2.2 in the RAG Production Engineering track:
- M1.x: Built foundational RAG system with vector search and generation
- M2.1: Implemented caching strategies for cost reduction
- **M2.2: Optimize prompts and route models intelligently** ← YOU ARE HERE
- M2.3: Build production monitoring dashboards
- M2.4: Implement error handling and reliability patterns

Prerequisites: M2.1 (Caching), working RAG system, OpenAI API access (optional for testing)
Estimated time: 60-90 minutes for implementation + practice
"""

from .module import (
    # Prompt templates
    RAGPromptLibrary,
    PromptTemplate,

    # Core utilities
    TokenEstimator,
    ModelRouter,
    ModelTier,
    format_context_optimally,

    # Testing framework
    PromptTester,
    PromptTestResult,
)

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_DEFAULT,
    MODEL_PRICE_TABLE,
    MODEL_CONTEXT_WINDOWS,
    OPTIMIZATION_LEVELS,
)

__version__ = "2.2.0"

__all__ = [
    # Templates
    "RAGPromptLibrary",
    "PromptTemplate",

    # Utilities
    "TokenEstimator",
    "ModelRouter",
    "ModelTier",
    "format_context_optimally",

    # Testing
    "PromptTester",
    "PromptTestResult",

    # Config
    "OPENAI_API_KEY",
    "OPENAI_MODEL_DEFAULT",
    "MODEL_PRICE_TABLE",
    "MODEL_CONTEXT_WINDOWS",
    "OPTIMIZATION_LEVELS",
]
