"""
Configuration for M2.2 Prompt Optimization & Model Selection
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_DEFAULT = "gpt-3.5-turbo"

# Model Pricing (per 1M tokens)
MODEL_PRICE_TABLE = {
    "gpt-3.5-turbo": {
        "input": 0.50,   # $0.50 per 1M input tokens
        "output": 1.50,  # $1.50 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00,
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.00,  # No output tokens for embeddings
    },
}

# Token Limits
MAX_CONTEXT_TOKENS_DEFAULT = 3000
MAX_OUTPUT_TOKENS_DEFAULT = 400
PROMPT_OVERHEAD_TOKENS = 350  # Typical system + user prompt overhead
SAFETY_MARGIN_TOKENS = 200    # Buffer for edge cases

# Model Context Windows
MODEL_CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
}

# Optimization Levels
OPTIMIZATION_LEVELS = {
    "conservative": {
        "max_context_tokens": 3000,
        "max_output_tokens": 400,
        "template_type": "STRUCTURED_RAG",
    },
    "balanced": {
        "max_context_tokens": 2500,
        "max_output_tokens": 300,
        "template_type": "CONCISE_RAG",
    },
    "aggressive": {
        "max_context_tokens": 2000,
        "max_output_tokens": 250,
        "template_type": "JSON_RAG",
    },
}
