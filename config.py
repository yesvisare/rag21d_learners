"""
Configuration module for M1.1 Vector Databases Learning Workspace

Reads environment variables and provides constants for:
- OpenAI API credentials and model configuration
- Pinecone API credentials and settings
- Vector database parameters (dimension, index name, etc.)
- Default query and filtering parameters
"""

import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API CREDENTIALS
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

# OpenAI embedding model
# Options: "text-embedding-3-small" (1536-D), "text-embedding-3-large" (3072-D)
EMBEDDING_MODEL = "text-embedding-3-small"

# Embedding dimensions - MUST match the model
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536
}

EMBEDDING_DIM = MODEL_DIMENSIONS[EMBEDDING_MODEL]

# ============================================================================
# PINECONE INDEX CONFIGURATION
# ============================================================================

INDEX_NAME = "tvh-m1-vectors"
DEFAULT_NAMESPACE = "demo"
METRIC = "cosine"  # Options: "cosine", "euclidean", "dotproduct"

# ============================================================================
# QUERY PARAMETERS
# ============================================================================

# Similarity score threshold for filtering results
# Range: -1 to 1 (cosine similarity)
# - 0.7+ = High similarity (recommended for most use cases)
# - 0.5-0.7 = Medium similarity
# - <0.5 = Low similarity (often not relevant)
SCORE_THRESHOLD = 0.7

# Default number of results to return
DEFAULT_TOP_K = 5

# Batch size for upserting vectors (Pinecone recommends 100-200)
BATCH_SIZE = 100

# ============================================================================
# RETRY CONFIGURATION FOR RATE LIMITING
# ============================================================================

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds
RETRY_BACKOFF_MULTIPLIER = 2  # exponential backoff

# ============================================================================
# INDEX INITIALIZATION
# ============================================================================

INDEX_READY_TIMEOUT = 120  # seconds to wait for index initialization
INDEX_READY_CHECK_INTERVAL = 5  # seconds between readiness checks

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def get_clients() -> Tuple:
    """
    Initialize and return OpenAI and Pinecone clients.

    Returns:
        Tuple[OpenAI, Pinecone]: Initialized OpenAI and Pinecone clients

    Raises:
        ValueError: If required API keys are not set
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file.\n"
            "Get your key from: https://platform.openai.com/api-keys"
        )

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found. Please set it in your .env file.\n"
            "Get your key from: https://app.pinecone.io/"
        )

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

        return openai_client, pinecone_client

    except ImportError as e:
        raise ImportError(
            f"Failed to import required libraries: {e}\n"
            "Please install dependencies: pip install -r requirements.txt"
        )


def validate_config() -> bool:
    """
    Validate that all required configuration is present.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    issues = []

    if not OPENAI_API_KEY:
        issues.append("❌ OPENAI_API_KEY is not set")
    else:
        issues.append("✓ OPENAI_API_KEY is set")

    if not PINECONE_API_KEY:
        issues.append("❌ PINECONE_API_KEY is not set")
    else:
        issues.append("✓ PINECONE_API_KEY is set")

    if PINECONE_REGION:
        issues.append(f"✓ PINECONE_REGION: {PINECONE_REGION}")

    issues.append(f"✓ EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    issues.append(f"✓ EMBEDDING_DIM: {EMBEDDING_DIM}")
    issues.append(f"✓ INDEX_NAME: {INDEX_NAME}")

    for issue in issues:
        print(issue)

    # Return True only if no errors
    return not any(issue.startswith("❌") for issue in issues)


if __name__ == "__main__":
    print("Configuration Validation")
    print("=" * 50)
    is_valid = validate_config()
    print("=" * 50)
    if is_valid:
        print("✓ Configuration is valid!")
    else:
        print("\n❌ Configuration has issues. Please check your .env file.")
        print("Copy .env.example to .env and add your API keys.")
