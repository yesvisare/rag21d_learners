"""
Configuration for M1.4 Query Pipeline & Response Generation.
Reads from .env and provides constants for the system.
"""
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Pinecone configuration
PINECONE_INDEX = "production-rag"
REGION = "us-east-1"
DEFAULT_NAMESPACE = "demo"

# Retrieval configuration
SCORE_THRESHOLD = 0.7
DEFAULT_TOP_K = 5
MAX_CONTEXT_LENGTH = 4000

# Reranking configuration
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 3


def get_clients() -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize and return OpenAI and Pinecone clients.
    Returns (None, None) if API keys are not found.

    Returns:
        Tuple of (openai_client, pinecone_client) or (None, None)
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key or not pinecone_key:
        return None, None

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        openai_client = OpenAI(api_key=openai_key)
        pinecone_client = Pinecone(api_key=pinecone_key)

        return openai_client, pinecone_client
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return None, None


def has_api_keys() -> bool:
    """Check if required API keys are present."""
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY"))
