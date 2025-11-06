"""
Configuration module for document processing pipeline.
Reads environment variables and provides client initialization.
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
OPENAI_MODEL = "text-embedding-3-small"
PINECONE_INDEX = "production-rag"
REGION = "us-east-1"
DEFAULT_NAMESPACE = "demo"

# Embedding configuration
EMBEDDING_DIMENSION = 1536  # For text-embedding-3-small
BATCH_SIZE = 100  # Pinecone batch size limit


def get_clients() -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize and return OpenAI and Pinecone clients.

    Returns:
        Tuple of (openai_client, pinecone_client).
        Either or both may be None if API keys are not configured.
    """
    openai_client = None
    pinecone_client = None

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            print(f"⚠️  Failed to initialize OpenAI client: {e}")
    else:
        print("⚠️  OPENAI_API_KEY not found in environment")

    # Initialize Pinecone client
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        try:
            from pinecone import Pinecone
            pinecone_client = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            print(f"⚠️  Failed to initialize Pinecone client: {e}")
    else:
        print("⚠️  PINECONE_API_KEY not found in environment")

    return openai_client, pinecone_client


def get_pinecone_region() -> str:
    """Get Pinecone region from environment or use default."""
    return os.getenv("PINECONE_REGION", REGION)
