"""
Configuration module for M1.2 Pinecone Advanced Indexing.
Reads API keys from .env file and provides constants for the application.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants
OPENAI_MODEL = "text-embedding-3-small"
PINECONE_INDEX = "hybrid-rag"
REGION = "us-east-1"
DEFAULT_NAMESPACE = "demo"
SCORE_THRESHOLD = 0.7


def get_clients():
    """
    Initialize and return OpenAI and Pinecone clients.

    Returns:
        tuple: (openai_client, pinecone_client) or (None, None) if keys are missing
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key or not pinecone_key:
        print("⚠️ API keys not found in .env file")
        return None, None

    try:
        openai_client = OpenAI(api_key=openai_key)
        pinecone_client = Pinecone(api_key=pinecone_key)
        return openai_client, pinecone_client
    except Exception as e:
        print(f"⚠️ Error initializing clients: {e}")
        return None, None
