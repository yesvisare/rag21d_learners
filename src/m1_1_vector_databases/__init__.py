"""
M1.1 Vector Databases - Semantic Search Foundations for RAG

Purpose
-------
This module teaches the foundation of Retrieval-Augmented Generation (RAG) systems:
vector databases for semantic search. You'll learn how to convert text into numerical
embeddings, store them efficiently, and retrieve relevant information based on meaning
rather than keyword matching.

Concepts Covered
----------------
1. **Vector Embeddings**: Converting text to 1536-dimensional numerical representations
   using OpenAI's text-embedding-3-small model

2. **Semantic Similarity**: Measuring how similar two pieces of text are using cosine
   similarity (values from -1 to 1)

3. **Approximate Nearest Neighbor (ANN)**: Understanding why vector databases are
   10,000x faster than brute-force similarity search

4. **Pinecone Operations**:
   - Index creation with readiness polling
   - Batch upsertion with rich metadata
   - Semantic search with score thresholding
   - Metadata filtering for multi-tenancy

5. **Production Patterns**:
   - Exponential backoff for rate limiting
   - Namespace isolation for data security
   - Error handling and debugging
   - Cost and latency optimization

6. **Decision Frameworks**: When to use vector databases vs alternatives
   (Elasticsearch, ChromaDB, pgvector, in-memory search)

After Completing
----------------
You will be able to:
- ✅ Build production-ready semantic search systems
- ✅ Understand the trade-offs between vector databases and traditional search
- ✅ Debug the 5 most common vector database failures
- ✅ Implement multi-tenant isolation using namespaces
- ✅ Optimize costs and latency for production workloads
- ✅ Make informed architectural decisions about search systems

Context in Track
----------------
This is Module 1.1 in the RAG21D learning track. It provides the foundational
knowledge for all subsequent modules:

- **M1.2**: Chunking Strategies - How to split documents for optimal retrieval
- **M1.3**: Embedding Models - Choosing and fine-tuning embedding models
- **M1.4**: Retrieval Strategies - Advanced querying techniques (hybrid search, reranking)
- **M2.x**: LLM Integration - Connecting retrieval to language models
- **M3.x**: Production RAG - Scaling, monitoring, and evaluation

Without understanding vector databases, you cannot build effective RAG systems.
This module is the critical first step.

Usage
-----
### As a Library:
```python
from src.m1_1_vector_databases import config, module

# Load data and generate embeddings
texts = module.load_example_texts()
embeddings = module.embed_texts_openai(texts)

# Create index and upsert
openai_client, pinecone_client = config.get_clients()
index = module.create_index_and_wait_pinecone(pinecone_client, config.INDEX_NAME)
# ... upsert and query
```

### As a CLI:
```bash
python -m src.m1_1_vector_databases.module --init
python -m src.m1_1_vector_databases.module --query "vector search"
```

### As an API:
```bash
uvicorn app:app --reload
curl -X POST http://localhost:8000/m1_1/query -d '{"query": "vector search"}'
```

See README.md for complete documentation and examples.
"""

from src.m1_1_vector_databases import config
from src.m1_1_vector_databases.module import (
    load_example_texts,
    embed_texts_openai,
    cosine_similarity,
    create_index_and_wait_pinecone,
    upsert_vectors,
    query_pinecone,
)

__version__ = "1.0.0"

__all__ = [
    "config",
    "load_example_texts",
    "embed_texts_openai",
    "cosine_similarity",
    "create_index_and_wait_pinecone",
    "upsert_vectors",
    "query_pinecone",
]
