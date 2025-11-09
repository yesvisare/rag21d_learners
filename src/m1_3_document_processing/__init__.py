"""
M1.3 — Document Processing Pipeline (Extraction → Cleaning → Chunking → Embedding → Storage)

## Purpose
Transform raw documents (PDF, TXT, Markdown) into searchable vector embeddings stored in Pinecone.
This module teaches the complete document processing pipeline for production RAG systems, handling
extraction, text normalization, intelligent chunking, metadata enrichment, and vector storage.

## Concepts Covered
- **Multi-format extraction**: PyMuPDF for PDFs, plain text readers with encoding fallback
- **Text cleaning**: Unicode normalization (NFKC), artifact removal, line-break fixing
- **Chunking strategies**: Fixed-size, semantic (recursive), and paragraph-aware approaches
- **Metadata enrichment**: IDs, content hashing, counts, semantic flags (code, lists, headings)
- **Batch embedding**: OpenAI text-embedding-3-small with efficient batching
- **Vector storage**: Pinecone upsert with metadata size limits and deduplication
- **Production patterns**: Error handling, logging, graceful degradation without API keys

## After Completing This Module
You will be able to:
- Extract text from PDFs and preserve per-page metadata
- Clean raw text to remove Unicode artifacts and PDF noise
- Choose appropriate chunking strategies for different document types
- Enrich chunks with semantic metadata for precise filtering
- Generate and batch embeddings cost-effectively (~$0.13 per million tokens)
- Store vectors in Pinecone with proper metadata trimming
- Handle common failures (Unicode errors, memory exhaustion, bad chunking, duplicate chunks)
- Make informed decisions between custom pipelines, managed services, and long-context embeddings

## Context in Track
- **M1.1**: Introduced vector databases and Pinecone fundamentals
- **M1.2**: Covered Pinecone data model, indexing, and metadata filtering
- **M1.3** (this module): Document processing pipeline (extraction to storage)
- **M1.4** (next): Query pipeline and response generation (retrieval to answer)

## Trade-offs
- **What you gain**: Semantic search, automation, multi-format support
- **What you lose**: Exact formatting (tables/images), cross-document reasoning, real-time updates
- **When NOT to build**: <10 documents, scanned PDFs without OCR, need for real-time processing

## Failure Modes
1. Unicode errors (smart quotes → � characters)
2. Memory exhaustion (large PDFs loaded entirely)
3. Bad chunking (code split mid-function)
4. Garbled table extraction (tables become text soup)
5. Duplicate chunks (reprocessing without deduplication)
"""

from src.m1_3_document_processing.module import (
    Document,
    Chunk,
    DocumentExtractor,
    TextCleaner,
    FixedSizeChunker,
    SemanticChunker,
    ParagraphChunker,
    MetadataExtractor,
    EmbeddingPipeline,
    process_document,
)

from src.m1_3_document_processing.config import (
    OPENAI_MODEL,
    PINECONE_INDEX,
    REGION,
    DEFAULT_NAMESPACE,
    EMBEDDING_DIMENSION,
    BATCH_SIZE,
    get_clients,
    get_pinecone_region,
)

__all__ = [
    # Data classes
    "Document",
    "Chunk",
    # Processing classes
    "DocumentExtractor",
    "TextCleaner",
    "FixedSizeChunker",
    "SemanticChunker",
    "ParagraphChunker",
    "MetadataExtractor",
    "EmbeddingPipeline",
    # Functions
    "process_document",
    # Config
    "OPENAI_MODEL",
    "PINECONE_INDEX",
    "REGION",
    "DEFAULT_NAMESPACE",
    "EMBEDDING_DIMENSION",
    "BATCH_SIZE",
    "get_clients",
    "get_pinecone_region",
]
