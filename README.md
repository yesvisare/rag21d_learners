# M1.3 — Document Processing Pipeline

**Extraction → Cleaning → Chunking → Embedding → Storage**

A production-ready document processing pipeline for RAG systems. Transforms raw documents (PDF, TXT, Markdown) into searchable vector embeddings stored in Pinecone.

---

## Purpose

Transform raw documents into searchable vector embeddings for production RAG systems. This module teaches the complete pipeline from document extraction to Pinecone storage, handling multi-format extraction, text normalization, intelligent chunking, metadata enrichment, and vector storage.

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

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
# Edit .env with your OpenAI and Pinecone API keys
```

### 3. Run Tests

```bash
# Smoke tests (FastAPI endpoints)
python tests/test_smoke.py

# Processing tests (no API keys required)
python tests/test_processing.py

# Or use pytest
pytest tests/
```

Tests run without API keys (they skip network calls gracefully).

### 4. Start the API Server

**Windows (PowerShell):**
```powershell
.\scripts\run_local.ps1
```

**Or manually:**
```powershell
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
```

**Unix/Linux/Mac:**
```bash
PYTHONPATH=$PWD uvicorn app:app --reload
```

Access the API at:
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/api/v1/metrics

### 5. Process Documents via CLI

```bash
# Single document
python -m src.m1_3_document_processing.module --process data/example/example_data.txt --index production-rag

# Batch processing
python -m src.m1_3_document_processing.module --process-batch docs/ --index production-rag --chunker semantic

# Try different chunking strategies
python -m src.m1_3_document_processing.module --process data/example/example_data.txt --chunker fixed --chunk-size 256
python -m src.m1_3_document_processing.module --process data/example/example_data.txt --chunker paragraph
```

### 6. Process Documents via API

```bash
# Ingest a document
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/example/example_data.txt",
    "chunker": "semantic",
    "chunk_size": 512,
    "index_name": "production-rag"
  }'

# Check metrics
curl http://localhost:8000/api/v1/metrics
```

### 7. Explore the Notebook

```bash
# Set PYTHONPATH and start Jupyter
export PYTHONPATH=$PWD  # Unix/Mac
# or
$env:PYTHONPATH="$PWD"  # Windows PowerShell

jupyter notebook notebooks/M1_3_Document_Processing_Pipeline.ipynb
```

---

## Project Structure

```
.
├── app.py                          # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # MIT License
├── README.md                       # This file
│
├── src/
│   └── m1_3_document_processing/   # Main package
│       ├── __init__.py             # Package init with learning arc
│       ├── config.py               # Configuration and client initialization
│       ├── module.py               # Core processing pipeline
│       └── router.py               # FastAPI routes
│
├── data/
│   └── example/
│       └── example_data.txt        # Sample document for testing
│
├── notebooks/
│   └── M1_3_Document_Processing_Pipeline.ipynb  # Interactive tutorial
│
├── tests/
│   ├── test_smoke.py               # FastAPI endpoint tests
│   └── test_processing.py          # Processing pipeline tests
│
└── scripts/
    └── run_local.ps1               # PowerShell script to run locally
```

---

## Pipeline Architecture

```
┌─────────────┐
│  Raw Doc    │  PDF, TXT, Markdown
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Extraction  │  PyMuPDF, plain text readers
└──────┬──────┘  → Document(text, metadata)
       │
       ▼
┌─────────────┐
│  Cleaning   │  Unicode normalization, artifact removal
└──────┬──────┘  → Cleaned text
       │
       ▼
┌─────────────┐
│  Chunking   │  Fixed / Semantic / Paragraph
└──────┬──────┘  → List of text chunks
       │
       ▼
┌─────────────┐
│  Metadata   │  IDs, hash, counts, semantic flags
└──────┬──────┘  → Chunks with enriched metadata
       │
       ▼
┌─────────────┐
│ Embeddings  │  OpenAI text-embedding-3-small
└──────┬──────┘  → Vectors (1536-dim)
       │
       ▼
┌─────────────┐
│  Pinecone   │  Batch upsert with metadata
└─────────────┘  → Searchable index
```

---

## API Reference

### Endpoints

#### `GET /health`
Health check for the service.

**Response:**
```json
{
  "status": "ok",
  "service": "document-processing-api"
}
```

#### `GET /api/v1/health`
Module-specific health check.

**Response:**
```json
{
  "status": "ok",
  "module": "m1_3_document_processing",
  "version": "1.0.0"
}
```

#### `POST /api/v1/ingest`
Process and ingest documents.

**Request:**
```json
{
  "file_path": "data/example/example_data.txt",
  "chunker": "semantic",
  "chunk_size": 512,
  "index_name": "production-rag"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Processed 1 document(s) into 15 chunks",
  "chunks_processed": 15,
  "documents_processed": 1,
  "skipped": false
}
```

If API keys are not configured, embedding/storage is skipped:
```json
{
  "status": "success",
  "message": "Processed 1 document(s) into 15 chunks (embedding/storage skipped)",
  "chunks_processed": 15,
  "documents_processed": 1,
  "skipped": true,
  "skip_reason": "⚠️ API keys not configured"
}
```

#### `POST /api/v1/query`
Query documents (stub for M1.4).

**Response:**
```json
{
  "status": "not_implemented",
  "message": "Query functionality will be implemented in M1.4",
  "results": []
}
```

#### `GET /api/v1/metrics`
Get pipeline metrics.

**Response:**
```json
{
  "status": "ok",
  "total_documents_processed": 5,
  "total_chunks_generated": 87,
  "api_keys_configured": {
    "openai": true,
    "pinecone": true
  }
}
```

---

## Chunking Strategies

### Fixed-Size Chunking

```python
from src.m1_3_document_processing import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(text)
```

**Pros**: Fast, predictable
**Cons**: May split mid-sentence

### Semantic Chunking

```python
from src.m1_3_document_processing import SemanticChunker

chunker = SemanticChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(text)
```

**Pros**: Preserves meaning, respects boundaries
**Cons**: Slower, variable chunk sizes

### Paragraph Chunking

```python
from src.m1_3_document_processing import ParagraphChunker

chunker = ParagraphChunker(max_chunk_size=1024)
chunks = chunker.chunk(text)
```

**Pros**: Preserves document structure
**Cons**: Requires clear paragraph boundaries

---

## Trade-offs & Costs

### What This Gives You

- **Semantic search**: Find content by meaning, not just keywords
- **Automation**: Process hundreds of documents without manual tagging
- **Flexibility**: Custom chunking and metadata for your domain

### What This Costs

- **Precision loss**: Exact formatting (tables, images) is sacrificed
- **Complexity**: More moving parts = more failure modes
- **Latency**: Embedding 1000 chunks takes ~30 seconds
- **$$**: OpenAI embeddings cost ~$0.13 per million tokens

### What This Doesn't Do

- **OCR**: Scanned PDFs with images (use Tesseract or cloud OCR first)
- **Cross-document reasoning**: No knowledge graph or entity linking
- **Real-time updates**: Reprocessing takes time; not suitable for live docs

---

## Common Failures & Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Unicode errors** (`�` characters) | Smart quotes, special chars | TextCleaner normalizes to ASCII equivalents |
| **Memory exhaustion** | 500MB PDF crashes extractor | Process page-by-page or use streaming reader |
| **Bad chunking** | Code split mid-function | Use `contains_code` metadata flag, custom splitter |
| **Garbled tables** | PDF tables become unstructured text | Use `pdfplumber` or Camelot for table extraction |
| **Duplicate chunks** | Reprocessing same doc | Check `content_hash` before upserting |
| **Metadata too large** | 50KB metadata exceeds Pinecone limit | `_trim_metadata()` removes large fields |

---

## When NOT to Build This

### Use Managed Services If:

- You need **OCR, table extraction, or multi-format support** (Unstructured.io, AWS Textract)
- You process **<50 documents** (manual chunking may be faster)
- You need **real-time updates** (streaming pipelines required)

### Use Long-Context Embeddings If:

- Your documents are **<100K tokens** (e.g., Voyage AI long-context models)
- You need **entire document context** without chunking loss

### Build Custom Pipeline If:

- You have **50+ consistent documents** (e.g., all technical PDFs)
- You need **domain-specific chunking** (e.g., legal contracts by clause)
- You want **full control** over metadata and storage

---

## Decision Card

**Input**: 100 PDF research papers (avg 20 pages)
**Goal**: Semantic search for citations
**Best Choice**: Custom pipeline with paragraph chunking + citation metadata

**Input**: 5 scanned invoices
**Goal**: Extract line items
**Best Choice**: AWS Textract (OCR + table extraction)

**Input**: 1000 Wikipedia articles
**Goal**: General Q&A
**Best Choice**: Long-context embeddings (entire article context)

---

## CLI Reference

### Process Single Document

```bash
python -m src.m1_3_document_processing.module \
  --process data/example/example_data.txt \
  --index production-rag \
  --chunker semantic \
  --chunk-size 512
```

### Process Batch

```bash
python -m src.m1_3_document_processing.module \
  --process-batch docs/ \
  --index production-rag \
  --chunker paragraph
```

### Available Options

- `--process <path>`: Process a single document
- `--process-batch <dir>`: Process all PDFs, TXTs, and MDs in a directory
- `--index <name>`: Pinecone index name (default: `production-rag`)
- `--chunker <type>`: Chunking strategy: `fixed`, `semantic`, or `paragraph` (default: `semantic`)
- `--chunk-size <int>`: Target chunk size in characters (default: 512)

---

## Windows Quick Start

For Windows users, use PowerShell commands:

### Run API Server
```powershell
.\scripts\run_local.ps1
```

### Process a Document
```powershell
$env:PYTHONPATH="$PWD"
python -m src.m1_3_document_processing.module --process data/example/example_data.txt
```

### Run Tests
```powershell
$env:PYTHONPATH="$PWD"
pytest tests/
```

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Smoke tests only
pytest tests/test_smoke.py -v

# Processing tests only
pytest tests/test_processing.py -v
```

### Code Style

The codebase follows:
- **Google-style docstrings** for all public functions/classes
- **Type hints** for function signatures
- **Intent comments** for non-obvious logic
- **Logging** at INFO and ERROR levels

---

## Next Module

**M1.4 — Query Pipeline & Response Generation**: Learn to query your Pinecone index, implement hybrid search (semantic + keyword), and handle retrieval failures.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Resources

- [OpenAI Embeddings Pricing](https://openai.com/pricing)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [PyMuPDF (fitz) Docs](https://pymupdf.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
