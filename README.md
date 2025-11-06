# M1.3 — Document Processing Pipeline

**Extraction → Cleaning → Chunking → Embedding → Storage**

A production-ready document processing pipeline for RAG systems. Transforms raw documents (PDF, TXT, Markdown) into searchable vector embeddings stored in Pinecone.

## What You'll Learn

This module teaches you to build a complete document processing pipeline:

- **Extract** text from PDFs (PyMuPDF), plain text, and Markdown files
- **Clean** text by normalizing Unicode, removing artifacts, and fixing line breaks
- **Chunk** documents using three strategies: fixed-size, semantic, and paragraph-aware
- **Enrich** chunks with metadata (IDs, hashes, counts, semantic flags)
- **Embed** text using OpenAI's `text-embedding-3-small`
- **Store** vectors in Pinecone with metadata for filtering and retrieval

You'll also learn the honest trade-offs: when to use custom pipelines vs. managed services vs. long-context embeddings.

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Run Tests

```bash
python tests_processing.py
```

Tests run without API keys (they skip network calls gracefully).

### 4. Process a Document

```bash
# Single document
python m1_3_document_processing.py --process example_data.txt --index production-rag

# Batch processing
python m1_3_document_processing.py --process-batch docs/ --index production-rag
```

### 5. Explore the Notebook

Open `M1_3_Document_Processing_Pipeline.ipynb` to see each pipeline stage in action:

```bash
jupyter notebook M1_3_Document_Processing_Pipeline.ipynb
```

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

## Chunking Strategies

### Fixed-Size Chunking

```python
chunker = FixedSizeChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(text)
```

**Pros**: Fast, predictable
**Cons**: May split mid-sentence

### Semantic Chunking

```python
chunker = SemanticChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(text)
```

**Pros**: Preserves meaning, respects boundaries
**Cons**: Slower, variable chunk sizes

### Paragraph Chunking

```python
chunker = ParagraphChunker(max_chunk_size=1024)
chunks = chunker.chunk(text)
```

**Pros**: Preserves document structure
**Cons**: Requires clear paragraph boundaries

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

## Common Failures & Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| **Unicode errors** (`�` characters) | Smart quotes, special chars | TextCleaner normalizes to ASCII equivalents |
| **Memory exhaustion** | 500MB PDF crashes extractor | Process page-by-page or use streaming reader |
| **Bad chunking** | Code split mid-function | Use `contains_code` metadata flag, custom splitter |
| **Garbled tables** | PDF tables become unstructured text | Use `pdfplumber` or Camelot for table extraction |
| **Duplicate chunks** | Reprocessing same doc | Check `content_hash` before upserting |
| **Metadata too large** | 50KB metadata exceeds Pinecone limit | `_trim_metadata()` removes large fields |

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

## CLI Reference

### Process Single Document

```bash
python m1_3_document_processing.py \
  --process path/to/document.pdf \
  --index production-rag \
  --chunker semantic \
  --chunk-size 512
```

### Process Batch

```bash
python m1_3_document_processing.py \
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

## Project Structure

```
.
├── m1_3_document_processing.py   # Main pipeline implementation
├── config.py                     # Configuration & client initialization
├── requirements.txt              # Dependencies
├── .env.example                  # Environment variable template
├── example_data.txt              # Sample document for testing
├── tests_processing.py           # Smoke tests (no API keys required)
├── M1_3_Document_Processing_Pipeline.ipynb  # Interactive tutorial
└── README.md                     # This file
```

## Next Module

**M1.4 — Vector Stores & Retrieval**: Learn to query your Pinecone index, implement hybrid search (semantic + keyword), and handle retrieval failures.

## License

Educational use only. See course materials for full terms.
