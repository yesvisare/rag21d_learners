"""
M1.3 — Document Processing Pipeline
Extraction → Cleaning → Chunking → Embedding → Storage

This module implements a production-ready document processing pipeline for RAG systems.
Handles PDF, TXT, and Markdown files with robust error handling and metadata enrichment.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import unicodedata

import fitz  # PyMuPDF
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents an extracted document with metadata."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    file_path: str


@dataclass
class Chunk:
    """Represents a text chunk with enriched metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class DocumentExtractor:
    """
    Extracts text and metadata from PDF, TXT, and Markdown files.

    Supports:
    - PDF: via PyMuPDF (per-page extraction)
    - TXT: plain text files
    - Markdown: .md files
    """

    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

    def extract(self, file_path: str) -> Document:
        """
        Extract text and metadata from a document.

        Args:
            file_path: Path to the document file

        Returns:
            Document object with text and metadata

        Raises:
            ValueError: If file type is unsupported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Generate document ID
        doc_id = self._generate_doc_id(file_path)

        # Extract based on file type
        if extension == '.pdf':
            text, metadata = self._extract_pdf(file_path)
        else:  # .txt or .md
            text, metadata = self._extract_text(file_path)

        # Add common metadata
        metadata.update({
            'file_name': path.name,
            'file_path': str(path.absolute()),
            'file_size': path.stat().st_size,
            'file_type': extension[1:]  # Remove leading dot
        })

        logger.info(f"Extracted document: {path.name} ({len(text)} chars, {metadata.get('page_count', 1)} pages)")

        return Document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            file_path=file_path
        )

    def _extract_pdf(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            pages_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pages_text.append(page.get_text())

            doc.close()

            full_text = "\n\n".join(pages_text)
            metadata = {
                'page_count': len(pages_text),
                'extraction_method': 'PyMuPDF'
            }

            return full_text, metadata

        except Exception as e:
            logger.error(f"Failed to extract PDF {file_path}: {e}")
            raise

    def _extract_text(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from TXT or Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            metadata = {
                'page_count': 1,
                'extraction_method': 'plain_text'
            }

            return text, metadata

        except UnicodeDecodeError:
            logger.error(f"Unicode decode error for {file_path}. Trying latin-1 encoding.")
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

            metadata = {
                'page_count': 1,
                'extraction_method': 'plain_text',
                'encoding_fallback': 'latin-1'
            }

            return text, metadata

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file path."""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]


class TextCleaner:
    """
    Cleans and normalizes extracted text.

    Operations:
    - Whitespace regularization
    - Unicode normalization (smart quotes → standard)
    - Artifact removal (headers, footers, page numbers)
    - Line-break fixes (hyphenated words)
    """

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Unicode normalization (NFKC form)
        text = unicodedata.normalize('NFKC', text)

        # Fix smart quotes and dashes
        text = self._normalize_punctuation(text)

        # Fix hyphenated line breaks (e.g., "exam-\nple" → "example")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Regularize whitespace
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple blank lines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs
        text = re.sub(r' \n', '\n', text)  # Trailing spaces

        # Remove common artifacts (page numbers at line start)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # Strip leading/trailing whitespace
        text = text.strip()

        logger.info(f"Cleaned text: {len(text)} chars")
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize smart quotes, apostrophes, and dashes."""
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text


class FixedSizeChunker:
    """
    Chunks text into fixed-size segments with configurable overlap.

    Fast but may split mid-sentence.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)

            start += (self.chunk_size - self.overlap)

        logger.info(f"Fixed-size chunking: {len(chunks)} chunks")
        return chunks


class SemanticChunker:
    """
    Chunks text recursively by semantic boundaries.

    Tries to split on: paragraphs → sentences → words.
    Preserves meaning better than fixed-size chunking.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text using semantic boundaries."""
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If paragraph is too large, split by sentences
            if para_size > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph
                chunks.extend(self._split_large_text(para))

            elif current_size + para_size > self.chunk_size:
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                # Add overlap from previous chunk
                if chunks and self.overlap > 0:
                    overlap_text = chunks[-1][-self.overlap:]
                    current_chunk = [overlap_text, para]
                    current_size = len(overlap_text) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size

            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f"Semantic chunking: {len(chunks)} chunks")
        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """Split large text by sentences."""
        # Simple sentence splitter
        sentences = re.split(r'([.!?]+\s+)', text)

        chunks = []
        current = []
        current_size = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            full_sentence = sentence + punct

            if current_size + len(full_sentence) > self.chunk_size and current:
                chunks.append(''.join(current))
                current = [full_sentence]
                current_size = len(full_sentence)
            else:
                current.append(full_sentence)
                current_size += len(full_sentence)

        if current:
            chunks.append(''.join(current))

        return chunks


class ParagraphChunker:
    """
    Chunks text while preserving paragraph boundaries.

    Best for documents with clear paragraph structure.
    """

    def __init__(self, max_chunk_size: int = 1024):
        """
        Initialize paragraph chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """Split text by paragraphs, grouping small ones."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if para_size > self.max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Add large paragraph as separate chunk
                chunks.append(para)

            elif current_size + para_size > self.max_chunk_size:
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                current_chunk = [para]
                current_size = para_size

            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f"Paragraph chunking: {len(chunks)} chunks")
        return chunks


class MetadataExtractor:
    """
    Enriches chunks with metadata for filtering and retrieval.

    Metadata includes:
    - chunk_id, content_hash
    - word_count, char_count, line_count
    - contains_code, is_list, has_heading
    """

    def extract(self, chunk_text: str, doc_metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        Extract metadata from a chunk.

        Args:
            chunk_text: The chunk text
            doc_metadata: Document-level metadata
            chunk_index: Index of this chunk in the document

        Returns:
            Dictionary of metadata
        """
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
        chunk_id = f"{doc_metadata.get('file_name', 'unknown')}_{chunk_index}_{content_hash}"

        metadata = {
            'chunk_id': chunk_id,
            'content_hash': content_hash,
            'chunk_index': chunk_index,
            'word_count': len(chunk_text.split()),
            'char_count': len(chunk_text),
            'line_count': chunk_text.count('\n') + 1,
            'contains_code': self._contains_code(chunk_text),
            'is_list': self._is_list(chunk_text),
            'has_heading': self._has_heading(chunk_text),
        }

        # Add document-level metadata (selective)
        for key in ['file_name', 'file_type', 'doc_id']:
            if key in doc_metadata:
                metadata[key] = doc_metadata[key]

        return metadata

    def _contains_code(self, text: str) -> bool:
        """Detect if chunk contains code."""
        code_indicators = [
            r'def \w+\(',  # Python function
            r'function \w+\(',  # JavaScript function
            r'class \w+',  # Class definition
            r'\bimport\b',  # Import statement
            r'=>',  # Arrow function
            r'```',  # Code fence
        ]

        for pattern in code_indicators:
            if re.search(pattern, text):
                return True

        return False

    def _is_list(self, text: str) -> bool:
        """Detect if chunk is primarily a list."""
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False

        # Count lines starting with list markers
        list_lines = sum(1 for line in lines if re.match(r'^\s*[-*•\d]+[.)]\s', line))

        return list_lines / len(lines) > 0.5

    def _has_heading(self, text: str) -> bool:
        """Detect if chunk starts with a heading."""
        first_line = text.strip().split('\n')[0]

        # Markdown heading
        if re.match(r'^#{1,6}\s', first_line):
            return True

        # All caps (potential heading)
        if first_line.isupper() and len(first_line) < 100:
            return True

        return False


class EmbeddingPipeline:
    """
    Generates embeddings and stores them in Pinecone.

    Handles:
    - Batch embedding generation
    - Metadata size limits
    - Error recovery
    """

    def __init__(self, openai_client=None, pinecone_client=None, index_name: str = None):
        """
        Initialize embedding pipeline.

        Args:
            openai_client: OpenAI client instance
            pinecone_client: Pinecone client instance
            index_name: Name of Pinecone index
        """
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index_name = index_name
        self.index = None

        if pinecone_client and index_name:
            try:
                self.index = pinecone_client.Index(index_name)
                logger.info(f"Connected to Pinecone index: {index_name}")
            except Exception as e:
                logger.warning(f"Could not connect to Pinecone index: {e}")

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to embed at once

        Returns:
            List of dictionaries with id, embedding, and metadata
        """
        if not self.openai_client:
            logger.warning("⚠️  Skipping embeddings (no OpenAI client)")
            return []

        vectors = []

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]

            try:
                # Generate embeddings
                from config import OPENAI_MODEL
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model=OPENAI_MODEL
                )

                # Create vectors
                for j, chunk in enumerate(batch):
                    embedding = response.data[j].embedding

                    # Trim metadata if too large (Pinecone limit: 40KB per vector)
                    metadata = self._trim_metadata(chunk.metadata)

                    vectors.append({
                        'id': chunk.chunk_id,
                        'values': embedding,
                        'metadata': metadata
                    })

            except Exception as e:
                logger.error(f"Failed to embed batch {i}-{i+batch_size}: {e}")
                continue

        logger.info(f"Generated {len(vectors)} embeddings")
        return vectors

    def upsert_to_pinecone(self, vectors: List[Dict[str, Any]], namespace: str = "default") -> int:
        """
        Upsert vectors to Pinecone index.

        Args:
            vectors: List of vector dictionaries
            namespace: Pinecone namespace

        Returns:
            Number of vectors upserted
        """
        if not self.index:
            logger.warning("⚠️  Skipping Pinecone upsert (no index connection)")
            return 0

        try:
            # Batch upsert (Pinecone limit: 100 vectors per request)
            batch_size = 100
            upserted_count = 0

            for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                upserted_count += len(batch)

            logger.info(f"Upserted {upserted_count} vectors to Pinecone")
            return upserted_count

        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")
            return 0

    def _trim_metadata(self, metadata: Dict[str, Any], max_size: int = 30000) -> Dict[str, Any]:
        """
        Trim metadata to fit within Pinecone size limits.

        Args:
            metadata: Original metadata
            max_size: Maximum size in bytes

        Returns:
            Trimmed metadata
        """
        import json

        # Convert to JSON to estimate size
        metadata_str = json.dumps(metadata)

        if len(metadata_str) <= max_size:
            return metadata

        # Remove large fields
        trimmed = metadata.copy()
        large_fields = ['full_text', 'raw_content', 'extracted_text']

        for field in large_fields:
            if field in trimmed:
                del trimmed[field]

        logger.warning(f"Trimmed metadata from {len(metadata_str)} to {len(json.dumps(trimmed))} bytes")
        return trimmed


def process_document(
    file_path: str,
    chunker_type: str = "semantic",
    chunk_size: int = 512,
    openai_client=None,
    pinecone_client=None,
    index_name: str = None
) -> List[Chunk]:
    """
    Process a single document through the full pipeline.

    Args:
        file_path: Path to document
        chunker_type: "fixed", "semantic", or "paragraph"
        chunk_size: Target chunk size
        openai_client: OpenAI client
        pinecone_client: Pinecone client
        index_name: Pinecone index name

    Returns:
        List of processed chunks
    """
    # Extract
    extractor = DocumentExtractor()
    doc = extractor.extract(file_path)

    # Clean
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(doc.text)

    # Chunk
    if chunker_type == "fixed":
        chunker = FixedSizeChunker(chunk_size=chunk_size)
    elif chunker_type == "semantic":
        chunker = SemanticChunker(chunk_size=chunk_size)
    elif chunker_type == "paragraph":
        chunker = ParagraphChunker(max_chunk_size=chunk_size * 2)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")

    chunk_texts = chunker.chunk(cleaned_text)

    # Enrich metadata
    metadata_extractor = MetadataExtractor()
    chunks = []

    for i, chunk_text in enumerate(chunk_texts):
        metadata = metadata_extractor.extract(chunk_text, doc.metadata, i)
        chunks.append(Chunk(
            chunk_id=metadata['chunk_id'],
            text=chunk_text,
            metadata=metadata
        ))

    # Embed and store
    if openai_client or pinecone_client:
        pipeline = EmbeddingPipeline(openai_client, pinecone_client, index_name)
        vectors = pipeline.embed_chunks(chunks)

        if vectors:
            pipeline.upsert_to_pinecone(vectors)

    return chunks


def main():
    """CLI entry point."""
    import argparse
    from config import get_clients, PINECONE_INDEX

    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument('--process', type=str, help='Path to document to process')
    parser.add_argument('--process-batch', type=str, help='Path to directory of documents')
    parser.add_argument('--index', type=str, default=PINECONE_INDEX, help='Pinecone index name')
    parser.add_argument('--chunker', type=str, default='semantic',
                        choices=['fixed', 'semantic', 'paragraph'],
                        help='Chunking strategy')
    parser.add_argument('--chunk-size', type=int, default=512, help='Target chunk size')

    args = parser.parse_args()

    # Get clients
    openai_client, pinecone_client = get_clients()

    if args.process:
        # Process single document
        logger.info(f"Processing document: {args.process}")
        chunks = process_document(
            args.process,
            chunker_type=args.chunker,
            chunk_size=args.chunk_size,
            openai_client=openai_client,
            pinecone_client=pinecone_client,
            index_name=args.index
        )
        logger.info(f"✓ Processed {len(chunks)} chunks")

    elif args.process_batch:
        # Process batch of documents
        batch_dir = Path(args.process_batch)
        if not batch_dir.is_dir():
            logger.error(f"Not a directory: {args.process_batch}")
            return

        files = list(batch_dir.glob('*.pdf')) + list(batch_dir.glob('*.txt')) + list(batch_dir.glob('*.md'))
        logger.info(f"Processing {len(files)} documents from {args.process_batch}")

        for file_path in files:
            try:
                chunks = process_document(
                    str(file_path),
                    chunker_type=args.chunker,
                    chunk_size=args.chunk_size,
                    openai_client=openai_client,
                    pinecone_client=pinecone_client,
                    index_name=args.index
                )
                logger.info(f"✓ {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"✗ Failed to process {file_path.name}: {e}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
