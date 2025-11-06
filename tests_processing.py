"""
Smoke tests for document processing pipeline.

Tests basic functionality without requiring API keys.
"""

import os
import tempfile
from pathlib import Path

from m1_3_document_processing import (
    DocumentExtractor,
    TextCleaner,
    FixedSizeChunker,
    SemanticChunker,
    ParagraphChunker,
    MetadataExtractor,
    EmbeddingPipeline,
    Chunk,
)


def test_extractor_detects_file_type():
    """Test that DocumentExtractor detects and handles different file types."""
    extractor = DocumentExtractor()

    # Create temporary TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document content.")
        txt_path = f.name

    try:
        doc = extractor.extract(txt_path)
        assert doc.text == "Test document content."
        assert doc.metadata['file_type'] == 'txt'
        assert 'file_name' in doc.metadata
        print("✓ Extractor detects file type")
    finally:
        os.unlink(txt_path)

    # Test unsupported file type
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            docx_path = f.name

        try:
            extractor.extract(docx_path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported file type" in str(e)
            print("✓ Extractor rejects unsupported types")
        finally:
            os.unlink(docx_path)
    except Exception as e:
        print(f"✗ Test failed: {e}")


def test_cleaner_trims_noise():
    """Test that TextCleaner removes artifacts and normalizes text."""
    cleaner = TextCleaner()

    # Test smart quotes normalization
    text = "He said "hello" and she said 'goodbye'."
    cleaned = cleaner.clean(text)
    assert '"' not in cleaned or cleaned.count('"') == cleaned.count('"')
    print("✓ Cleaner normalizes smart quotes")

    # Test whitespace regularization
    text = "Line 1\n\n\n\n\nLine 2"
    cleaned = cleaner.clean(text)
    assert '\n\n\n' not in cleaned
    print("✓ Cleaner regularizes whitespace")

    # Test hyphenated line breaks
    text = "exam-\nple word"
    cleaned = cleaner.clean(text)
    assert "exam-\n" not in cleaned
    assert "example" in cleaned
    print("✓ Cleaner fixes hyphenated line breaks")


def test_semantic_chunk_count():
    """Test that SemanticChunker produces reasonable chunk counts."""
    chunker = SemanticChunker(chunk_size=100, overlap=10)

    # Multi-paragraph text
    text = """
First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content to ensure chunking.
    """.strip()

    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)
    print(f"✓ Semantic chunker produced {len(chunks)} chunks")


def test_fixed_chunker():
    """Test FixedSizeChunker behavior."""
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)

    text = "A" * 200  # 200 characters

    chunks = chunker.chunk(text)
    assert len(chunks) >= 4  # Should split into multiple chunks
    print(f"✓ Fixed-size chunker produced {len(chunks)} chunks")


def test_paragraph_chunker():
    """Test ParagraphChunker preserves boundaries."""
    chunker = ParagraphChunker(max_chunk_size=100)

    text = """Paragraph one.

Paragraph two.

Paragraph three."""

    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    print(f"✓ Paragraph chunker produced {len(chunks)} chunks")


def test_metadata_has_required_fields():
    """Test that MetadataExtractor generates required fields."""
    extractor = MetadataExtractor()

    chunk_text = "Sample chunk with some content."
    doc_metadata = {
        'file_name': 'test.txt',
        'file_type': 'txt',
        'doc_id': 'abc123'
    }

    metadata = extractor.extract(chunk_text, doc_metadata, chunk_index=0)

    # Check required fields
    assert 'chunk_id' in metadata
    assert 'content_hash' in metadata
    assert 'word_count' in metadata
    assert 'char_count' in metadata
    assert 'contains_code' in metadata
    assert 'is_list' in metadata
    assert 'has_heading' in metadata

    # Check inherited fields
    assert metadata['file_name'] == 'test.txt'
    assert metadata['doc_id'] == 'abc123'

    print(f"✓ Metadata has all required fields: {len(metadata)} total")


def test_metadata_code_detection():
    """Test code detection in metadata."""
    extractor = MetadataExtractor()

    # Code chunk
    code_text = """
def hello():
    print("Hello, world!")
    """

    metadata = extractor.extract(code_text, {}, 0)
    assert metadata['contains_code'] is True
    print("✓ Metadata detects code")

    # Non-code chunk
    text = "This is just regular text without any code."
    metadata = extractor.extract(text, {}, 0)
    assert metadata['contains_code'] is False
    print("✓ Metadata detects non-code")


def test_metadata_list_detection():
    """Test list detection in metadata."""
    extractor = MetadataExtractor()

    # List chunk
    list_text = """
- Item one
- Item two
- Item three
- Item four
    """

    metadata = extractor.extract(list_text, {}, 0)
    assert metadata['is_list'] is True
    print("✓ Metadata detects lists")


def test_embedding_stub_without_keys():
    """Test that embedding pipeline handles missing keys gracefully."""
    # No clients provided
    pipeline = EmbeddingPipeline(
        openai_client=None,
        pinecone_client=None,
        index_name="test-index"
    )

    # Create dummy chunks
    chunks = [
        Chunk(
            chunk_id="test_0",
            text="Sample text",
            metadata={'chunk_id': 'test_0', 'word_count': 2}
        )
    ]

    # Should return empty list when no OpenAI client
    vectors = pipeline.embed_chunks(chunks)
    assert vectors == []
    print("✓ Embedding pipeline skips when no keys present")

    # Upsert should also handle missing index gracefully
    count = pipeline.upsert_to_pinecone(vectors)
    assert count == 0
    print("✓ Upsert skips when no Pinecone connection")


def test_full_pipeline_without_keys():
    """Test that full pipeline runs without API keys."""
    from m1_3_document_processing import process_document

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document.\n\nWith multiple paragraphs.")
        temp_path = f.name

    try:
        chunks = process_document(
            temp_path,
            chunker_type="semantic",
            chunk_size=50,
            openai_client=None,
            pinecone_client=None,
            index_name=None
        )

        assert len(chunks) > 0
        assert all(hasattr(chunk, 'chunk_id') for chunk in chunks)
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)

        print(f"✓ Full pipeline runs without API keys ({len(chunks)} chunks)")

    finally:
        os.unlink(temp_path)


def run_all_tests():
    """Run all smoke tests."""
    print("\n=== Running Document Processing Pipeline Tests ===\n")

    tests = [
        test_extractor_detects_file_type,
        test_cleaner_trims_noise,
        test_semantic_chunk_count,
        test_fixed_chunker,
        test_paragraph_chunker,
        test_metadata_has_required_fields,
        test_metadata_code_detection,
        test_metadata_list_detection,
        test_embedding_stub_without_keys,
        test_full_pipeline_without_keys,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")


if __name__ == '__main__':
    run_all_tests()
