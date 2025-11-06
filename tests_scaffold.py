#!/usr/bin/env python3
"""
Test Structure Scaffolder
Creates idempotent test directory structures for portfolio projects.

Usage:
    python tests_scaffold.py [--path /path/to/project]

This script creates a complete test structure with:
- Test directories for each module
- Basic test files with examples
- Fixtures directory
- pytest configuration
- Test utilities
"""

import argparse
import os
from pathlib import Path
from typing import List


class TestScaffolder:
    """Creates test directory structures for projects."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.tests_root = self.base_path / "tests"
        self.created_items: List[str] = []

    def create_structure(self) -> None:
        """Create the complete test structure."""
        print(f"Creating test structure in: {self.tests_root}\n")

        # Create directories
        self._create_directories()

        # Create configuration files
        self._create_config_files()

        # Create test files
        self._create_test_files()

        # Create fixtures
        self._create_fixtures()

        # Create utilities
        self._create_utilities()

        # Summary
        self._print_summary()

    def _create_directories(self) -> None:
        """Create test directory structure."""
        dirs = [
            "",
            "unit",
            "integration",
            "fixtures",
            "utils",
            "unit/core",
            "unit/api",
            "integration/api",
        ]

        for dir_path in dirs:
            full_path = self.tests_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

            # Create __init__.py for each test directory
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Test package."""\n')
                self.created_items.append(str(init_file))

    def _create_config_files(self) -> None:
        """Create pytest configuration files."""

        # pytest.ini
        pytest_ini = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --cov=backend
    --cov-report=term-missing
    --cov-report=html
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
"""
        self._write_file("pytest.ini", pytest_ini)

        # conftest.py (root)
        conftest = '''"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Paris is the capital of France.",
            "metadata": {"source": "test.txt"}
        },
        {
            "id": "doc2",
            "text": "The Eiffel Tower is in Paris.",
            "metadata": {"source": "test.txt"}
        }
    ]
'''
        self._write_file("conftest.py", conftest)

        # .coveragerc
        coveragerc = """[run]
source = backend
omit =
    */tests/*
    */test_*.py
    */__pycache__/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
"""
        self._write_file(".coveragerc", coveragerc)

    def _create_test_files(self) -> None:
        """Create example test files."""

        # Unit test example
        test_unit = '''"""
Unit tests for search functionality.
"""

import pytest


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_add_exchange(self):
        """Test adding exchanges to memory."""
        from backend.core.search import ConversationMemory

        memory = ConversationMemory(max_history=3)
        memory.add_exchange("Question 1?", "Answer 1")
        memory.add_exchange("Question 2?", "Answer 2")

        assert len(memory.history) == 2
        assert memory.history[0]["question"] == "Question 1?"
        assert memory.history[0]["answer"] == "Answer 1"

    def test_max_history_enforcement(self):
        """Test that max history is enforced."""
        from backend.core.search import ConversationMemory

        memory = ConversationMemory(max_history=2)
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.add_exchange("Q3", "A3")

        assert len(memory.history) == 2
        assert memory.history[0]["question"] == "Q2"
        assert memory.history[1]["question"] == "Q3"

    def test_clear_history(self):
        """Test clearing conversation history."""
        from backend.core.search import ConversationMemory

        memory = ConversationMemory()
        memory.add_exchange("Q1", "A1")
        memory.clear()

        assert len(memory.history) == 0

    def test_get_context(self):
        """Test getting conversation context as string."""
        from backend.core.search import ConversationMemory

        memory = ConversationMemory()
        memory.add_exchange("What is 2+2?", "The answer is 4")

        context = memory.get_context()
        assert "What is 2+2?" in context
        assert "The answer is 4" in context
'''
        self._write_file("unit/core/test_search.py", test_unit)

        # Integration test example
        test_integration = '''"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from backend.api.main import app
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_endpoint_structure(client):
    """Test query endpoint returns expected structure."""
    response = client.post(
        "/api/query",
        json={"question": "test question"}
    )

    # Even if not implemented, should return expected structure
    assert "answer" in response.json()


@pytest.mark.slow
def test_query_endpoint_with_real_data(client):
    """Test query endpoint with real data (slow test)."""
    # This would test with actual API calls
    # Mark as slow so it can be skipped in CI
    pass
'''
        self._write_file("integration/api/test_endpoints.py", test_integration)

    def _create_fixtures(self) -> None:
        """Create test fixtures and sample data."""

        # Sample test data
        sample_data = '''{
    "test_queries": [
        {
            "query": "What is the capital of France?",
            "expected_answer": "Paris",
            "expected_source": "geography.txt"
        },
        {
            "query": "How do I authenticate?",
            "expected_answer": "Use API key",
            "expected_source": "docs.md"
        }
    ],
    "test_documents": [
        {
            "id": "doc1",
            "text": "Paris is the capital of France.",
            "metadata": {"source": "geography.txt", "page": 1}
        },
        {
            "id": "doc2",
            "text": "Use your API key in the Authorization header.",
            "metadata": {"source": "docs.md", "page": 5}
        }
    ]
}
'''
        self._write_file("fixtures/test_data.json", sample_data)

        # Fixtures README
        fixtures_readme = """# Test Fixtures

This directory contains test data and fixtures used across tests.

## Files

- `test_data.json` - Sample queries and documents for testing
- Add more fixtures as needed

## Usage

```python
import json
from pathlib import Path

fixtures_dir = Path(__file__).parent / "fixtures"
with open(fixtures_dir / "test_data.json") as f:
    test_data = json.load(f)
```

Or use the pytest fixture:

```python
def test_something(test_data_dir):
    data_file = test_data_dir / "test_data.json"
    # use data_file
```
"""
        self._write_file("fixtures/README.md", fixtures_readme)

    def _create_utilities(self) -> None:
        """Create test utilities."""

        test_utils = '''"""
Test utilities and helper functions.
"""

from typing import Dict, Any, List
import time


def assert_valid_response_time(func, max_ms: int = 1000):
    """Assert function completes within time limit."""
    start = time.time()
    result = func()
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms < max_ms, f"Function took {elapsed_ms:.0f}ms (max: {max_ms}ms)"
    return result


def create_mock_search_results(num_results: int = 5) -> List[Dict[str, Any]]:
    """Create mock search results for testing."""
    return [
        {
            "id": f"doc{i}",
            "score": 0.9 - (i * 0.1),
            "text": f"Sample document text {i}",
            "metadata": {"source": f"doc{i}.txt"}
        }
        for i in range(num_results)
    ]


def create_mock_llm_response(question: str) -> Dict[str, Any]:
    """Create mock LLM response for testing."""
    return {
        "answer": f"This is a mock answer to: {question}",
        "citations": [{"id": "doc1", "source": "test.txt"}],
        "code": None,
        "sources": ["test.txt"]
    }
'''
        self._write_file("utils/helpers.py", test_utils)

    def _write_file(self, relative_path: str, content: str) -> None:
        """Write a file relative to tests root."""
        file_path = self.tests_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Only create if doesn't exist (idempotent)
        if not file_path.exists():
            file_path.write_text(content)
            self.created_items.append(str(file_path))
        else:
            print(f"⏭️  Skipped (exists): {relative_path}")

    def _print_summary(self) -> None:
        """Print creation summary."""
        print("\n" + "="*70)
        print("✅ Test Structure Created Successfully!")
        print("="*70)
        print(f"\nLocation: {self.tests_root}")
        print(f"Created {len(self.created_items)} new files/directories")

        print("\n📋 Next Steps:")
        print("1. Review the test examples in tests/unit/ and tests/integration/")
        print("2. Run tests: pytest tests/")
        print("3. Check coverage: pytest --cov=backend tests/")
        print("4. Add more tests for your specific functionality")

        print("\n💡 Test Commands:")
        print("  pytest tests/                    # Run all tests")
        print("  pytest tests/unit/               # Run only unit tests")
        print("  pytest -m 'not slow'            # Skip slow tests")
        print("  pytest --cov=backend tests/     # Run with coverage")
        print("  pytest -v                        # Verbose output")

        print("\n📚 Test Structure:")
        print("  tests/")
        print("  ├── unit/           # Fast, isolated tests")
        print("  ├── integration/    # API and system tests")
        print("  ├── fixtures/       # Test data")
        print("  └── utils/          # Test helpers")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create idempotent test structure for projects"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Base path for test creation (default: current directory)"
    )

    args = parser.parse_args()

    scaffolder = TestScaffolder(args.path)
    scaffolder.create_structure()


if __name__ == "__main__":
    main()
