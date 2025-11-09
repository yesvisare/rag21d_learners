#!/usr/bin/env python3
"""
Portfolio Project Scaffolder - DocuMentor Example
Creates a complete repository structure for a RAG-based portfolio project.

Usage:
    python m4_3_portfolio_scaffold.py [project_name] [--path /output/path]

Example:
    python m4_3_portfolio_scaffold.py DocuMentor --path ./projects
    python m4_3_portfolio_scaffold.py MyProject --path ./output --dry-run
    python m4_3_portfolio_scaffold.py MyProject --path ./output --force --no-frontend
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


class PortfolioScaffolder:
    """Creates professional portfolio project structure."""

    def __init__(
        self,
        project_name: str,
        base_path: str = ".",
        dry_run: bool = False,
        force: bool = False,
        no_frontend: bool = False,
        no_ci: bool = False
    ):
        self.project_name = project_name
        self.base_path = Path(base_path) / project_name.lower()
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []
        self.dry_run = dry_run
        self.force = force
        self.no_frontend = no_frontend
        self.no_ci = no_ci
        self.planned_files: List[str] = []
        self.planned_dirs: List[str] = []

    def create_structure(self) -> None:
        """Create the complete project structure."""
        if self.dry_run:
            print(f"[DRY RUN] Planning portfolio project: {self.project_name}")
        else:
            print(f"Creating portfolio project: {self.project_name}")
        print(f"Location: {self.base_path}\n")

        # Check if directory exists
        if self.base_path.exists() and not self.force and not self.dry_run:
            print(f"❌ Error: Directory {self.base_path} already exists.")
            print("Use --force to overwrite or choose a different path.")
            return

        # Create directory structure
        self._create_directories()

        # Create configuration files
        self._create_config_files()

        # Create backend structure
        self._create_backend_files()

        # Create frontend structure (if not disabled)
        if not self.no_frontend:
            self._create_frontend_files()

        # Create documentation
        self._create_documentation()

        # Create CI/CD (if not disabled)
        if not self.no_ci:
            self._create_cicd()

        # Summary
        self._print_summary()

        # Write summary JSON (if not dry-run)
        if not self.dry_run:
            self._write_summary_json()

    def _create_directories(self) -> None:
        """Create all necessary directories."""
        dirs = [
            "",
            "backend",
            "backend/api",
            "backend/core",
            "backend/ingestion",
            "backend/tests",
            "docs",
            "scripts",
        ]

        # Add CI directories if not disabled
        if not self.no_ci:
            dirs.append(".github/workflows")

        # Add frontend directories if not disabled
        if not self.no_frontend:
            dirs.extend([
                "frontend",
                "frontend/public",
                "frontend/src",
                "frontend/src/components",
            ])

        for dir_path in dirs:
            full_path = self.base_path / dir_path

            if self.dry_run:
                self.planned_dirs.append(str(full_path))
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(str(full_path))

    def _create_config_files(self) -> None:
        """Create root-level configuration files."""

        # .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Dependencies
node_modules/
package-lock.json

# Build
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
"""
        self._write_file(".gitignore", gitignore_content)

        # .env.example
        env_example = """# Vector Database
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=documentor

# LLM Provider
OPENAI_API_KEY=your_openai_key_here

# Optional: Alternative Providers
# ANTHROPIC_API_KEY=your_anthropic_key_here
# QDRANT_URL=http://localhost:6333

# Backend Configuration
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:3000

# Logging
LOG_LEVEL=INFO
"""
        self._write_file(".env.example", env_example)

        # docker-compose.yml
        docker_compose = f"""version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PINECONE_API_KEY=${{PINECONE_API_KEY}}
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
      - LOG_LEVEL=${{LOG_LEVEL:-INFO}}
    volumes:
      - ./backend:/app/backend
    command: uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend/src:/app/src
      - ./frontend/public:/app/public
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    command: npm start
    depends_on:
      - backend

networks:
  default:
    name: {self.project_name.lower()}-network
"""
        self._write_file("docker-compose.yml", docker_compose)

        # Dockerfile (backend)
        dockerfile = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/
COPY setup.py .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        self._write_file("Dockerfile", dockerfile)

        # setup.py
        setup_py = f"""from setuptools import setup, find_packages

setup(
    name="{self.project_name.lower()}",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.3.0",
        "pinecone-client>=2.2.0",
        "rank-bm25>=0.2.2",
        "nltk>=3.8.1",
    ],
    python_requires=">=3.10",
)
"""
        self._write_file("setup.py", setup_py)

        # LICENSE (MIT)
        license_content = f"""MIT License

Copyright (c) 2025 {self.project_name} Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        self._write_file("LICENSE", license_content)

    def _create_backend_files(self) -> None:
        """Create backend Python files with basic structure."""

        # backend/__init__.py
        self._write_file("backend/__init__.py", '"""Backend package for RAG system."""\n')

        # backend/api/__init__.py
        self._write_file("backend/api/__init__.py", '"""API endpoints."""\n')

        # backend/api/main.py
        api_main = """\"\"\"
FastAPI backend for portfolio RAG project.
\"\"\"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio RAG API",
    description="Intelligent documentation assistant with hybrid search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    alpha: float = 0.5


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]
    code: Optional[str] = None
    sources: List[str]


@app.get("/api/health")
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return {
        "status": "healthy",
        "service": "portfolio-rag-api",
        "version": "1.0.0"
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    \"\"\"
    Main query endpoint.
    TODO: Implement search and LLM integration.
    \"\"\"
    logger.info(f"Query received: {request.question}")

    # TODO: Add search engine and LLM integration
    return QueryResponse(
        answer="This is a placeholder response. Implement search logic.",
        citations=[],
        code=None,
        sources=[]
    )


@app.delete("/api/conversation/{session_id}")
async def clear_conversation(session_id: str):
    \"\"\"Clear conversation history.\"\"\"
    # TODO: Implement conversation clearing
    return {"message": f"Conversation {session_id} cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        self._write_file("backend/api/main.py", api_main)

        # backend/core/__init__.py
        self._write_file("backend/core/__init__.py", '"""Core RAG functionality."""\n')

        # backend/core/config.py
        config_py = """\"\"\"
Configuration management for the application.
\"\"\"

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    \"\"\"Application settings.\"\"\"

    # API Keys
    pinecone_api_key: str
    openai_api_key: str
    anthropic_api_key: Optional[str] = None

    # Vector Database
    pinecone_environment: str = "us-east1-gcp"
    pinecone_index_name: str = "documentor"

    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4"

    # Application
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    \"\"\"Get application settings.\"\"\"
    return Settings()
"""
        self._write_file("backend/core/config.py", config_py)

        # backend/core/search.py (stub)
        search_stub = """\"\"\"
Hybrid search implementation placeholder.
TODO: Implement HybridSearchEngine and ConversationMemory classes.
\"\"\"

from typing import List, Dict, Any, Optional


class HybridSearchEngine:
    \"\"\"Combines BM25 and dense vector search.\"\"\"

    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str = "documentor"):
        self.index_name = index_name
        # TODO: Initialize Pinecone and OpenAI clients
        pass

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        \"\"\"Perform hybrid search.\"\"\"
        # TODO: Implement hybrid search logic
        return []


class ConversationMemory:
    \"\"\"Maintains conversation context.\"\"\"

    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str):
        \"\"\"Add Q&A to history.\"\"\"
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self) -> str:
        \"\"\"Get conversation context as string.\"\"\"
        return "\\n".join([f"Q: {ex['question']}\\nA: {ex['answer']}" for ex in self.history])

    def clear(self):
        \"\"\"Clear history.\"\"\"
        self.history = []
"""
        self._write_file("backend/core/search.py", search_stub)

        # backend/core/llm.py (stub)
        llm_stub = """\"\"\"
LLM integration placeholder.
TODO: Implement DocumentationAssistant class.
\"\"\"

from typing import List, Dict, Any


class DocumentationAssistant:
    \"\"\"Generates answers with citations.\"\"\"

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.model = model
        # TODO: Initialize OpenAI client
        pass

    def generate_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        \"\"\"Generate answer with citations.\"\"\"
        # TODO: Implement answer generation
        return {
            "answer": "Placeholder answer",
            "citations": [],
            "code": None,
            "sources": []
        }
"""
        self._write_file("backend/core/llm.py", llm_stub)

        # backend/ingestion/__init__.py
        self._write_file("backend/ingestion/__init__.py", '"""Data ingestion modules."""\n')

        # backend/ingestion/scraper.py (stub)
        scraper_stub = """\"\"\"
Documentation scraper placeholder.
TODO: Implement documentation scraping logic.
\"\"\"

from typing import List, Dict, Any


class DocumentationScraper:
    \"\"\"Scrapes and processes documentation.\"\"\"

    def scrape(self, url: str) -> List[Dict[str, Any]]:
        \"\"\"Scrape documentation from URL.\"\"\"
        # TODO: Implement scraping logic
        return []

    def chunk_documents(self, documents: List[str], chunk_size: int = 512) -> List[Dict[str, Any]]:
        \"\"\"Chunk documents for indexing.\"\"\"
        # TODO: Implement chunking logic
        return []
"""
        self._write_file("backend/ingestion/scraper.py", scraper_stub)

        # backend/tests/test_search.py
        test_search = """\"\"\"
Tests for search functionality.
\"\"\"

import pytest
from backend.core.search import ConversationMemory


def test_conversation_memory_add_exchange():
    \"\"\"Test adding exchanges to memory.\"\"\"
    memory = ConversationMemory(max_history=3)
    memory.add_exchange("Question 1?", "Answer 1")
    memory.add_exchange("Question 2?", "Answer 2")

    assert len(memory.history) == 2
    assert memory.history[0]["question"] == "Question 1?"


def test_conversation_memory_max_history():
    \"\"\"Test max history enforcement.\"\"\"
    memory = ConversationMemory(max_history=2)
    memory.add_exchange("Q1", "A1")
    memory.add_exchange("Q2", "A2")
    memory.add_exchange("Q3", "A3")

    assert len(memory.history) == 2
    assert memory.history[0]["question"] == "Q2"


def test_conversation_memory_clear():
    \"\"\"Test clearing memory.\"\"\"
    memory = ConversationMemory()
    memory.add_exchange("Q1", "A1")
    memory.clear()

    assert len(memory.history) == 0
"""
        self._write_file("backend/tests/test_search.py", test_search)

    def _create_frontend_files(self) -> None:
        """Create frontend structure with placeholders."""

        # frontend/package.json
        package_json = f"""{{
  "name": "{self.project_name.lower()}-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "react-scripts": "5.0.1"
  }},
  "scripts": {{
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }},
  "eslintConfig": {{
    "extends": [
      "react-app"
    ]
  }},
  "browserslist": {{
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }}
}}
"""
        self._write_file("frontend/package.json", package_json)

        # frontend/Dockerfile
        frontend_dockerfile = """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
"""
        self._write_file("frontend/Dockerfile", frontend_dockerfile)

        # frontend/public/index.html
        index_html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="{self.project_name} - Intelligent Documentation Assistant" />
    <title>{self.project_name}</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
"""
        self._write_file("frontend/public/index.html", index_html)

        # frontend/src/index.jsx
        index_jsx = """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""
        self._write_file("frontend/src/index.jsx", index_jsx)

        # frontend/src/App.jsx
        app_jsx = f"""import React, {{ useState }} from 'react';
import './App.css';

function App() {{
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {{
    e.preventDefault();
    setLoading(true);

    try {{
      const res = await fetch('http://localhost:8000/api/query', {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
        }},
        body: JSON.stringify({{ question: query }}),
      }});

      const data = await res.json();
      setResponse(data);
    }} catch (error) {{
      console.error('Error:', error);
      setResponse({{ answer: 'Error connecting to backend' }});
    }} finally {{
      setLoading(false);
    }}
  }};

  return (
    <div className="App">
      <header>
        <h1>{self.project_name}</h1>
        <p>Intelligent Documentation Assistant</p>
      </header>

      <main>
        <form onSubmit={{handleSubmit}}>
          <input
            type="text"
            value={{query}}
            onChange={{(e) => setQuery(e.target.value)}}
            placeholder="Ask a question about the documentation..."
            disabled={{loading}}
          />
          <button type="submit" disabled={{loading}}>
            {{loading ? 'Searching...' : 'Search'}}
          </button>
        </form>

        {{response && (
          <div className="response">
            <h2>Answer:</h2>
            <p>{{response.answer}}</p>
          </div>
        )}}
      </main>
    </div>
  );
}}

export default App;
"""
        self._write_file("frontend/src/App.jsx", app_jsx)

        # frontend/src/App.css
        app_css = """* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.App {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
}

header {
  text-align: center;
  margin-bottom: 3rem;
}

header h1 {
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

main {
  max-width: 800px;
  margin: 0 auto;
}

form {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
}

input {
  flex: 1;
  padding: 1rem;
  font-size: 1rem;
  border: none;
  border-radius: 8px;
}

button {
  padding: 1rem 2rem;
  font-size: 1rem;
  background: #fff;
  color: #667eea;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: bold;
}

button:hover {
  background: #f0f0f0;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.response {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 2rem;
  border-radius: 12px;
}

.response h2 {
  margin-bottom: 1rem;
}
"""
        self._write_file("frontend/src/App.css", app_css)

    def _create_documentation(self) -> None:
        """Create README and documentation files."""

        readme = f"""# {self.project_name} 🤖📚

An intelligent documentation assistant using hybrid search and LLMs to answer technical questions with accurate citations.

## Features

- 🔍 **Hybrid Search**: Combines semantic and keyword search for accurate retrieval
- 💬 **Conversational**: Maintains context across questions
- 📝 **Citations**: Always provides source documentation
- 💻 **Code Generation**: Creates code examples based on docs
- 🚀 **Fast**: Optimized for sub-second response times

## Tech Stack

- **Vector Database**: Pinecone
- **LLM**: OpenAI GPT-4
- **Backend**: FastAPI, Python 3.10+
- **Frontend**: React
- **Search**: BM25 + Dense Embeddings

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/{self.project_name.lower()}.git
cd {self.project_name.lower()}

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run with Docker
docker-compose up

# Or run locally
pip install -r requirements.txt
cd backend && uvicorn api.main:app --reload
cd frontend && npm install && npm start
```

Visit http://localhost:3000 to start asking questions!

## Project Structure

```
{self.project_name.lower()}/
├── backend/           # FastAPI backend
│   ├── api/          # API endpoints
│   ├── core/         # Core search and LLM logic
│   ├── ingestion/    # Documentation scraping
│   └── tests/        # Unit tests
├── frontend/         # React frontend
│   ├── public/       # Static files
│   └── src/          # React components
├── docs/             # Documentation
├── .github/          # CI/CD workflows
└── docker-compose.yml
```

## Development

```bash
# Run tests
pytest backend/tests/

# Format code
black backend/
isort backend/

# Type checking
mypy backend/
```

## Performance

- Query latency: <500ms average
- Handles 100+ concurrent users
- Index size: 50K document chunks

## Roadmap

- [ ] Multi-language support
- [ ] Custom documentation sources
- [ ] API rate limiting
- [ ] User feedback system

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](linkedin.com/in/yourprofile)

---

Built with modern AI/ML tools for the RAG21D course.
"""
        self._write_file("README.md", readme)

        # CONTRIBUTING.md
        contributing = """# Contributing to This Project

Thank you for your interest in contributing!

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/project.git

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest backend/tests/
```

## Code Standards

- Follow PEP 8 for Python code
- Add type hints to all functions
- Write docstrings for classes and functions
- Include tests for new features
- Keep commits atomic and well-described

## Pull Request Process

1. Update documentation for any API changes
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Request review from maintainers

Thank you for contributing!
"""
        self._write_file("CONTRIBUTING.md", contributing)

    def _create_cicd(self) -> None:
        """Create GitHub Actions CI/CD workflows."""

        tests_workflow = """name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov black isort mypy

      - name: Code formatting check
        run: |
          black --check backend/
          isort --check-only backend/

      - name: Type checking
        run: |
          mypy backend/ --ignore-missing-imports

      - name: Run tests
        run: |
          pytest backend/tests/ -v --cov=backend --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
"""
        self._write_file(".github/workflows/tests.yml", tests_workflow)

    def _write_file(self, relative_path: str, content: str) -> None:
        """Write a file with given content."""
        file_path = self.base_path / relative_path

        if self.dry_run:
            self.planned_files.append(str(file_path))
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.created_files.append(str(file_path))

    def _print_summary(self) -> None:
        """Print creation summary."""
        print("\n" + "="*60)

        if self.dry_run:
            print("📋 DRY RUN PLAN")
            print("="*60)
            print(f"\nProject: {self.project_name}")
            print(f"Location: {self.base_path}")
            print(f"\nWould create {len(self.planned_dirs)} directories")
            print(f"Would create {len(self.planned_files)} files")

            if self.no_frontend:
                print("\n⚠️  Frontend: DISABLED (--no-frontend)")
            if self.no_ci:
                print("⚠️  CI/CD: DISABLED (--no-ci)")

            print("\n📁 Sample directories:")
            for d in self.planned_dirs[:5]:
                print(f"  - {d}")
            if len(self.planned_dirs) > 5:
                print(f"  ... and {len(self.planned_dirs) - 5} more")

            print("\n📄 Sample files:")
            for f in self.planned_files[:10]:
                print(f"  - {f}")
            if len(self.planned_files) > 10:
                print(f"  ... and {len(self.planned_files) - 10} more")

            print("\n💡 To actually create the scaffold, run without --dry-run")
        else:
            print("✅ Portfolio Project Scaffold Created Successfully!")
            print("="*60)
            print(f"\nProject: {self.project_name}")
            print(f"Location: {self.base_path}")
            print(f"\nCreated {len(self.created_dirs)} directories")
            print(f"Created {len(self.created_files)} files")

            print("\n📋 Next Steps:")
            print(f"1. cd {self.base_path}")
            print("2. cp .env.example .env")
            print("3. Edit .env with your API keys")
            print("4. docker-compose up")
            print("\n💡 Tips:")
            print("- Review backend/core/ files and implement TODOs")
            print("- Add your documentation scraping logic")
            if not self.no_frontend:
                print("- Customize frontend components")
            print("- Write comprehensive tests")
            print("- Update README with your specifics")
            print("\n🚀 Happy building!")

    def _write_summary_json(self) -> None:
        """Write scaffold summary to JSON file."""
        summary = {
            "project_name": self.project_name,
            "location": str(self.base_path),
            "created_dirs": self.created_dirs,
            "created_files": self.created_files,
            "options": {
                "no_frontend": self.no_frontend,
                "no_ci": self.no_ci
            }
        }

        summary_path = self.base_path / "scaffold_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Summary written to: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a portfolio project scaffold for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python m4_3_portfolio_scaffold.py DocuMentor --path ./projects

  # Preview without creating files
  python m4_3_portfolio_scaffold.py MyProject --dry-run

  # Skip frontend generation
  python m4_3_portfolio_scaffold.py BackendOnly --no-frontend

  # Force overwrite existing directory
  python m4_3_portfolio_scaffold.py DocuMentor --force
        """
    )
    parser.add_argument(
        "project_name",
        nargs="?",
        default="DocuMentor",
        help="Name of the project (default: DocuMentor)"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Base path for project creation (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing directory"
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Skip frontend (React) structure generation"
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Skip CI/CD (GitHub Actions) workflow generation"
    )

    args = parser.parse_args()

    scaffolder = PortfolioScaffolder(
        args.project_name,
        args.path,
        dry_run=args.dry_run,
        force=args.force,
        no_frontend=args.no_frontend,
        no_ci=args.no_ci
    )
    scaffolder.create_structure()


if __name__ == "__main__":
    main()
