# Lab 3: Document Retrieval System

Semantic search system using ChromaDB and sentence transformers for ARIN 5360.

## Quick Start

```bash
# Install dependencies
uv sync

# Start server
uv run uvicorn src.retrieval.main:app --reload
```

Server starts at http://localhost:8000

## Usage

### Via API

**Check health:**
```bash
curl http://localhost:8000/health
```

### Via Browser

Visit http://localhost:8000 (requires `static/index.html`).

### Testing
```bash
# Run all tests with coverage
uv run pytest
```

### Code Quality
```bash
# Check formatting
uv run ruff format --check

# Format code
uv run ruff format

# Lint
uv run ruff check
```

## Project Structure
```
lab3
├── documents
│   └── sample1.txt
├── pyproject.toml
├── README.md
├── src
│   └── retrieval
│       ├── __init__.py
│       └── main.py
├── static
│   ├── index.html
│   └── style.css
├── tests
│   ├── __init__.py
│   └── test_smoke.py
└── uv.lock
```

## Architecture
* Loader: Reads .txt files from documents/
* Embedder: Converts text to vectors using sentence-transformers
* Store: Manages ChromaDB collection for similarity search
* Retriever: Coordinates components for end-to-end retrieval
* API: FastAPI endpoints for health checks and search

## Adding Documents
Place .txt files in the ```documents/``` directory and restart the server. Documents are indexed automatically at startup.