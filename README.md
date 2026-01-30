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

