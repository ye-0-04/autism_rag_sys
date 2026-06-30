# Autism Nutrition RAG System

A Retrieval-Augmented Generation (RAG) system that analyzes genetics test reports via OCR, retrieves relevant nutrition knowledge from a vector database, and generates personalized nutrition plans using a local LLM.

## Architecture

```
User uploads PDF/image → OCR (Tesseract) → Genetic marker extraction
  → ChromaDB retrieval (BAAI/bge-small-en-v1.5 embeddings)
  → Local LLM (Ollama/qwen2.5:1.5b) → Structured nutrition plan JSON
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with WSL2 backend on Windows)
- [Ollama](https://ollama.ai/) — local LLM runtime
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/ye-0-04/autism_rag_sys.git
cd autism_rag_sys
```

### 2. Pull the LLM model

```bash
ollama pull qwen2.5:1.5b
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local dev)
```

### 4. Start supporting services

```bash
docker compose up -d chromadb redis
```

### 5. Ingest the knowledge base

```bash
python scripts/ingest_knowledge_base.py
```

### 6. Start the API (locally)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Test it

```bash
# Health check
curl -s http://localhost:8000/health

# Generate a nutrition plan
# (use a genetics test PDF with markers like MTHFR C677T, COMT V158M, etc.)
# Headers: X-API-Key: your-secret-api-key-change-this
# Body: multipart form with file + patient_id
```

## Docker (full stack)

```bash
# Build and start everything
docker compose up -d --build
```

This starts ChromaDB (port 8001), Redis (6379), and the API (8000).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with LLM & KB status |
| `/generate-nutrition-plan` | POST | Upload genetics report → get nutrition plan |

### Authentication

All endpoints require an `X-API-Key` header matching `API_SECRET_KEY` in `.env`.

### Request format

```
POST /generate-nutrition-plan
Content-Type: multipart/form-data
X-API-Key: your-secret-api-key-change-this

Fields:
  file:       PDF or image (JPEG, PNG, TIFF) — genetics test report
  patient_id: string — unique patient identifier
```

## Testing

```bash
# Run test suite
pytest tests/ -v

# Load testing (requires locust)
locust -f tests/locustfile.py --host http://localhost:8000
```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Settings loader
│   ├── orchestrator.py      # RAG pipeline coordinator
│   ├── prompts.py           # LLM prompt templates
│   ├── formatter.py         # Output JSON parser
│   ├── llm/                 # LLM providers (Ollama, vLLM, OpenAI, Anthropic)
│   ├── ocr/                 # OCR pipeline (Tesseract)
│   ├── retriever/           # ChromaDB retriever
│   └── models/              # Pydantic models
├── knowledge_base/
│   └── documents/           # Nutrition reference PDFs
├── scripts/
│   └── ingest_knowledge_base.py  # KB ingestion
├── tests/
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | LLM provider |
| `OLLAMA_MODEL_NAME` | `qwen2.5:1.5b` | Ollama model |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `API_SECRET_KEY` | `your-secret-api-key-change-this` | Auth key |
| `RATE_LIMIT_PER_MINUTE` | `10` | Rate limit |
