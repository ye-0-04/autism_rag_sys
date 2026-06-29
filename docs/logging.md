# Autism Nutrition RAG System — Implementation Log

## Date: 2026-06-30

### Task: Create Project Structure Files

Created the full project directory structure and all source files based on `autism_rag_architecture.md` and `autism_rag_implementation_guide.md`.

#### Files Created

**Infrastructure**
- `.env` — Environment variables for all services
- `docker-compose.yml` — Docker Compose with ChromaDB, Redis, vLLM, and API service
- `Dockerfile` — Python 3.10-slim container with system deps for OpenCV/PaddleOCR
- `requirements.txt` — All Python dependencies (FastAPI, Pydantic, PaddleOCR, ChromaDB, etc.)

**App Core**
- `app/__init__.py`
- `app/config.py` — Pydantic Settings with `LLM_BACKEND` switcher + `get_llm_provider()` factory
- `app/main.py` — FastAPI entrypoint with `/health` and `/generate-nutrition-plan` endpoints
- `app/orchestrator.py` — `RAGOrchestrator` coordinating OCR -> Retrieval -> LLM -> Formatter pipeline
- `app/prompts.py` — System prompt + user prompt template for LLM generation
- `app/formatter.py` — `OutputFormatter` with JSON parsing, retry logic, and fallback plan

**Models**
- `app/models/__init__.py`
- `app/models/genetic_profile.py` — `GeneticMarker` and `GeneticProfile` Pydantic models
- `app/models/nutrition_plan.py` — `DailyTargets`, `Supplement`, `NutritionPlanContent`, `NutritionPlan` models

**OCR Module**
- `app/ocr/__init__.py`
- `app/ocr/preprocessor.py` — `ImagePreprocessor`: PDF->images, grayscale, denoise, deskew, binarize
- `app/ocr/extractor.py` — `OCRExtractor`: PaddleOCR integration with confidence filtering
- `app/ocr/parser.py` — `GeneticDataParser`: regex extraction of gene/variant/status from OCR text

**LLM Providers (Provider-Agnostic Layer)**
- `app/llm/__init__.py`
- `app/llm/base.py` — `LLMProvider` ABC + `LLMResponse` dataclass
- `app/llm/local_vllm.py` — `LocalvLLMProvider` via vLLM OpenAI-compatible API
- `app/llm/ollama.py` — `OllamaProvider`
- `app/llm/openai_provider.py` — `OpenAIProvider`
- `app/llm/anthropic_provider.py` — `AnthropicProvider`

**Retriever Module**
- `app/retriever/__init__.py`
- `app/retriever/ingestion.py` — PDF text extraction + semantic chunking utilities
- `app/retriever/retriever.py` — `NutritionRetriever`: ChromaDB similarity search
- `app/retriever/reranker.py` — `CrossEncoderReranker` (optional precision improvement)

**Middleware**
- `app/middleware/__init__.py`
- `app/middleware/auth.py` — API key authentication via `X-API-Key` header

**Scripts**
- `scripts/ingest_knowledge_base.py` — CLI to load PDFs into ChromaDB

**Tests**
- `tests/__init__.py`
- `tests/test_ocr.py` — 8 tests covering preprocessor, extractor, parser, and full OCR pipeline
- `tests/test_llm.py` — 4 tests for provider factory, health check, generation, and interface consistency
- `tests/test_retriever.py` — 4 tests for collection, retrieval, edge cases, and score ordering
- `tests/test_formatter.py` — 4 tests for JSON parsing, retry logic, fallback, and doctor review flag
- `tests/test_orchestrator.py` — Full end-to-end integration test
- `tests/test_api.py` — 6 tests for health, auth, validation, and happy path

**Directories Created**
- `knowledge_base/documents/` — Place doctor-approved nutrition PDFs here
- `sample_data/` — Place sample genetics report PDFs here
- `docs/` — Documentation and logs

---

### Task 1.1 — Docker Compose Setup (2026-06-30)

**Goal:** Get ChromaDB, Redis, and Ollama running and reachable.

**Changes made during setup:**
- Updated `.env` to use `LLM_BACKEND=ollama` with `qwen3.5:0.8b` model (user's local model)
- Removed vLLM from `docker-compose.yml` (replaced by local Ollama)
- Removed `version` key from `docker-compose.yml` (deprecated warning)
- Removed vLLM dependency from `api` service

**Running services:**
| Service | Status | Port |
|---------|--------|------|
| ChromaDB | Up | 8001 → 8000 |
| Redis | Up | 6379 |
| Ollama (native) | Up | 11434 |

**Verification tests (Task 1.1):**
| Test | Result |
|------|--------|
| `GET /api/v2/heartbeat` (ChromaDB) | ✅ `{"nanosecond heartbeat": ...}` |
| `redis-cli ping` (Redis) | ✅ `PONG` |
| `GET /api/tags` (Ollama) | ✅ Model `qwen3.5:0.8b` listed |
| Ollama chat inference | ✅ Model responds (thinking model, ~47s total) |

**Notes:**
- ChromaDB v1 heartbeat endpoint is deprecated; using v2 endpoint instead
- vLLM was removed in favor of user's local Ollama with `qwen3.5:0.8b`
- The model is a thinking/chain-of-thought variant (response includes `thinking` field)
- RTX 3050 GPU available (4GB VRAM) — sufficient for the 0.8B model

---

### Task 1.2 — Verify GPU Inference via Ollama (2026-06-30)

**Goal:** Confirm `qwen3.5:0.8b` runs on GPU and returns responses at acceptable speed.

**GPU info:** NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM

**Benchmark results (5 high-protein foods, 100 tokens):**
| Run | Time |
|-----|------|
| 1 (cold start) | 4.00s |
| 2 | 1.38s |
| 3 | 1.32s |
| **Average** | **2.23s** |

**Target:** Under 10s → ✅ PASS

**GPU utilization during inference:** 76% | Memory: 1365 MiB

**Test 1.2 verification:**
| Check | Result |
|-------|--------|
| Response < 10s | ✅ Avg 2.23s |
| GPU active during inference | ✅ 76% util |
| Model is reachable | ✅ qwen3.5:0.8b |

**⚠️ Observed issue:** `qwen3.5:0.8b` is a thinking-model variant. It puts its reasoning in the `thinking` field and leaves `content` empty. The `OllamaProvider` in `app/llm/ollama.py` reads `data["message"]["content"]` which will return empty strings. This needs to be addressed:
- Option A: Fallback to `thinking` field if `content` is empty
- Option B: Use a non-thinking model variant
- Option C: Prompt the model to not use thinking mode

**Fixes applied during testing:**
- Updated `.env`: `OLLAMA_BASE_URL=http://localhost:11434` (was `host.docker.internal`), `CHROMA_HOST=localhost`, `CHROMA_PORT=8001`
- Updated `app/config.py`: Pydantic v2 `model_config` with `extra="ignore"`, default `LLM_BACKEND=ollama`, default `OLLAMA_MODEL_NAME=qwen3.5:0.8b`
- Adapted `tests/test_llm.py` from vLLM to Ollama: replaced vLLM tests with Ollama tests
- Installed missing deps: `pydantic-settings`, `pytest-asyncio`, `openai`, `anthropic`

**Python test results (`pytest tests/test_llm.py -v`):**
| Test | Result |
|-----|--------|
| `test_provider_factory_returns_correct_type` | ✅ PASS |
| `test_ollama_health_check` | ✅ PASS |
| `test_ollama_generate_returns_llmresponse` | ✅ PASS |
| `test_provider_interface_is_consistent` | ✅ PASS |
