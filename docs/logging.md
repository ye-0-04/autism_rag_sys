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

---

### Task 1.3 — LLM Provider ABC + Provider Implementations (2026-06-30)

**Files:**
- `app/llm/base.py` — `LLMProvider` ABC with `generate()` and `health_check()` abstract methods + `LLMResponse` dataclass
- `app/llm/local_vllm.py` — `LocalvLLMProvider` (vLLM OpenAI-compatible API)
- `app/llm/ollama.py` — `OllamaProvider` (Ollama native API)
- `app/llm/openai_provider.py` — `OpenAIProvider` (OpenAI SDK)
- `app/llm/anthropic_provider.py` — `AnthropicProvider` (Anthropic SDK)

**Fixes during testing:**
- `OllamaProvider`: Added fallback to `thinking` field when `content` is empty — required for thinking-model variants like `qwen3.5:0.8b`

**Test results:**
| Test | Result |
|-----|--------|
| `test_provider_factory_returns_correct_type` | ✅ PASS |
| `test_ollama_health_check` | ✅ PASS |
| `test_ollama_generate_returns_llmresponse` | ✅ PASS (incl. non-empty content) |
| `test_provider_interface_is_consistent` | ✅ PASS |

---

### Task 1.4 — Config Loader (2026-06-30)

**File:** `app/config.py`

- Pydantic v2 `BaseSettings` with `SettingsConfigDict(env_file=".env", extra="ignore")`
- `LLM_BACKEND` Literal type restricts to `local_vllm`, `ollama`, `openai`, `anthropic`
- `get_llm_provider()` factory function returns correct provider based on config
- `.env` updated for local dev: Ollama at `localhost:11434`, ChromaDB at `localhost:8001`

**Verification:** All provider and config-dependent tests pass.

---

### Task 2.1 — Image Preprocessor (2026-06-30)

**Goal:** OCR-ready image preprocessing pipeline (PDF→images, grayscale, denoise, deskew, binarize).

**File:** `app/ocr/preprocessor.py` — `ImagePreprocessor` class with full pipeline.

**Fixes:**
- Restructured `tests/test_ocr.py` imports: moved `OCRExtractor` and `GeneticDataParser` imports into test functions to avoid requiring PaddleOCR for preprocessor tests

**Test results (`pytest tests/test_ocr.py::test_preprocessor_* -v`):**
| Test | Result |
|-----|--------|
| `test_preprocessor_loads_pdf` | ✅ PASS |
| `test_preprocessor_output_is_binary` | ✅ PASS |
| `test_preprocessor_handles_already_grayscale` | ✅ PASS |

---

### Task 2.2 — OCR Extractor (2026-06-30)

**Goal:** Extract text from preprocessed images using OCR.

**Initial approach:** PaddleOCR 2.7.3 with `use_angle_cls=True`, `use_gpu=False`, `show_log=False`.

**Problems encountered and resolutions:**
1. **PaddleOCR v3.7 API change** — User had v3.7.0 installed. `use_gpu` and `show_log` removed; `use_angle_cls` deprecated in favor of `use_textline_orientation`. `ocr()` method deprecated in favor of `predict()`. → Fixed API params locally.
2. **PaddlePaddle 3.0.0 model incompatibility** — After fixing API, `paddlepaddle==3.0.0` could not load pre-trained PP-OCRv6 models (`Type of attribute: strides is not right`). → Decided to downgrade to PaddleOCR 2.7.3 + PaddlePaddle 2.6.2.
3. **Disk space exhaustion** — Multiple installs ate available disk space. → User freed up space (8.5GB available).
4. **PaddlePaddle 2.6.2 + numpy 2.x ABI mismatch** — PaddleOCR 2.7.3 pins `opencv-python<=4.6.0.66`, compiled against numpy 1.x ABI. numpy 2.x ABI mismatch → `cv2` module loaded but had no attributes.
5. **Final resolution: Switched to Tesseract OCR** — Tesseract 5.3.3 already installed via msys2. `pytesseract` wrapper (14KB) installed. `opencv-python-headless==4.13.0.92` compatible with numpy 2.3.5. `eng.traineddata` downloaded for English language support.

**Fixes applied:**
- `app/ocr/extractor.py` — Rewritten from PaddleOCR to `pytesseract` with `--psm 6` config for uniform text blocks, whitelist for alphanumeric + genetic-report symbols, `TESSDATA_PREFIX` env var set programmatically.
- `tests/test_ocr.py` — Added missing `GeneticDataParser` and `GeneticProfile` imports in `test_full_ocr_to_profile_pipeline`.
- `requirements.txt` — Replaced `paddlepaddle==2.6.1`, `paddleocr==2.7.3`, `opencv-python-headless==4.9.0.80`, `numpy==1.26.4` with `pytesseract==0.3.13`, `opencv-python-headless>=4.9.0`, `numpy>=2.0.0`.

**All 10 OCR tests passing:**
| Test | Result |
|-----|--------|
| `test_preprocessor_loads_pdf` | ✅ PASS |
| `test_preprocessor_output_is_binary` | ✅ PASS |
| `test_preprocessor_handles_already_grayscale` | ✅ PASS |
| `test_ocr_extracts_text_from_synthetic_image` | ✅ PASS |
| `test_ocr_returns_empty_string_for_blank_image` | ✅ PASS |
| `test_parser_extracts_mthfr_from_text` | ✅ PASS |
| `test_parser_normalizes_status` | ✅ PASS |
| `test_parser_returns_profile_with_no_markers_gracefully` | ✅ PASS |
| `test_genetic_profile_to_retrieval_query` | ✅ PASS |
| `test_full_ocr_to_profile_pipeline` | ✅ PASS |

**LLM tests still passing (no regressions):**
| Test | Result |
|-----|--------|
| `test_provider_factory_returns_correct_type` | ✅ PASS |
| `test_ollama_health_check` | ✅ PASS |
| `test_ollama_generate_returns_llmresponse` | ✅ PASS |
| `test_provider_interface_is_consistent` | ✅ PASS |

---

### Task 3.1 — Document Ingestion, Task 3.2 — Retriever Module, Task 3.3 — Re-Ranker (2026-06-30)

**Files:**
- `scripts/ingest_knowledge_base.py` — CLI to load PDFs into ChromaDB with SentenceTransformer embeddings
- `app/retriever/ingestion.py` — PDF text extraction + semantic chunking utilities
- `app/retriever/retriever.py` — `NutritionRetriever`: ChromaDB cosine similarity search
- `app/retriever/reranker.py` — `CrossEncoderReranker` (optional precision improvement)

**Notes:**
- chromadb 0.5.0 incompatible with numpy 2.x (`np.float_` removed)
- Upgraded to chromadb 1.5.9 which works with numpy 2.3.5
- `python-multipart` was missing for the API endpoints — installed
- Ingestion ran successfully: sample nutrition PDF → 3 chunks in ChromaDB

**Test results (`pytest tests/test_retriever.py -v`):**
| Test | Result |
|-----|--------|
| `test_retriever_collection_is_not_empty` | ✅ PASS |
| `test_retriever_returns_chunks_for_mthfr_profile` | ✅ PASS |
| `test_retriever_returns_fewer_chunks_than_requested_if_db_is_small` | ✅ PASS |
| `test_retrieval_quality_scores_are_ordered` | ✅ PASS |

---

### Tasks 4.1-4.3 — Prompt Templates, Formatter, Orchestrator (2026-06-30)

**Fixes during testing:**
- API endpoint had `request` parameter without type annotation → 422 errors. Added `request: Request`.
- LLM model `qwen3.5:0.8b` (thinking model) produced empty `content` — answers were in `thinking` field. Fallback picked up reasoning text instead of final answer. Model never output JSON.
- **Solution:** Pulled `qwen2.5:0.5b` (397MB, non-thinking). Outputs valid JSON in `content`.
- `.env` updated: `OLLAMA_MODEL_NAME=qwen2.5:0.5b`
- Formatter bug: `s.get("dose", "")` returns `None` when key exists with `null` in JSON → Pydantic error. Fixed with `or ""` on supplement fields.

**Test results (`pytest tests/test_formatter.py -v`):**
| Test | Result |
|-----|--------|
| `test_formatter_parses_valid_json` | ✅ PASS |
| `test_formatter_retries_on_invalid_json` | ✅ PASS |
| `test_formatter_returns_fallback_after_all_retries_fail` | ✅ PASS |
| `test_requires_doctor_review_is_true_for_low_confidence` | ✅ PASS |

---

### Tasks 5.1-5.5 — FastAPI Application (2026-06-30)

**Test results (`pytest tests/test_api.py -v`):**
| Test | Result |
|-----|--------|
| `test_health_endpoint_returns_ok` | ✅ PASS |
| `test_missing_api_key_returns_401` | ✅ PASS |
| `test_wrong_api_key_returns_401` | ✅ PASS |
| `test_missing_file_returns_422` | ✅ PASS |
| `test_invalid_file_type_returns_415` | ✅ PASS |
| `test_valid_request_returns_nutrition_plan` | ✅ PASS |

---

### Task 6.1 — Full Pipeline Integration Test (2026-06-30)

**Notes:**
- End-to-end: synthetic PDF → OCR → GeneticProfile → Retrieval → LLM → Formatter → NutritionPlan
- Initially failed due to qwen3.5:0.8b thinking model (no JSON output)
- After switching to qwen2.5:0.5b and fixing the formatter None fallback bug

**Test result:**
| Test | Result |
|-----|--------|
| `test_full_pipeline_returns_nutrition_plan` | ✅ PASS |

---

### Full Test Suite Summary (2026-06-30)

**Result:** All **29/29 tests passing.**

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_ocr.py` | 10 | ✅ All pass |
| `tests/test_llm.py` | 4 | ✅ All pass |
| `tests/test_retriever.py` | 4 | ✅ All pass |
| `tests/test_formatter.py` | 4 | ✅ All pass |
| `tests/test_orchestrator.py` | 1 | ✅ Pass |
| `tests/test_api.py` | 6 | ✅ All pass |

**Model:** `qwen2.5:0.5b` (replaced `qwen3.5:0.8b` thinking model)
**OCR:** Tesseract via pytesseract (replaced PaddleOCR)
**ChromaDB:** v1.5.9 (upgraded from 0.5.0 for numpy 2.x compatibility)

---

### Model Upgrade: qwen2.5:0.5b → qwen2.5:1.5b (2026-06-30)

**Motivation:** The 0.5B model produced malformed JSON (unquoted values in `dose` fields like `"dose": 1000-2000mcg daily,`), truncated mid-output, and frequently fell back to the fallback plan.

**Action:**
- Pulled `qwen2.5:1.5b` (986MB) — completed download
- Updated `.env`: `OLLAMA_MODEL_NAME=qwen2.5:1.5b`

**Formatter resilience fix (`app/formatter.py`):**
- The 1.5B model outputs `meal_timing_notes` as `null` and `notes` as a `list` instead of `string`
- Added `or ""` fallback for `meal_timing_notes`
- Added list-to-string conversion for `notes` (joins with ` | `)

**Test results:**
| Test | Result |
|-----|--------|
| `test_full_pipeline_returns_nutrition_plan` | ✅ PASS (with 1.5B) |
| Full suite (30 tests) | ✅ All pass |

**Final test suite (30/30 passing):**

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_ocr.py` | 10 | ✅ All pass |
| `tests/test_llm.py` | 4 | ✅ All pass |
| `tests/test_retriever.py` | 4 | ✅ All pass |
| `tests/test_formatter.py` | 4 | ✅ All pass |
| `tests/test_orchestrator.py` | 1 | ✅ Pass |
| `tests/test_api.py` | 7 | ✅ All pass |

---

### Docker End-to-End Verification (2026-06-30)

**Goal:** Confirm the full stack (ChromaDB + Redis + API) works in Docker.

**Issues fixed:**
- `requirements.txt` — Added `demjson3>=3.0.6` (was missing, causing `ModuleNotFoundError`)
- `requirements.txt` — Added `torch>=1.11.0` to ensure CPU-only torch from earlier install step is recognized (prevents CUDA wheel download)
- `Dockerfile` — Added CPU-only PyTorch install step before main `pip install`, reducing Docker image size from ~2.5GB to ~574MB
- `app/ocr/extractor.py` — Made Tesseract path portable: detects `platform.system()`, uses Windows path only on Windows, Linux uses default `tesseract` from PATH

**Build time:** ~4 minutes (CPU-only PyTorch: 175MB vs CUDA variant: ~2GB)

**Verification results:**

| Endpoint | Result |
|----------|--------|
| `GET /health` | ✅ `{"status":"ok","llm_healthy":true,"knowledge_base_chunks":16}` |
| `POST /generate-nutrition-plan` | ✅ 200 — Markers: MTHFR C677T + COMT V158M, Model: qwen2.5:1.5b, Supplements: 2, Confidence: 0.94 |

**Running containers:**
| Container | Status | Port |
|-----------|--------|------|
| chromadb | Up | 8001 → 8000 |
| redis | Up | 6379 |
| rag_api | Up | 8000 → 8000 |

---

### Final Status (2026-06-30)

### Phase 6 Remaining Tasks Log

**Task 6.2 — Load Test (locustfile)**
- File: `tests/locustfile.py` — Created earlier, defines a Locust user that hits `/generate-nutrition-plan` with a synthetic PCR PDF.
- Status: ✅ File exists, not yet run against Docker stack (requires `locust` CLI).

**Task 6.3 — Async Task Queue (Celery)**
- Status: ⏸️ Conditional — guide says "only if load tests show queuing pressure needed". Not yet required.

**Task 6.4 — Structured Logging**
- Status: ✅ Already built in. `app/main.py` uses JSON-format logging (module, level, message, time) via `logging` with a custom formatter. Docker logs confirmed JSON output.

**Task 6.5 — OpenAPI Documentation**
- File: `openapi_spec.json` — Already generated (previous session) via `scripts/generate_openapi.py`.
- Status: ✅ Generated and committed.

---

### Final Status (2026-06-30)

**All tasks complete:**
- ✅ 30/30 tests passing (qwen2.5:1.5b model)
- ✅ Docker Compose full stack running (ChromaDB + Redis + API)
- ✅ Health endpoint responding
- ✅ Nutrition plan generation working end-to-end via Docker
- ✅ Portability fix for Tesseract path (Windows + Linux)
- ✅ OpenAPI spec generated
- ✅ Locust load test file ready
- ⏸️ Celery async queue — conditional (not needed yet)

---

### Parser Tuning for CamScanner Research Paper (2026-06-30)

**Problem:** User uploaded a CamScanner PDF of a research paper ("AI and Nutrigenomics in IDD"). The parser had two issues:
1. `VARIANT_PATTERN` contained `[A-Z]\d+[A-Z]` which matched OCR noise "S4S" as a false variant for COMT
2. No distinction between structured test reports (with "Gene:", "Variant:", "Status:" formatting) vs running text in a research paper

**Changes to `app/ocr/parser.py`:**
- `VARIANT_PATTERN` — Removed overly broad `[A-Z]\d+[A-Z]`; replaced with specific known variants only (`rs\d+|C677T|A1298C|V158M|TaqI|A66G|rs429358|rs7412|P199P|Q158R`)
- `STATUS_PATTERN` — Removed `\bvariant\b(?!\s*:)` which matched "variant" in running text; made status terms more specific
- `GENE_PATTERN` — Expanded to include 20+ additional genes from the broader neurodevelopmental/nutrigenomics literature (OPHN1, PAK3, RPS6KA3, IL1RAPL, FMR2, GDI1, CASK, TM4SF2, BDNF, GRIN1/2B, TBR1, CNTNAP2, etc.)
- Added `_is_structured_line()` — guards variant/status extraction: only assigns variant/status when the line contains structured formatting (`Gene:`, `Variant:`, `Status:`, `Result:`, `Genotype:`)
- Genes in running text are still detected (for count/reference) but reported as UNKNOWN variant/UNKNOWN status

**Before fix:** `COMT S4S (UNKNOWN)` — false positive variant from OCR noise
**After fix:** 15 genes detected, all `UNKNOWN` variant/UNKNOWN status — correctly reflects research paper input format

**Test result:** CamScanner PDF pipeline returns 200 with 15 genetic markers (all UNKNOWN), confidence 0.55, generic nutrition plan generated from retrieval context. Full test suite: 30/30 passing.

**Files modified:**
- `app/ocr/parser.py` — Variant/status pattern tightening, context-aware extraction, expanded gene list
