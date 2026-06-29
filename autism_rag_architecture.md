# Autism Nutrition RAG System — Architecture & Execution Plan

---

## Overview

A self-hosted, GPU-accelerated RAG system that receives a genetics test (PDF/scanned image), extracts structured genetic data via OCR, retrieves relevant nutrition guidance from a doctor-approved knowledge base, and returns a structured nutrition plan via a FastAPI endpoint consumed by a mobile app.

---

## Design Principles

- **Provider-agnostic LLM** — a single interface layer abstracts the model backend; swap between local vLLM, Ollama, OpenAI, or Anthropic via one config variable.
- **Privacy-first** — genetic data never leaves the server in normal operation; the local 7B model runs entirely on-premises.
- **Separation of concerns** — OCR, retrieval, generation, and output formatting are independent modules with clean interfaces.
- **Stateless API** — each request is self-contained; no session state stored in the API layer.

---

## System Architecture

```
Mobile App
    │
    │  POST /generate-nutrition-plan
    │  { file: PDF/image, patient_id: str, metadata: {...} }
    ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
│  - Auth middleware (API key / JWT)                       │
│  - Request validation (Pydantic)                        │
│  - Rate limiting                                         │
│  - Async request handling                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator Service                    │
│  Coordinates the full pipeline as a single transaction  │
└──┬──────────────┬──────────────┬────────────────────────┘
   │              │              │
   ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌───────────────────────────┐
│   OCR    │  │Retriever │  │     LLM Interface Layer    │
│  Module  │  │ Module   │  │  (Provider-Agnostic)       │
└──────────┘  └──────────┘  └───────────────────────────┘
   │              │              │
   ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌───────────────────────────┐
│Tesseract │  │  Vector  │  │  Local: vLLM / Ollama      │
│ + OpenCV │  │   DB     │  │  Remote: OpenAI/Anthropic  │
│(preproc) │  │(ChromaDB)│  │  (Mistral-7B default)      │
└──────────┘  └──────────┘  └───────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Knowledge   │
              │    Base      │
              │ (Nutrition   │
              │  Documents)  │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Output      │
              │  Formatter   │
              │  (structured │
              │  JSON plan)  │
              └──────────────┘
```

---

## Module Breakdown

### 1. FastAPI Server
- **Framework:** FastAPI + Uvicorn
- **Endpoint:** `POST /generate-nutrition-plan`
- **Input:** multipart/form-data (PDF or image) + JSON metadata
- **Output:** structured JSON nutrition plan
- **Responsibilities:** auth, validation, rate limiting, async task dispatch

### 2. OCR Module
- **Stack:** OpenCV (preprocessing) → Tesseract or PaddleOCR
- **Steps:**
  1. Deskew, denoise, binarize the scanned image/PDF page
  2. Run OCR to extract raw text
  3. Pass raw text to a lightweight parser to extract structured fields (gene variants, SNP markers, flagged values)
- **Output:** `GeneticProfile` Pydantic model

### 3. Knowledge Base Ingestion Pipeline (offline, run once)
- Load doctor-approved nutrition documents (PDF/DOCX)
- Chunk by semantic section (not fixed token windows)
- Embed with a local embedding model (e.g. `BAAI/bge-small-en-v1.5`)
- Store vectors + metadata in ChromaDB (self-hosted)

### 4. Retriever Module
- Takes the `GeneticProfile` and constructs a retrieval query
- Performs similarity search against ChromaDB
- Returns top-K relevant nutrition document chunks
- Optional: re-rank with a cross-encoder for precision

### 5. LLM Interface Layer (Provider-Agnostic)
```
LLMProvider (abstract base class)
    ├── LocalvLLMProvider     → vLLM REST API (Mistral-7B on GPU)
    ├── OllamaProvider        → Ollama local server
    ├── OpenAIProvider        → OpenAI API
    └── AnthropicProvider     → Anthropic API

Config:
  LLM_BACKEND = "local_vllm"  # change to switch providers
```
- All providers expose one method: `generate(prompt: str, system: str) -> str`
- Prompt template is provider-independent

### 6. Orchestrator Service
Ties everything together in sequence:
```
1. OCR Module           → GeneticProfile
2. Retriever Module     → List[NutritionChunk]
3. Prompt Builder       → Final prompt string
4. LLM Interface Layer  → Raw LLM response
5. Output Formatter     → Structured NutritionPlan JSON
```

### 7. Output Formatter
- Parses LLM response into a strict Pydantic schema
- Handles malformed output with retry logic (up to 2 retries with corrective prompt)
- Final output schema:
```json
{
  "patient_id": "string",
  "generated_at": "ISO timestamp",
  "genetic_markers_detected": ["..."],
  "nutrition_plan": {
    "summary": "string",
    "daily_targets": { "calories": int, "protein_g": int, ... },
    "recommended_foods": ["..."],
    "foods_to_avoid": ["..."],
    "supplements": [{ "name": "...", "dose": "...", "reason": "..." }],
    "notes": "string"
  },
  "source_chunks_used": ["..."],
  "confidence_score": float,
  "requires_doctor_review": boolean
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| OCR | OpenCV + PaddleOCR (or Tesseract) |
| Embedding Model | BAAI/bge-small-en-v1.5 (local) |
| Vector DB | ChromaDB (self-hosted) |
| LLM (default) | Mistral-7B-Instruct via vLLM |
| LLM runtime | vLLM (NVIDIA GPU) |
| Data validation | Pydantic v2 |
| Async task queue | FastAPI BackgroundTasks or Celery + Redis |
| Containerization | Docker + Docker Compose |
| Secrets/Config | python-dotenv / environment variables |

---

## Execution Plan

### Phase 1 — Infrastructure Setup (Days 1–3)
- [ ] **Task 1.1** — Set up Docker Compose with services: FastAPI, ChromaDB, Redis, vLLM
- [ ] **Task 1.2** — Install and verify vLLM with Mistral-7B on the NVIDIA GPU (test inference speed)
- [ ] **Task 1.3** — Implement the `LLMProvider` abstract base class and the `LocalvLLMProvider` concrete implementation
- [ ] **Task 1.4** — Write a config loader that reads `LLM_BACKEND` from `.env` and returns the correct provider instance

### Phase 2 — OCR Pipeline (Days 4–7)
- [ ] **Task 2.1** — Build the image preprocessor (OpenCV: deskew, denoise, threshold)
- [ ] **Task 2.2** — Integrate PaddleOCR; test on sample genetics lab reports
- [ ] **Task 2.3** — Build the genetic data parser: extract SNPs, gene names, flagged markers from raw OCR text using regex + a small extraction prompt
- [ ] **Task 2.4** — Define and validate the `GeneticProfile` Pydantic model
- [ ] **Task 2.5** — Unit test the full OCR → GeneticProfile pipeline on 5+ sample inputs

### Phase 3 — Knowledge Base & Retriever (Days 8–11)
- [ ] **Task 3.1** — Build the document ingestion script: load nutrition docs, semantic chunking, embed with bge-small, upsert into ChromaDB
- [ ] **Task 3.2** — Run ingestion on the doctor-approved nutrition documents
- [ ] **Task 3.3** — Build the Retriever module: query builder from `GeneticProfile` → ChromaDB similarity search → top-K chunks
- [ ] **Task 3.4** — (Optional) Add a cross-encoder re-ranker for precision improvement
- [ ] **Task 3.5** — Test retrieval quality: verify correct chunks are returned for sample genetic profiles

### Phase 4 — LLM Generation & Output Formatting (Days 12–15)
- [ ] **Task 4.1** — Design the system prompt and user prompt template (include genetic data + retrieved chunks)
- [ ] **Task 4.2** — Build the Output Formatter with Pydantic schema + retry logic
- [ ] **Task 4.3** — Build the Orchestrator that chains OCR → Retriever → Prompt → LLM → Formatter
- [ ] **Task 4.4** — Add a `confidence_score` and `requires_doctor_review` flag (rule-based: low retrieval score or ambiguous markers triggers review flag)
- [ ] **Task 4.5** — End-to-end test: PDF in → NutritionPlan JSON out

### Phase 5 — FastAPI Layer & Security (Days 16–18)
- [ ] **Task 5.1** — Build the `/generate-nutrition-plan` endpoint (async, multipart input)
- [ ] **Task 5.2** — Add API key authentication middleware
- [ ] **Task 5.3** — Add request validation, error handling, and structured error responses
- [ ] **Task 5.4** — Add rate limiting (slowapi)
- [ ] **Task 5.5** — Add a `/health` endpoint for the mobile team to ping

### Phase 6 — Testing & Hardening (Days 19–22)
- [ ] **Task 6.1** — Integration tests for the full pipeline
- [ ] **Task 6.2** — Load test the API (simulate concurrent requests from mobile)
- [ ] **Task 6.3** — Implement async request handling / task queue if generation latency is high
- [ ] **Task 6.4** — Add logging (structured JSON logs) and basic monitoring
- [ ] **Task 6.5** — Document the API with FastAPI's auto-generated OpenAPI spec for the mobile team

---

## Key Design Decisions & Rationale

**Why Mistral-7B?**
Strong instruction-following, medical/nutrition domain competence, fits comfortably on a single consumer or server-grade NVIDIA GPU at 4-bit quantization (≈5GB VRAM). Can swap to Llama-3-8B or Phi-3-medium with zero code changes.

**Why ChromaDB?**
Lightweight, fully self-hosted, no cloud dependency, simple Python API. Can migrate to Qdrant or Weaviate later with a retriever swap.

**Why PaddleOCR over Tesseract?**
Better accuracy on low-quality scans and non-standard fonts common in lab reports. Falls back to Tesseract if needed.

**Why the provider-agnostic layer matters here?**
If a specific genetic marker requires a stronger model for a complex case, the system can be configured to route that request to Claude or GPT-4 while keeping routine cases on the local 7B model — all without changing any other code.

**The `requires_doctor_review` flag**
Non-negotiable for medical safety. Any plan generated by the system should be surfaced to the approving doctor before delivery to end users. This flag triggers automatically on low confidence or ambiguous inputs.
