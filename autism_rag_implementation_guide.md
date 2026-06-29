# Autism Nutrition RAG System — Detailed Implementation & Testing Guide

> This document is the step-by-step engineering companion to the architecture plan.
> Every task has exact commands, full code, file paths, and a dedicated test section.
> Follow tasks in order — later tasks depend on earlier ones.

---

## Project Structure (create this before starting)

```
autism-nutrition-rag/
├── docker-compose.yml
├── .env
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py                  # Config loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── genetic_profile.py     # GeneticProfile Pydantic model
│   │   └── nutrition_plan.py      # NutritionPlan Pydantic model
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── preprocessor.py        # OpenCV image preprocessing
│   │   ├── extractor.py           # PaddleOCR runner
│   │   └── parser.py              # Genetic data parser
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── ingestion.py           # Knowledge base ingestion pipeline
│   │   ├── retriever.py           # ChromaDB similarity search
│   │   └── reranker.py            # Optional cross-encoder re-ranker
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                # LLMProvider abstract base class
│   │   ├── local_vllm.py          # LocalvLLMProvider
│   │   ├── ollama.py              # OllamaProvider
│   │   ├── openai_provider.py     # OpenAIProvider
│   │   └── anthropic_provider.py  # AnthropicProvider
│   ├── orchestrator.py            # Pipeline orchestrator
│   ├── formatter.py               # Output formatter + retry logic
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py                # API key middleware
│       └── rate_limit.py          # Rate limiting
├── knowledge_base/
│   └── documents/                 # Drop doctor-approved PDFs here
├── tests/
│   ├── __init__.py
│   ├── test_ocr.py
│   ├── test_retriever.py
│   ├── test_llm.py
│   ├── test_formatter.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── scripts/
│   └── ingest_knowledge_base.py   # Run once to populate ChromaDB
└── sample_data/
    └── sample_genetics_report.pdf # For testing
```

---

## Prerequisites — Install Before Starting

```bash
# Python 3.10+
python --version

# CUDA toolkit (verify GPU is accessible)
nvidia-smi

# Docker & Docker Compose
docker --version
docker compose version

# Create the project
mkdir autism-nutrition-rag && cd autism-nutrition-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Create `requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.2.1
python-multipart==0.0.9
python-dotenv==1.0.1
httpx==0.27.0
paddlepaddle==2.6.1
paddleocr==2.7.3
opencv-python-headless==4.9.0.80
numpy==1.26.4
chromadb==0.5.0
sentence-transformers==3.0.0
PyMuPDF==1.24.3
langchain-text-splitters==0.2.0
slowapi==0.1.9
pytest==8.2.0
pytest-asyncio==0.23.6
httpx==0.27.0
openai==1.30.1
anthropic==0.28.0
```

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Infrastructure Setup (Days 1–3)

---

### Task 1.1 — Docker Compose Setup

**Goal:** Get all services running in containers so the app has ChromaDB, Redis, and vLLM available as networked services.

**File: `docker-compose.yml`**
```yaml
version: "3.9"

services:

  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    networks:
      - rag_network

  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    networks:
      - rag_network

  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    ports:
      - "8002:8000"
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model mistralai/Mistral-7B-Instruct-v0.3
      --quantization awq
      --max-model-len 8192
      --gpu-memory-utilization 0.85
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - rag_network

  api:
    build: .
    container_name: rag_api
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./knowledge_base:/app/knowledge_base
    depends_on:
      - chromadb
      - redis
      - vllm
    networks:
      - rag_network

volumes:
  chroma_data:
  huggingface_cache:

networks:
  rag_network:
    driver: bridge
```

**File: `.env`**
```env
# LLM Backend — change this to switch providers
# Options: local_vllm | ollama | openai | anthropic
LLM_BACKEND=local_vllm

# vLLM (local)
VLLM_BASE_URL=http://vllm:8000/v1
VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

# Ollama (local alternative)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL_NAME=mistral:7b-instruct

# OpenAI (remote fallback)
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini

# Anthropic (remote fallback)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_NAME=claude-3-haiku-20240307

# Hugging Face (for downloading Mistral)
HF_TOKEN=hf_...

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=nutrition_knowledge

# Security
API_SECRET_KEY=your-secret-api-key-change-this
RATE_LIMIT_PER_MINUTE=10

# App
DEBUG=false
LOG_LEVEL=INFO
```

**File: `Dockerfile`**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download PaddleOCR models at build time (avoids runtime delay)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**How to run:**
```bash
docker compose up chromadb redis -d   # Start supporting services first
docker compose up vllm -d             # Start vLLM (takes 2-5 min to load model)
docker compose logs -f vllm           # Watch until you see "Application startup complete"
```

**✅ Test 1.1:**
```bash
# ChromaDB health
curl http://localhost:8001/api/v1/heartbeat
# Expected: {"nanosecond heartbeat": <timestamp>}

# vLLM health
curl http://localhost:8002/health
# Expected: {"status":"ok"}

# vLLM models endpoint
curl http://localhost:8002/v1/models
# Expected: JSON listing Mistral-7B

# Redis
docker exec -it redis redis-cli ping
# Expected: PONG
```

---

### Task 1.2 — Verify vLLM GPU Inference

**Goal:** Confirm Mistral-7B is loaded on the GPU and returns a response at acceptable speed.

```bash
# Check GPU memory usage — should show ~10-14GB used for Mistral-7B AWQ
nvidia-smi

# Send a test completion
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
      {"role": "system", "content": "You are a nutrition expert."},
      {"role": "user", "content": "What foods are high in omega-3?"}
    ],
    "max_tokens": 200,
    "temperature": 0.3
  }'
```

**✅ Test 1.2:**
- Response arrives in under 10 seconds → GPU is being used correctly.
- If response takes 60+ seconds → model fell back to CPU. Check `nvidia-smi` during inference and verify the Docker GPU reservation is working.
- Benchmark: run the curl 3 times and average. Target is under 8 seconds for 200 tokens.

```bash
# Quick benchmark script
time curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3","messages":[{"role":"user","content":"List 5 high-protein foods."}],"max_tokens":100}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
```

---

### Task 1.3 — LLM Provider Abstract Base Class + LocalvLLMProvider

**Goal:** Implement the provider-agnostic interface so all future LLM calls go through one contract.

**File: `app/llm/base.py`**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement generate().
    The rest of the system only interacts with this interface.
    """

    @abstractmethod
    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        """
        Send a prompt to the LLM and return the response.

        Args:
            user_prompt: The main content prompt (genetic data + retrieved chunks).
            system_prompt: The role/behavior instruction for the model.
            temperature: Sampling temperature. Lower = more deterministic.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and token usage.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and ready."""
        pass
```

**File: `app/llm/local_vllm.py`**
```python
import httpx
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class LocalvLLMProvider(LLMProvider):
    """
    Calls a locally running vLLM server via its OpenAI-compatible REST API.
    vLLM exposes /v1/chat/completions — same schema as OpenAI.
    """

    def __init__(self):
        self.base_url = settings.VLLM_BASE_URL
        self.model_name = settings.VLLM_MODEL_NAME
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"Sending request to vLLM: {self.base_url}/chat/completions")

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
        )

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url.replace('/v1', '')}/health"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False
```

**File: `app/llm/ollama.py`**
```python
import httpx
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Calls a locally running Ollama server."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL_NAME
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=data["model"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
```

**File: `app/llm/openai_provider.py`**
```python
from openai import AsyncOpenAI
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.OPENAI_MODEL_NAME

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
```

**File: `app/llm/anthropic_provider.py`**
```python
import anthropic
from app.llm.base import LLMProvider, LLMResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model_name = settings.ANTHROPIC_MODEL_NAME

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def health_check(self) -> bool:
        try:
            # Send a minimal message to verify connectivity
            await self.client.messages.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
```

---

### Task 1.4 — Config Loader

**File: `app/config.py`**
```python
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # LLM Backend selection
    LLM_BACKEND: Literal["local_vllm", "ollama", "openai", "anthropic"] = "local_vllm"

    # vLLM
    VLLM_BASE_URL: str = "http://vllm:8000/v1"
    VLLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "mistral:7b-instruct"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL_NAME: str = "claude-3-haiku-20240307"

    # ChromaDB
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "nutrition_knowledge"

    # Security
    API_SECRET_KEY: str = "change-this-in-production"
    RATE_LIMIT_PER_MINUTE: int = 10

    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def get_llm_provider():
    """
    Factory function — reads LLM_BACKEND from config and returns
    the correct provider instance. Add new providers here.
    """
    from app.llm.local_vllm import LocalvLLMProvider
    from app.llm.ollama import OllamaProvider
    from app.llm.openai_provider import OpenAIProvider
    from app.llm.anthropic_provider import AnthropicProvider

    providers = {
        "local_vllm": LocalvLLMProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(settings.LLM_BACKEND)
    if not provider_class:
        raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND}")

    return provider_class()
```

**✅ Test 1.3 & 1.4:**
```bash
# Create tests/test_llm.py
cat > tests/test_llm.py << 'EOF'
import pytest
import asyncio
import os
os.environ["LLM_BACKEND"] = "local_vllm"  # Override for test

from app.config import get_llm_provider
from app.llm.base import LLMResponse


@pytest.mark.asyncio
async def test_provider_factory_returns_correct_type():
    """Config loader returns the right class for each backend."""
    os.environ["LLM_BACKEND"] = "local_vllm"
    from app.config import get_settings
    get_settings.cache_clear()
    provider = get_llm_provider()
    from app.llm.local_vllm import LocalvLLMProvider
    assert isinstance(provider, LocalvLLMProvider)


@pytest.mark.asyncio
async def test_vllm_health_check():
    """vLLM server is reachable."""
    from app.llm.local_vllm import LocalvLLMProvider
    provider = LocalvLLMProvider()
    result = await provider.health_check()
    assert result is True, "vLLM health check failed — is the container running?"


@pytest.mark.asyncio
async def test_vllm_generate_returns_llmresponse():
    """vLLM returns a valid LLMResponse with content."""
    from app.llm.local_vllm import LocalvLLMProvider
    provider = LocalvLLMProvider()
    response = await provider.generate(
        user_prompt="What is vitamin D?",
        system_prompt="You are a nutrition expert. Be concise.",
        max_tokens=100,
    )
    assert isinstance(response, LLMResponse)
    assert len(response.content) > 10
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0


@pytest.mark.asyncio
async def test_provider_interface_is_consistent():
    """All providers expose the same generate() signature."""
    from app.llm.local_vllm import LocalvLLMProvider
    from app.llm.ollama import OllamaProvider
    from app.llm.openai_provider import OpenAIProvider
    from app.llm.anthropic_provider import AnthropicProvider

    for ProviderClass in [LocalvLLMProvider, OllamaProvider, OpenAIProvider, AnthropicProvider]:
        provider = ProviderClass()
        assert hasattr(provider, "generate")
        assert hasattr(provider, "health_check")
        assert callable(provider.generate)
        assert callable(provider.health_check)
EOF

pytest tests/test_llm.py -v
```

---

## Phase 2 — OCR Pipeline (Days 4–7)

---

### Task 2.1 — Image Preprocessor

**Goal:** Take a raw scanned image or PDF page and clean it up so OCR accuracy is maximized. Scanned lab reports often have noise, skew, and low contrast.

**File: `app/ocr/preprocessor.py`**
```python
import cv2
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
import logging
from typing import List

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Converts a PDF or image file into clean, OCR-ready images.
    Pipeline: PDF → pages → grayscale → denoise → deskew → binarize.
    """

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert each page of a PDF to a numpy image array at the given DPI.
        300 DPI is the minimum recommended for good OCR quality.
        """
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Scale matrix: 300 DPI / 72 DPI (PDF default) = 4.17x
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(img)
        doc.close()
        logger.info(f"Extracted {len(images)} pages from PDF")
        return images

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image file (JPG, PNG, TIFF) as a numpy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a single image.
        Returns a clean binary image ready for OCR.
        """
        img = self._to_grayscale(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._binarize(img)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Gaussian blur to remove scanner noise without destroying text edges.
        Kernel size (3,3) is gentle — preserves fine text strokes.
        """
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct page rotation using Hough Line Transform.
        Most lab reports have slight rotation from scanner misalignment.
        """
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return img

        angles = []
        for line in lines[:20]:  # Use top 20 lines only
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 10:  # Only correct small skews (<10 degrees)
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:  # Skip trivial corrections
            return img

        logger.debug(f"Deskewing by {median_angle:.2f} degrees")
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding — handles uneven lighting across the page.
        Better than global Otsu for documents with shadows or gradients.
        """
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

    def preprocess_file(self, file_path: str) -> List[np.ndarray]:
        """
        Entry point: accepts a PDF or image path, returns list of
        preprocessed page images ready for OCR.
        """
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            raw_images = self.pdf_to_images(file_path)
        elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            raw_images = [self.load_image(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        preprocessed = [self.preprocess(img) for img in raw_images]
        logger.info(f"Preprocessed {len(preprocessed)} page(s) from {path.name}")
        return preprocessed
```

**✅ Test 2.1:**
```python
# tests/test_ocr.py (partial — add to this file as tasks progress)
import pytest
import numpy as np
from app.ocr.preprocessor import ImagePreprocessor


def test_preprocessor_loads_pdf():
    """PDF pages are extracted as numpy arrays."""
    preprocessor = ImagePreprocessor()
    # Create a minimal test PDF with fitz
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    doc.save("sample_data/test_report.pdf")
    doc.close()

    images = preprocessor.preprocess_file("sample_data/test_report.pdf")
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)
    assert images[0].dtype == np.uint8


def test_preprocessor_output_is_binary():
    """Preprocessed image contains only 0 and 255 values (binary)."""
    preprocessor = ImagePreprocessor()
    # Create a synthetic grayscale image
    img = np.random.randint(100, 200, (500, 500, 3), dtype=np.uint8)
    result = preprocessor.preprocess(img)
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255}), "Image is not binary"


def test_preprocessor_handles_already_grayscale():
    """Preprocessor handles grayscale (2D) images without crashing."""
    preprocessor = ImagePreprocessor()
    img = np.ones((300, 300), dtype=np.uint8) * 180
    result = preprocessor.preprocess(img)
    assert result.shape == (300, 300)
```

```bash
mkdir -p sample_data
pytest tests/test_ocr.py::test_preprocessor_loads_pdf -v
pytest tests/test_ocr.py::test_preprocessor_output_is_binary -v
```

---

### Task 2.2 — PaddleOCR Extractor

**File: `app/ocr/extractor.py`**
```python
from paddleocr import PaddleOCR
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Runs PaddleOCR on preprocessed images and returns extracted text.
    Uses angle classification to handle rotated text blocks in lab reports.
    Initialized once and reused — PaddleOCR model loading is expensive.
    """

    def __init__(self):
        logger.info("Loading PaddleOCR models...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,   # Detect 90/180/270 degree rotations
            lang="en",
            use_gpu=False,        # PaddleOCR CPU is fine; heavy lifting is in preprocessing
            show_log=False,
        )
        logger.info("PaddleOCR models loaded.")

    def extract_text(self, image: np.ndarray) -> str:
        """
        Run OCR on a single preprocessed image.
        Returns concatenated text with newlines between detected lines.
        """
        result = self.ocr.ocr(image, cls=True)

        if not result or not result[0]:
            logger.warning("OCR returned no results for this page.")
            return ""

        lines = []
        for line in result[0]:
            # line format: [[bounding_box], (text, confidence)]
            text = line[1][0]
            confidence = line[1][1]

            if confidence >= 0.6:  # Filter low-confidence detections
                lines.append(text)
            else:
                logger.debug(f"Skipping low-confidence OCR result: '{text}' ({confidence:.2f})")

        extracted = "\n".join(lines)
        logger.info(f"Extracted {len(lines)} text lines from page")
        return extracted

    def extract_from_pages(self, pages: List[np.ndarray]) -> str:
        """
        Extract text from multiple pages and concatenate.
        Pages are separated by a clear delimiter for downstream parsing.
        """
        all_text = []
        for i, page in enumerate(pages):
            page_text = self.extract_text(page)
            all_text.append(f"--- PAGE {i + 1} ---\n{page_text}")

        return "\n\n".join(all_text)
```

**✅ Test 2.2:**
```python
# Add to tests/test_ocr.py
from app.ocr.extractor import OCRExtractor
import numpy as np


def test_ocr_extracts_text_from_synthetic_image():
    """OCR returns non-empty text from an image containing text."""
    import fitz
    import cv2

    # Create a test PDF with known genetics content
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "GENETIC REPORT", fontsize=16)
    page.insert_text((50, 140), "Patient: Test Patient")
    page.insert_text((50, 170), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    page.insert_text((50, 200), "Gene: COMT   Variant: V158M  Status: Homozygous")
    pdf_path = "sample_data/test_ocr_input.pdf"
    doc.save(pdf_path)
    doc.close()

    from app.ocr.preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    pages = preprocessor.preprocess_file(pdf_path)

    extractor = OCRExtractor()
    text = extractor.extract_from_pages(pages)

    assert "MTHFR" in text or "mthfr" in text.lower(), \
        f"Expected MTHFR in OCR output, got: {text[:500]}"
    assert len(text) > 20


def test_ocr_returns_empty_string_for_blank_image():
    """OCR gracefully handles blank pages."""
    extractor = OCRExtractor()
    blank = np.ones((500, 500), dtype=np.uint8) * 255  # All white
    result = extractor.extract_text(blank)
    assert isinstance(result, str)
```

```bash
pytest tests/test_ocr.py -v -k "ocr"
```

---

### Task 2.3 — Genetic Data Parser

**Goal:** Take raw OCR text and extract structured genetic data (gene names, variants, zygosity status). Uses regex first, LLM extraction as fallback.

**File: `app/ocr/parser.py`**
```python
import re
import json
import logging
from typing import List, Optional
from app.models.genetic_profile import GeneticMarker, GeneticProfile

logger = logging.getLogger(__name__)

# Common SNP patterns seen in genetics lab reports
GENE_PATTERN = re.compile(
    r"(?i)(MTHFR|COMT|VDR|APOE|FTO|TCF7L2|BCMO1|FADS1|FADS2|SOD2|CBS|MTR|MTRR|BHMT)"
)

VARIANT_PATTERN = re.compile(
    r"(?i)(rs\d+|[A-Z]\d+[A-Z]|C677T|A1298C|V158M|rs429358|rs7412)"
)

STATUS_PATTERN = re.compile(
    r"(?i)(homozygous\s+(?:variant|risk|alternate)|heterozygous|wild[\s-]?type|"
    r"\+/\+|\+/-|-/-|variant|normal|positive|negative|detected|not\s+detected)"
)

FLAG_PATTERN = re.compile(r"(?i)(HIGH RISK|MODERATE RISK|LOW RISK|FLAG|ABNORMAL|ATTENTION)")


class GeneticDataParser:
    """
    Parses raw OCR text into a structured GeneticProfile.
    Strategy: regex extraction first, then LLM-assisted extraction
    for any markers not captured by regex.
    """

    def parse(self, raw_text: str, patient_id: str = "unknown") -> GeneticProfile:
        """
        Main entry point. Returns a GeneticProfile from raw OCR text.
        """
        markers = self._regex_extract(raw_text)
        flagged = self._extract_flags(raw_text)
        raw_text_summary = self._clean_text(raw_text)

        if len(markers) == 0:
            logger.warning(
                "Regex found no markers — text may be poorly structured. "
                "Consider LLM-assisted extraction."
            )

        profile = GeneticProfile(
            patient_id=patient_id,
            markers=markers,
            flagged_values=flagged,
            raw_text_summary=raw_text_summary[:2000],  # Cap to avoid bloating prompts
            extraction_confidence=self._score_confidence(markers, raw_text),
        )

        logger.info(
            f"Parsed {len(markers)} genetic markers for patient {patient_id}"
        )
        return profile

    def _regex_extract(self, text: str) -> List[GeneticMarker]:
        """
        Sliding window extraction: find gene names, then look nearby for
        variant and status information.
        """
        lines = text.split("\n")
        markers = []
        seen_genes = set()

        for i, line in enumerate(lines):
            gene_match = GENE_PATTERN.search(line)
            if not gene_match:
                continue

            gene = gene_match.group(0).upper()
            if gene in seen_genes:
                continue
            seen_genes.add(gene)

            # Look at current line and the next 2 lines for variant + status
            context = " ".join(lines[i : i + 3])
            variant_match = VARIANT_PATTERN.search(context)
            status_match = STATUS_PATTERN.search(context)

            markers.append(
                GeneticMarker(
                    gene=gene,
                    variant=variant_match.group(0).upper() if variant_match else "UNKNOWN",
                    status=self._normalize_status(
                        status_match.group(0) if status_match else "UNKNOWN"
                    ),
                    raw_line=line.strip(),
                )
            )

        return markers

    def _normalize_status(self, raw_status: str) -> str:
        """Normalize varied status strings to canonical forms."""
        s = raw_status.lower().strip()
        if any(x in s for x in ["+/+", "homozygous variant", "homozygous risk"]):
            return "HOMOZYGOUS_VARIANT"
        if any(x in s for x in ["+/-", "heterozygous"]):
            return "HETEROZYGOUS"
        if any(x in s for x in ["-/-", "wild type", "wild-type", "normal"]):
            return "WILD_TYPE"
        return raw_status.upper()

    def _extract_flags(self, text: str) -> List[str]:
        return list(set(FLAG_PATTERN.findall(text)))

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and page markers for storage."""
        text = re.sub(r"--- PAGE \d+ ---", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _score_confidence(self, markers: List[GeneticMarker], text: str) -> float:
        """
        Simple heuristic confidence score for the extraction.
        Lower score = more likely to need doctor review.
        """
        if len(markers) == 0:
            return 0.1
        known_markers = sum(1 for m in markers if m.status != "UNKNOWN")
        known_variants = sum(1 for m in markers if m.variant != "UNKNOWN")
        score = (known_markers + known_variants) / (len(markers) * 2)
        return round(min(score, 1.0), 2)
```

---

### Task 2.4 — GeneticProfile Pydantic Model

**File: `app/models/genetic_profile.py`**
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class GeneticMarker(BaseModel):
    gene: str = Field(..., description="Gene name, e.g. MTHFR, COMT")
    variant: str = Field(..., description="Variant identifier, e.g. C677T, rs429358")
    status: str = Field(
        ...,
        description="Zygosity: HOMOZYGOUS_VARIANT | HETEROZYGOUS | WILD_TYPE | UNKNOWN",
    )
    raw_line: str = Field(default="", description="Original OCR line for audit trail")


class GeneticProfile(BaseModel):
    patient_id: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    markers: List[GeneticMarker] = Field(default_factory=list)
    flagged_values: List[str] = Field(default_factory=list)
    raw_text_summary: str = Field(default="")
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_retrieval_query(self) -> str:
        """
        Convert the genetic profile into a natural language query
        for ChromaDB similarity search.
        """
        if not self.markers:
            return "general autism nutrition recommendations dietary guidelines"

        parts = []
        for m in self.markers:
            if m.status in ["HOMOZYGOUS_VARIANT", "HETEROZYGOUS"]:
                parts.append(
                    f"{m.gene} {m.variant} {m.status.lower().replace('_', ' ')} "
                    f"nutrition diet supplementation"
                )

        if not parts:
            parts.append("autism nutrition recommendations diet")

        return " | ".join(parts[:5])  # Limit query length

    def has_actionable_markers(self) -> bool:
        return any(
            m.status in ["HOMOZYGOUS_VARIANT", "HETEROZYGOUS"] for m in self.markers
        )
```

**File: `app/models/nutrition_plan.py`**
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DailyTargets(BaseModel):
    calories: Optional[int] = None
    protein_g: Optional[int] = None
    carbohydrates_g: Optional[int] = None
    fat_g: Optional[int] = None
    fiber_g: Optional[int] = None
    water_ml: Optional[int] = None


class Supplement(BaseModel):
    name: str
    dose: str
    frequency: str
    reason: str


class NutritionPlanContent(BaseModel):
    summary: str
    daily_targets: DailyTargets = Field(default_factory=DailyTargets)
    recommended_foods: List[str] = Field(default_factory=list)
    foods_to_avoid: List[str] = Field(default_factory=list)
    supplements: List[Supplement] = Field(default_factory=list)
    meal_timing_notes: str = ""
    notes: str = ""


class NutritionPlan(BaseModel):
    patient_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    genetic_markers_detected: List[str] = Field(default_factory=list)
    nutrition_plan: NutritionPlanContent
    source_chunks_used: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_doctor_review: bool = True  # Default TRUE — always safe
    llm_model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

**✅ Test 2.4 & 2.5:**
```python
# Add to tests/test_ocr.py
from app.ocr.parser import GeneticDataParser
from app.models.genetic_profile import GeneticProfile, GeneticMarker


def test_parser_extracts_mthfr_from_text():
    sample_text = """
    --- PAGE 1 ---
    GENETIC ANALYSIS REPORT
    Gene: MTHFR  Variant: C677T  Status: Heterozygous
    Gene: COMT   Variant: V158M  Status: Homozygous Variant
    Gene: VDR    Variant: rs2228570  Status: Wild Type
    HIGH RISK: Folate metabolism impairment detected
    """
    parser = GeneticDataParser()
    profile = parser.parse(sample_text, patient_id="test-001")

    assert isinstance(profile, GeneticProfile)
    assert len(profile.markers) >= 2

    gene_names = [m.gene for m in profile.markers]
    assert "MTHFR" in gene_names
    assert "COMT" in gene_names


def test_parser_normalizes_status():
    sample_text = "Gene: MTHFR  Variant: C677T  Status: +/+"
    parser = GeneticDataParser()
    profile = parser.parse(sample_text)
    assert profile.markers[0].status == "HOMOZYGOUS_VARIANT"


def test_parser_returns_profile_with_no_markers_gracefully():
    """Parser handles empty or garbage text without crashing."""
    parser = GeneticDataParser()
    profile = parser.parse("This is not a genetics report.", patient_id="bad-input")
    assert len(profile.markers) == 0
    assert profile.extraction_confidence < 0.5


def test_genetic_profile_to_retrieval_query():
    """to_retrieval_query generates a non-empty search string."""
    profile = GeneticProfile(
        patient_id="test",
        markers=[
            GeneticMarker(gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line="")
        ],
    )
    query = profile.to_retrieval_query()
    assert "MTHFR" in query
    assert len(query) > 5


def test_full_ocr_to_profile_pipeline():
    """End-to-end: PDF → preprocessed image → OCR text → GeneticProfile."""
    import fitz
    from app.ocr.preprocessor import ImagePreprocessor
    from app.ocr.extractor import OCRExtractor

    # Create synthetic test PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "MTHFR C677T Heterozygous")
    page.insert_text((50, 140), "COMT V158M Homozygous Variant")
    doc.save("sample_data/pipeline_test.pdf")
    doc.close()

    preprocessor = ImagePreprocessor()
    extractor = OCRExtractor()
    parser = GeneticDataParser()

    pages = preprocessor.preprocess_file("sample_data/pipeline_test.pdf")
    text = extractor.extract_from_pages(pages)
    profile = parser.parse(text, patient_id="pipeline-test")

    assert isinstance(profile, GeneticProfile)
    # OCR on synthetic PDFs is highly accurate
    assert len(profile.markers) >= 1
```

```bash
pytest tests/test_ocr.py -v
# All 8 tests should pass
```

---

## Phase 3 — Knowledge Base & Retriever (Days 8–11)

---

### Task 3.1 — Document Ingestion Pipeline

**File: `scripts/ingest_knowledge_base.py`**
```python
#!/usr/bin/env python3
"""
Run this script ONCE (or whenever knowledge base documents change).
It loads nutrition PDFs from knowledge_base/documents/, chunks them,
embeds them, and stores them in ChromaDB.

Usage:
    python scripts/ingest_knowledge_base.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.config import settings

DOCUMENTS_DIR = Path("knowledge_base/documents")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split text into semantic chunks using recursive character splitting.
    Chunk size 512 tokens with 50 token overlap for context continuity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "source": source_name,
            "chunk_index": i,
            "id": hashlib.md5(f"{source_name}:{i}:{chunk[:50]}".encode()).hexdigest(),
        }
        for i, chunk in enumerate(chunks)
        if len(chunk.strip()) > 50  # Skip trivially short chunks
    ]


def main():
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Using collection: {settings.CHROMA_COLLECTION_NAME}")

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {DOCUMENTS_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) to ingest")

    all_chunks = []
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, source_name=pdf_path.name)
        all_chunks.extend(chunks)
        logger.info(f"  → {len(chunks)} chunks created")

    logger.info(f"Total chunks to embed: {len(all_chunks)}")

    # Embed in batches of 64
    BATCH_SIZE = 64
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch]

        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    final_count = collection.count()
    logger.info(f"✅ Ingestion complete. Collection now has {final_count} chunks.")


if __name__ == "__main__":
    main()
```

**How to run:**
```bash
# Put your nutrition PDFs in knowledge_base/documents/
mkdir -p knowledge_base/documents
# Copy your doctor-approved documents there, then:
python scripts/ingest_knowledge_base.py
```

---

### Task 3.2 — Retriever Module

**File: `app/retriever/retriever.py`**
```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List
from dataclasses import dataclass
import logging

from app.config import settings
from app.models.genetic_profile import GeneticProfile

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class NutritionChunk:
    text: str
    source: str
    chunk_index: int
    similarity_score: float


class NutritionRetriever:
    """
    Retrieves relevant nutrition knowledge base chunks
    given a GeneticProfile as the query context.
    Uses cosine similarity search in ChromaDB.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("NutritionRetriever initialized.")

    def retrieve(self, profile: GeneticProfile, top_k: int = 6) -> List[NutritionChunk]:
        """
        Build a retrieval query from the genetic profile,
        embed it, and return the top-K most relevant chunks.
        """
        query = profile.to_retrieval_query()
        logger.info(f"Retrieval query: {query[:200]}")

        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score: 1 - (distance / 2)
                similarity = round(1 - (dist / 2), 4)
                chunks.append(
                    NutritionChunk(
                        text=doc,
                        source=meta.get("source", "unknown"),
                        chunk_index=meta.get("chunk_index", 0),
                        similarity_score=similarity,
                    )
                )

        logger.info(
            f"Retrieved {len(chunks)} chunks. "
            f"Top score: {chunks[0].similarity_score if chunks else 'N/A'}"
        )
        return chunks

    def get_collection_size(self) -> int:
        return self.collection.count()
```

---

### Task 3.3 — Optional Re-ranker

**File: `app/retriever/reranker.py`**
```python
from sentence_transformers import CrossEncoder
from typing import List
from app.retriever.retriever import NutritionChunk
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.
    Cross-encoders are slower but more accurate than bi-encoders
    because they see both the query and the document together.
    Only run on top-K candidates from the retriever (not the whole DB).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[NutritionChunk], top_k: int = 4
    ) -> List[NutritionChunk]:
        if not chunks:
            return chunks

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.similarity_score = float(score)

        reranked = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
        logger.info(
            f"Re-ranked {len(chunks)} chunks → keeping top {top_k}. "
            f"Top score: {reranked[0].similarity_score:.4f}"
        )
        return reranked[:top_k]
```

**✅ Test 3.1 – 3.3:**
```python
# tests/test_retriever.py
import pytest
from app.retriever.retriever import NutritionRetriever, NutritionChunk
from app.models.genetic_profile import GeneticProfile, GeneticMarker


def test_retriever_collection_is_not_empty():
    """ChromaDB collection has been populated (ingestion ran)."""
    retriever = NutritionRetriever()
    count = retriever.get_collection_size()
    assert count > 0, (
        f"Collection is empty. Run: python scripts/ingest_knowledge_base.py"
    )


def test_retriever_returns_chunks_for_mthfr_profile():
    """Retriever returns relevant chunks for an MTHFR heterozygous profile."""
    retriever = NutritionRetriever()
    profile = GeneticProfile(
        patient_id="test-001",
        markers=[
            GeneticMarker(gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line="")
        ],
    )
    chunks = retriever.retrieve(profile, top_k=4)

    assert len(chunks) > 0
    assert all(isinstance(c, NutritionChunk) for c in chunks)
    assert all(0.0 <= c.similarity_score <= 1.0 for c in chunks)
    # At least one chunk should mention folate/methylation (relevant to MTHFR)
    combined_text = " ".join(c.text.lower() for c in chunks)
    assert any(
        keyword in combined_text
        for keyword in ["folate", "methyl", "b12", "nutrition", "vitamin"]
    ), "Retrieved chunks seem unrelated to MTHFR nutrition"


def test_retriever_returns_fewer_chunks_than_requested_if_db_is_small():
    """Retriever does not crash when top_k exceeds collection size."""
    retriever = NutritionRetriever()
    profile = GeneticProfile(patient_id="test", markers=[])
    chunks = retriever.retrieve(profile, top_k=1000)
    assert len(chunks) <= retriever.get_collection_size()


def test_retrieval_quality_scores_are_ordered():
    """Chunks are returned in descending similarity order."""
    retriever = NutritionRetriever()
    profile = GeneticProfile(
        patient_id="test",
        markers=[GeneticMarker(gene="COMT", variant="V158M", status="HOMOZYGOUS_VARIANT", raw_line="")]
    )
    chunks = retriever.retrieve(profile, top_k=5)
    scores = [c.similarity_score for c in chunks]
    assert scores == sorted(scores, reverse=True), "Chunks not in descending score order"
```

```bash
pytest tests/test_retriever.py -v
```

---

## Phase 4 — LLM Generation & Output Formatting (Days 12–15)

---

### Task 4.1 — Prompt Templates

**File: `app/prompts.py`**
```python
SYSTEM_PROMPT = """You are an expert clinical nutritionist specializing in pediatric autism spectrum disorder (ASD).
You generate evidence-based, personalized nutrition plans based on a child's genetic profile.

Your recommendations must:
1. Be grounded ONLY in the provided knowledge base excerpts.
2. Be specific and actionable (exact foods, portion guidance, supplement doses).
3. Acknowledge genetic variants that directly affect nutrient metabolism.
4. Flag any recommendations that require physician confirmation.
5. Never invent information not present in the knowledge base excerpts.

You respond ONLY with a valid JSON object. No preamble. No explanation outside the JSON.
"""

USER_PROMPT_TEMPLATE = """
## Patient Genetic Profile

Patient ID: {patient_id}

Detected Genetic Markers:
{markers_text}

Flagged Values: {flagged_values}

OCR Text Summary (for reference):
{raw_text_summary}

---

## Relevant Knowledge Base Excerpts

{knowledge_chunks}

---

## Instructions

Based ONLY on the genetic profile and the knowledge base excerpts above,
generate a comprehensive nutrition plan in the following exact JSON format.
Do not include any text outside the JSON object.

{{
  "summary": "2-3 sentence overview of the nutrition approach for this genetic profile",
  "daily_targets": {{
    "calories": <integer or null>,
    "protein_g": <integer or null>,
    "carbohydrates_g": <integer or null>,
    "fat_g": <integer or null>,
    "fiber_g": <integer or null>,
    "water_ml": <integer or null>
  }},
  "recommended_foods": ["food 1", "food 2", ...],
  "foods_to_avoid": ["food 1", "food 2", ...],
  "supplements": [
    {{
      "name": "supplement name",
      "dose": "e.g. 400mcg",
      "frequency": "e.g. once daily with food",
      "reason": "why this is recommended based on genetics"
    }}
  ],
  "meal_timing_notes": "any relevant meal timing or preparation guidance",
  "notes": "any important caveats, contraindications, or doctor review items"
}}
"""


def build_user_prompt(
    patient_id: str,
    markers,
    flagged_values: list,
    raw_text_summary: str,
    chunks,
) -> str:
    markers_text = "\n".join(
        f"- {m.gene} | {m.variant} | {m.status}" for m in markers
    ) or "No markers successfully extracted."

    knowledge_chunks = "\n\n---\n\n".join(
        f"[Source: {c.source}, Score: {c.similarity_score:.2f}]\n{c.text}"
        for c in chunks
    )

    return USER_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        markers_text=markers_text,
        flagged_values=", ".join(flagged_values) if flagged_values else "None",
        raw_text_summary=raw_text_summary[:500],
        knowledge_chunks=knowledge_chunks,
    )
```

---

### Task 4.2 — Output Formatter with Retry Logic

**File: `app/formatter.py`**
```python
import json
import re
import logging
from typing import Optional
from app.models.nutrition_plan import NutritionPlan, NutritionPlanContent, DailyTargets, Supplement
from app.models.genetic_profile import GeneticProfile
from app.retriever.retriever import NutritionChunk
from app.llm.base import LLMResponse

logger = logging.getLogger(__name__)

CORRECTION_PROMPT = """Your previous response could not be parsed as valid JSON.
Please respond ONLY with a valid JSON object in the exact format specified.
No markdown, no backticks, no explanation — just the raw JSON object starting with {{.

Previous response (broken):
{previous_response}

Try again:"""


class OutputFormatter:
    """
    Parses LLM text output into a structured NutritionPlan.
    Retries with a corrective prompt if JSON parsing fails.
    """

    def __init__(self, llm_provider, system_prompt: str):
        self.llm = llm_provider
        self.system_prompt = system_prompt

    async def parse_with_retry(
        self,
        llm_response: LLMResponse,
        profile: GeneticProfile,
        chunks: list[NutritionChunk],
        max_retries: int = 2,
    ) -> NutritionPlan:
        raw_content = llm_response.content
        attempt = 0

        while attempt <= max_retries:
            parsed = self._try_parse_json(raw_content)
            if parsed is not None:
                return self._build_nutrition_plan(
                    parsed, profile, chunks, llm_response
                )

            attempt += 1
            if attempt > max_retries:
                break

            logger.warning(f"JSON parse failed (attempt {attempt}). Retrying with correction prompt.")
            correction = CORRECTION_PROMPT.format(previous_response=raw_content[:500])
            retry_response = await self.llm.generate(
                user_prompt=correction,
                system_prompt=self.system_prompt,
                temperature=0.1,  # Very low temp for correction
            )
            raw_content = retry_response.content

        # All retries exhausted — return a safe fallback plan
        logger.error("All retry attempts failed. Returning fallback nutrition plan.")
        return self._fallback_plan(profile, chunks, llm_response)

    def _try_parse_json(self, text: str) -> Optional[dict]:
        """Try multiple JSON extraction strategies."""
        # Strategy 1: direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _build_nutrition_plan(
        self,
        data: dict,
        profile: GeneticProfile,
        chunks: list[NutritionChunk],
        llm_response: LLMResponse,
    ) -> NutritionPlan:
        daily_targets_data = data.get("daily_targets", {}) or {}
        supplements_data = data.get("supplements", []) or []

        avg_retrieval_score = (
            sum(c.similarity_score for c in chunks) / len(chunks) if chunks else 0.0
        )
        confidence = round(
            (profile.extraction_confidence * 0.4) + (avg_retrieval_score * 0.6), 2
        )

        return NutritionPlan(
            patient_id=profile.patient_id,
            genetic_markers_detected=[
                f"{m.gene} {m.variant} ({m.status})" for m in profile.markers
            ],
            nutrition_plan=NutritionPlanContent(
                summary=data.get("summary", ""),
                daily_targets=DailyTargets(**{
                    k: v for k, v in daily_targets_data.items()
                    if k in DailyTargets.model_fields and v is not None
                }),
                recommended_foods=data.get("recommended_foods", []),
                foods_to_avoid=data.get("foods_to_avoid", []),
                supplements=[
                    Supplement(
                        name=s.get("name", ""),
                        dose=s.get("dose", ""),
                        frequency=s.get("frequency", ""),
                        reason=s.get("reason", ""),
                    )
                    for s in supplements_data
                    if isinstance(s, dict)
                ],
                meal_timing_notes=data.get("meal_timing_notes", ""),
                notes=data.get("notes", ""),
            ),
            source_chunks_used=[f"{c.source}[{c.chunk_index}]" for c in chunks],
            confidence_score=confidence,
            requires_doctor_review=(
                confidence < 0.6
                or not profile.has_actionable_markers()
                or len(chunks) == 0
            ),
            llm_model_used=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )

    def _fallback_plan(
        self, profile: GeneticProfile, chunks: list, llm_response: LLMResponse
    ) -> NutritionPlan:
        return NutritionPlan(
            patient_id=profile.patient_id,
            genetic_markers_detected=[],
            nutrition_plan=NutritionPlanContent(
                summary="Nutrition plan could not be generated automatically. Manual review required.",
                notes="LLM output parsing failed after maximum retries. Please review manually.",
            ),
            source_chunks_used=[],
            confidence_score=0.0,
            requires_doctor_review=True,
            llm_model_used=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )
```

---

### Task 4.3 — Orchestrator

**File: `app/orchestrator.py`**
```python
import logging
import tempfile
import os
from app.ocr.preprocessor import ImagePreprocessor
from app.ocr.extractor import OCRExtractor
from app.ocr.parser import GeneticDataParser
from app.retriever.retriever import NutritionRetriever
from app.formatter import OutputFormatter
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models.nutrition_plan import NutritionPlan
from app.config import get_llm_provider

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Coordinates the full RAG pipeline:
    File → OCR → GeneticProfile → Retrieval → LLM → NutritionPlan
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.extractor = OCRExtractor()
        self.parser = GeneticDataParser()
        self.retriever = NutritionRetriever()
        self.llm = get_llm_provider()
        self.formatter = OutputFormatter(self.llm, SYSTEM_PROMPT)
        logger.info("RAGOrchestrator initialized.")

    async def process(self, file_bytes: bytes, filename: str, patient_id: str) -> NutritionPlan:
        """
        Full pipeline. Accepts raw file bytes, returns a NutritionPlan.
        """
        logger.info(f"Processing file '{filename}' for patient '{patient_id}'")

        # Step 1: Save bytes to a temp file (preprocessor needs a file path)
        suffix = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Step 2: OCR
            logger.info("Step 1/5: Preprocessing image(s)")
            pages = self.preprocessor.preprocess_file(tmp_path)

            logger.info("Step 2/5: Extracting text via OCR")
            raw_text = self.extractor.extract_from_pages(pages)

            logger.info("Step 3/5: Parsing genetic data")
            profile = self.parser.parse(raw_text, patient_id=patient_id)

            # Step 3: Retrieve relevant chunks
            logger.info("Step 4/5: Retrieving knowledge base chunks")
            chunks = self.retriever.retrieve(profile, top_k=6)

            # Step 4: Build prompt and call LLM
            logger.info("Step 5/5: Generating nutrition plan via LLM")
            user_prompt = build_user_prompt(
                patient_id=patient_id,
                markers=profile.markers,
                flagged_values=profile.flagged_values,
                raw_text_summary=profile.raw_text_summary,
                chunks=chunks,
            )

            llm_response = await self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            # Step 5: Parse and validate output
            plan = await self.formatter.parse_with_retry(llm_response, profile, chunks)

        finally:
            os.unlink(tmp_path)

        logger.info(
            f"Pipeline complete for patient {patient_id}. "
            f"Confidence: {plan.confidence_score}. "
            f"Requires review: {plan.requires_doctor_review}"
        )
        return plan
```

**✅ Test 4.1 – 4.5:**
```python
# tests/test_formatter.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.formatter import OutputFormatter
from app.llm.base import LLMResponse
from app.models.genetic_profile import GeneticProfile, GeneticMarker
from app.retriever.retriever import NutritionChunk
from app.prompts import SYSTEM_PROMPT


def make_mock_llm(response_text: str):
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=LLMResponse(
        content=response_text,
        model="test-model",
        prompt_tokens=100,
        completion_tokens=200,
    ))
    return mock


VALID_JSON_RESPONSE = '''
{
  "summary": "This child has MTHFR C677T heterozygous variant affecting folate metabolism.",
  "daily_targets": {"calories": 1800, "protein_g": 60, "carbohydrates_g": 220, "fat_g": 60, "fiber_g": 25, "water_ml": 1500},
  "recommended_foods": ["leafy greens", "lentils", "eggs", "salmon"],
  "foods_to_avoid": ["processed foods", "artificial colors"],
  "supplements": [{"name": "Methylfolate", "dose": "400mcg", "frequency": "once daily", "reason": "MTHFR impairs folic acid conversion"}],
  "meal_timing_notes": "Spread meals evenly across 3 meals and 2 snacks.",
  "notes": "Physician should review methylfolate dosing."
}
'''


@pytest.mark.asyncio
async def test_formatter_parses_valid_json():
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)

    profile = GeneticProfile(
        patient_id="p001",
        markers=[GeneticMarker(gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line="")],
        extraction_confidence=0.9,
    )
    chunks = [NutritionChunk(text="Folate is important.", source="doc.pdf", chunk_index=0, similarity_score=0.85)]
    llm_response = LLMResponse(content=VALID_JSON_RESPONSE, model="test", prompt_tokens=10, completion_tokens=50)

    plan = await formatter.parse_with_retry(llm_response, profile, chunks)

    assert plan.patient_id == "p001"
    assert len(plan.nutrition_plan.recommended_foods) > 0
    assert len(plan.nutrition_plan.supplements) > 0
    assert plan.confidence_score > 0


@pytest.mark.asyncio
async def test_formatter_retries_on_invalid_json():
    """Formatter retries when first response is broken JSON."""
    bad_response = "Here is your plan: oops this is not JSON {"
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)  # Retry returns valid JSON

    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p002", markers=[], extraction_confidence=0.5)
    chunks = []
    llm_response = LLMResponse(content=bad_response, model="test", prompt_tokens=10, completion_tokens=10)

    plan = await formatter.parse_with_retry(llm_response, profile, chunks, max_retries=1)
    # Should have retried and used the valid JSON
    assert plan is not None


@pytest.mark.asyncio
async def test_formatter_returns_fallback_after_all_retries_fail():
    """Formatter returns fallback plan when all retries fail."""
    mock_llm = make_mock_llm("still broken {{{")
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p003", markers=[], extraction_confidence=0.3)
    llm_response = LLMResponse(content="broken", model="test", prompt_tokens=5, completion_tokens=5)

    plan = await formatter.parse_with_retry(llm_response, profile, [], max_retries=1)
    assert plan.requires_doctor_review is True
    assert plan.confidence_score == 0.0


@pytest.mark.asyncio
async def test_requires_doctor_review_is_true_for_low_confidence():
    """Plans with low confidence always require review."""
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p004", markers=[], extraction_confidence=0.1)
    chunks = [NutritionChunk(text="nutrition info", source="doc.pdf", chunk_index=0, similarity_score=0.2)]
    llm_response = LLMResponse(content=VALID_JSON_RESPONSE, model="test", prompt_tokens=10, completion_tokens=50)

    plan = await formatter.parse_with_retry(llm_response, profile, chunks)
    assert plan.requires_doctor_review is True
```

```bash
pytest tests/test_formatter.py -v
```

---

## Phase 5 — FastAPI Layer & Security (Days 16–18)

---

### Task 5.1 – 5.5 — FastAPI Application

**File: `app/middleware/auth.py`**
```python
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
```

**File: `app/main.py`**
```python
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.middleware.auth import verify_api_key
from app.orchestrator import RAGOrchestrator
from app.models.nutrition_plan import NutritionPlan

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# Singleton orchestrator — initialized once at startup
orchestrator: RAGOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    logger.info("Initializing RAG orchestrator...")
    orchestrator = RAGOrchestrator()
    logger.info("Orchestrator ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Autism Nutrition RAG API",
    description="Generates personalized nutrition plans from genetics test reports.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429,
    content={"error": "Rate limit exceeded. Please wait before retrying."},
))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to mobile app domains
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check for mobile team."""
    llm_ok = await orchestrator.llm.health_check() if orchestrator else False
    db_size = orchestrator.retriever.get_collection_size() if orchestrator else 0
    return {
        "status": "ok",
        "llm_backend": settings.LLM_BACKEND,
        "llm_healthy": llm_ok,
        "knowledge_base_chunks": db_size,
    }


@app.post(
    "/generate-nutrition-plan",
    response_model=NutritionPlan,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def generate_nutrition_plan(
    request,  # Required by slowapi
    file: UploadFile = File(..., description="Genetics test report — PDF or image"),
    patient_id: str = Form(..., description="Unique patient identifier"),
):
    """
    Accepts a genetics test report (PDF or image scan) and a patient ID.
    Returns a structured nutrition plan generated from the genetic data.
    """
    # Validate file type
    allowed_types = {"application/pdf", "image/jpeg", "image/png", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    # Validate file size (max 20MB)
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 20MB.",
        )

    start_time = time.time()
    logger.info(f"Request received for patient_id={patient_id}, file={file.filename}")

    try:
        plan = await orchestrator.process(
            file_bytes=file_bytes,
            filename=file.filename,
            patient_id=patient_id,
        )
    except Exception as e:
        logger.error(f"Pipeline error for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during plan generation. Please retry.",
        )

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Request complete for patient {patient_id} in {elapsed}s")
    return plan
```

**✅ Test 5.1 – 5.5:**
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from app.main import app
from app.models.nutrition_plan import NutritionPlan, NutritionPlanContent, DailyTargets
from datetime import datetime
import io

client = TestClient(app)
VALID_API_KEY = "your-secret-api-key-change-this"  # Match .env


def make_mock_plan():
    return NutritionPlan(
        patient_id="test-patient",
        generated_at=datetime.utcnow(),
        genetic_markers_detected=["MTHFR C677T (HETEROZYGOUS)"],
        nutrition_plan=NutritionPlanContent(
            summary="Test plan summary.",
            daily_targets=DailyTargets(calories=1800),
            recommended_foods=["spinach"],
            foods_to_avoid=["processed sugar"],
        ),
        source_chunks_used=["doc.pdf[0]"],
        confidence_score=0.75,
        requires_doctor_review=False,
        llm_model_used="test-model",
    )


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_backend" in data


def test_missing_api_key_returns_401():
    response = client.post("/generate-nutrition-plan")
    assert response.status_code == 401


def test_wrong_api_key_returns_401():
    response = client.post(
        "/generate-nutrition-plan",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_missing_file_returns_422():
    response = client.post(
        "/generate-nutrition-plan",
        headers={"X-API-Key": VALID_API_KEY},
        data={"patient_id": "p001"},
    )
    assert response.status_code == 422


def test_invalid_file_type_returns_415():
    response = client.post(
        "/generate-nutrition-plan",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
        data={"patient_id": "p001"},
    )
    assert response.status_code == 415


@patch("app.main.orchestrator")
def test_valid_request_returns_nutrition_plan(mock_orchestrator):
    """Full happy-path test with mocked orchestrator."""
    mock_orchestrator.llm.health_check = AsyncMock(return_value=True)
    mock_orchestrator.retriever.get_collection_size.return_value = 50
    mock_orchestrator.process = AsyncMock(return_value=make_mock_plan())

    pdf_bytes = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/generate-nutrition-plan",
        headers={"X-API-Key": VALID_API_KEY},
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
        data={"patient_id": "test-patient"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "test-patient"
    assert "nutrition_plan" in data
    assert "confidence_score" in data
    assert "requires_doctor_review" in data
```

```bash
pytest tests/test_api.py -v
```

---

## Phase 6 — Testing & Hardening (Days 19–22)

---

### Task 6.1 — Integration Test (Full Pipeline)

```python
# tests/test_orchestrator.py
import pytest
import fitz
from app.orchestrator import RAGOrchestrator


@pytest.mark.asyncio
async def test_full_pipeline_returns_nutrition_plan():
    """
    Full end-to-end integration test.
    Requires: vLLM running, ChromaDB populated.
    """
    # Create a synthetic genetics report PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "GENETIC ANALYSIS REPORT")
    page.insert_text((50, 140), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    page.insert_text((50, 180), "Gene: COMT  Variant: V158M  Status: Wild Type")
    pdf_bytes = doc.tobytes()
    doc.close()

    orchestrator = RAGOrchestrator()
    plan = await orchestrator.process(
        file_bytes=pdf_bytes,
        filename="integration_test.pdf",
        patient_id="integration-001",
    )

    assert plan.patient_id == "integration-001"
    assert plan.nutrition_plan.summary != ""
    assert plan.confidence_score >= 0.0
    assert isinstance(plan.requires_doctor_review, bool)
    assert len(plan.nutrition_plan.recommended_foods) > 0
    assert plan.llm_model_used != ""
    print(f"\n✅ Integration test passed. Confidence: {plan.confidence_score}")
    print(f"   Markers detected: {plan.genetic_markers_detected}")
    print(f"   Requires review: {plan.requires_doctor_review}")
    print(f"   Recommended foods: {plan.nutrition_plan.recommended_foods[:3]}")
```

```bash
pytest tests/test_orchestrator.py -v -s
```

---

### Task 6.2 — Load Test

```bash
pip install locust
```

**File: `tests/locustfile.py`**
```python
import io
import fitz
from locust import HttpUser, task, between


def make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "MTHFR C677T Heterozygous")
    pdf = doc.tobytes()
    doc.close()
    return pdf


class RAGAPIUser(HttpUser):
    wait_time = between(5, 15)
    host = "http://localhost:8000"

    def on_start(self):
        self.pdf_bytes = make_pdf_bytes()

    @task
    def generate_plan(self):
        self.client.post(
            "/generate-nutrition-plan",
            headers={"X-API-Key": "your-secret-api-key-change-this"},
            files={"file": ("report.pdf", io.BytesIO(self.pdf_bytes), "application/pdf")},
            data={"patient_id": "load-test-patient"},
            timeout=120,
        )
```

```bash
# Run load test: 5 users, spawn 1 per second, run for 60 seconds
locust -f tests/locustfile.py --headless -u 5 -r 1 -t 60s

# Target: 95th percentile response time < 15 seconds
# If p95 > 15s, consider adding async task queue (Celery + Redis)
```

---

### Task 6.3 — Async Task Queue (if needed after load test)

If the load test shows queuing pressure, wrap the orchestrator call in Celery:

```bash
pip install celery[redis]==5.3.6
```

```python
# app/tasks.py
from celery import Celery
import asyncio

celery_app = Celery("rag_tasks", broker="redis://redis:6379/0", backend="redis://redis:6379/1")


@celery_app.task(bind=True, max_retries=2)
def generate_plan_task(self, file_bytes: bytes, filename: str, patient_id: str):
    from app.orchestrator import RAGOrchestrator
    orchestrator = RAGOrchestrator()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            orchestrator.process(file_bytes, filename, patient_id)
        ).model_dump_json()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
    finally:
        loop.close()
```

---

### Task 6.4 — Structured Logging

Already built into `app/main.py`. To verify:

```bash
# Run the API and check log output format
docker compose up api -d
docker compose logs -f api | head -20
# Each line should be valid JSON:
# {"time":"...","level":"INFO","module":"app.main","message":"Orchestrator ready."}

# Parse logs programmatically
docker compose logs api | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{'):
        try:
            d = json.loads(line)
            print(d.get('level'), '-', d.get('message'))
        except:
            pass
"
```

---

### Task 6.5 — OpenAPI Documentation

FastAPI generates this automatically. Share with the mobile team:

```bash
# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Interactive docs (Swagger UI)
open http://localhost:8000/docs

# Raw OpenAPI JSON spec (share this with mobile team)
curl http://localhost:8000/openapi.json > openapi_spec.json

# Or ReDoc format
open http://localhost:8000/redoc
```

The mobile team can import `openapi_spec.json` directly into Postman, Insomnia, or generate typed API clients from it.

---

## Final Checklist Before Handoff

```
Phase 1 — Infrastructure
  [ ] docker compose up runs without errors
  [ ] vLLM responds in < 10s (GPU confirmed)
  [ ] All 4 LLM providers implement generate() and health_check()
  [ ] LLM_BACKEND switch works with zero code changes

Phase 2 — OCR
  [ ] preprocessor handles PDF and image inputs
  [ ] PaddleOCR extracts text from sample lab reports
  [ ] Parser extracts at least the genes present in sample reports
  [ ] GeneticProfile model validates correctly

Phase 3 — Knowledge Base
  [ ] ingest_knowledge_base.py runs without error
  [ ] ChromaDB collection has > 0 chunks
  [ ] Retriever returns relevant chunks for MTHFR query
  [ ] Scores are in descending order

Phase 4 — LLM + Formatting
  [ ] Prompt templates render without errors
  [ ] Formatter parses valid JSON on first attempt
  [ ] Formatter retries and succeeds on broken JSON
  [ ] Fallback plan returned when all retries fail
  [ ] requires_doctor_review=True for low confidence plans
  [ ] End-to-end orchestrator test passes

Phase 5 — API
  [ ] /health returns 200 with llm and db status
  [ ] Missing API key → 401
  [ ] Wrong file type → 415
  [ ] Valid request → 200 with NutritionPlan JSON
  [ ] Rate limiter returns 429 after limit exceeded

Phase 6 — Hardening
  [ ] Integration test passes on real PDFs
  [ ] Load test p95 < 15 seconds
  [ ] Logs are structured JSON
  [ ] openapi_spec.json shared with mobile team
```
