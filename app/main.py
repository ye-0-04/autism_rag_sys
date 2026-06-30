import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
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
app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before retrying."},
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
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
    request: Request,
    file: UploadFile = File(..., description="Genetics test report — PDF or image"),
    patient_id: str = Form(..., description="Unique patient identifier"),
):
    allowed_types = {"application/pdf", "image/jpeg", "image/png", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(allowed_types)}",
        )

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
