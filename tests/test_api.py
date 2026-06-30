import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from app.main import app
from app.models.nutrition_plan import NutritionPlan, NutritionPlanContent, DailyTargets
from datetime import datetime
import io

client = TestClient(app)
VALID_API_KEY = "your-secret-api-key-change-this"


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


@patch("app.main.orchestrator")
def test_rate_limiter_returns_429_after_limit(mock_orchestrator):
    mock_orchestrator.process = AsyncMock(return_value=make_mock_plan())
    mock_orchestrator.llm.health_check = AsyncMock(return_value=True)
    mock_orchestrator.retriever.get_collection_size.return_value = 50

    pdf_bytes = b"%PDF-1.4 fake pdf content"
    headers = {"X-API-Key": VALID_API_KEY}
    files = {"file": ("report.pdf", pdf_bytes, "application/pdf")}
    data = {"patient_id": "rate-test"}

    for i in range(15):
        resp = client.post(
            "/generate-nutrition-plan", headers=headers, files=files, data=data
        )
        if resp.status_code == 429:
            return

    pytest.fail("Rate limiter did not return 429 after 15 requests")
