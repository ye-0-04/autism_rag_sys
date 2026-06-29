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
    mock.generate = AsyncMock(
        return_value=LLMResponse(
            content=response_text,
            model="test-model",
            prompt_tokens=100,
            completion_tokens=200,
        )
    )
    return mock


VALID_JSON_RESPONSE = """
{
  "summary": "This child has MTHFR C677T heterozygous variant affecting folate metabolism.",
  "daily_targets": {"calories": 1800, "protein_g": 60, "carbohydrates_g": 220, "fat_g": 60, "fiber_g": 25, "water_ml": 1500},
  "recommended_foods": ["leafy greens", "lentils", "eggs", "salmon"],
  "foods_to_avoid": ["processed foods", "artificial colors"],
  "supplements": [{"name": "Methylfolate", "dose": "400mcg", "frequency": "once daily", "reason": "MTHFR impairs folic acid conversion"}],
  "meal_timing_notes": "Spread meals evenly across 3 meals and 2 snacks.",
  "notes": "Physician should review methylfolate dosing."
}
"""


@pytest.mark.asyncio
async def test_formatter_parses_valid_json():
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)

    profile = GeneticProfile(
        patient_id="p001",
        markers=[
            GeneticMarker(
                gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line=""
            )
        ],
        extraction_confidence=0.9,
    )
    chunks = [
        NutritionChunk(
            text="Folate is important.",
            source="doc.pdf",
            chunk_index=0,
            similarity_score=0.85,
        )
    ]
    llm_response = LLMResponse(
        content=VALID_JSON_RESPONSE,
        model="test",
        prompt_tokens=10,
        completion_tokens=50,
    )

    plan = await formatter.parse_with_retry(llm_response, profile, chunks)

    assert plan.patient_id == "p001"
    assert len(plan.nutrition_plan.recommended_foods) > 0
    assert len(plan.nutrition_plan.supplements) > 0
    assert plan.confidence_score > 0


@pytest.mark.asyncio
async def test_formatter_retries_on_invalid_json():
    bad_response = "Here is your plan: oops this is not JSON {"
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)

    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p002", markers=[], extraction_confidence=0.5)
    chunks = []
    llm_response = LLMResponse(
        content=bad_response, model="test", prompt_tokens=10, completion_tokens=10
    )

    plan = await formatter.parse_with_retry(
        llm_response, profile, chunks, max_retries=1
    )
    assert plan is not None


@pytest.mark.asyncio
async def test_formatter_returns_fallback_after_all_retries_fail():
    mock_llm = make_mock_llm("still broken {{{")
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p003", markers=[], extraction_confidence=0.3)
    llm_response = LLMResponse(
        content="broken", model="test", prompt_tokens=5, completion_tokens=5
    )

    plan = await formatter.parse_with_retry(llm_response, profile, [], max_retries=1)
    assert plan.requires_doctor_review is True
    assert plan.confidence_score == 0.0


@pytest.mark.asyncio
async def test_requires_doctor_review_is_true_for_low_confidence():
    mock_llm = make_mock_llm(VALID_JSON_RESPONSE)
    formatter = OutputFormatter(mock_llm, SYSTEM_PROMPT)
    profile = GeneticProfile(patient_id="p004", markers=[], extraction_confidence=0.1)
    chunks = [
        NutritionChunk(
            text="nutrition info", source="doc.pdf", chunk_index=0, similarity_score=0.2
        )
    ]
    llm_response = LLMResponse(
        content=VALID_JSON_RESPONSE,
        model="test",
        prompt_tokens=10,
        completion_tokens=50,
    )

    plan = await formatter.parse_with_retry(llm_response, profile, chunks)
    assert plan.requires_doctor_review is True
