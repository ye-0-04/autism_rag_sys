import pytest
import fitz
from app.orchestrator import RAGOrchestrator


@pytest.mark.asyncio
async def test_full_pipeline_returns_nutrition_plan():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "GENETIC ANALYSIS REPORT")
    page.insert_text((50, 140), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    page.insert_text((50, 180), "Gene: COMT  Variant: V158M  Status: Wild Type")
    pdf_bytes = doc.tobytes()
    doc.close()

    orchestrator = RAGOrchestrator()
    if not await orchestrator.llm.health_check():
        pytest.skip(f"Configured LLM backend ({orchestrator.llm.__class__.__name__}) is not running/healthy")
        
    plan = await orchestrator.process(
        file_bytes=pdf_bytes,
        filename="integration_test.pdf",
        patient_id="integration-001",
    )

    assert plan.patient_id == "integration-001"
    assert plan.nutrition_plan.summary != ""
    assert plan.confidence_score >= 0.0
    assert isinstance(plan.requires_doctor_review, bool)
    assert plan.llm_model_used != ""
