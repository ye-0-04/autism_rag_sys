import pytest
from app.retriever.retriever import NutritionRetriever, NutritionChunk
from app.models.genetic_profile import GeneticProfile, GeneticMarker


def test_retriever_collection_is_not_empty():
    retriever = NutritionRetriever()
    count = retriever.get_collection_size()
    assert count > 0, (
        f"Collection is empty. Run: python scripts/ingest_knowledge_base.py"
    )


def test_retriever_returns_chunks_for_mthfr_profile():
    retriever = NutritionRetriever()
    profile = GeneticProfile(
        patient_id="test-001",
        markers=[
            GeneticMarker(
                gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line=""
            )
        ],
    )
    chunks = retriever.retrieve(profile, top_k=4)

    assert len(chunks) > 0
    assert all(isinstance(c, NutritionChunk) for c in chunks)
    assert all(0.0 <= c.similarity_score <= 1.0 for c in chunks)
    combined_text = " ".join(c.text.lower() for c in chunks)
    assert any(
        keyword in combined_text
        for keyword in ["folate", "methyl", "b12", "nutrition", "vitamin"]
    ), "Retrieved chunks seem unrelated to MTHFR nutrition"


def test_retriever_returns_fewer_chunks_than_requested_if_db_is_small():
    retriever = NutritionRetriever()
    profile = GeneticProfile(patient_id="test", markers=[])
    chunks = retriever.retrieve(profile, top_k=1000)
    assert len(chunks) <= retriever.get_collection_size()


def test_retrieval_quality_scores_are_ordered():
    retriever = NutritionRetriever()
    profile = GeneticProfile(
        patient_id="test",
        markers=[
            GeneticMarker(
                gene="COMT", variant="V158M", status="HOMOZYGOUS_VARIANT", raw_line=""
            )
        ],
    )
    chunks = retriever.retrieve(profile, top_k=5)
    scores = [c.similarity_score for c in chunks]
    assert scores == sorted(scores, reverse=True), (
        "Chunks not in descending score order"
    )
