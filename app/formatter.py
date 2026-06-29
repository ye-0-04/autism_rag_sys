import json
import re
import logging
from typing import Optional
from app.models.nutrition_plan import (
    NutritionPlan,
    NutritionPlanContent,
    DailyTargets,
    Supplement,
)
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
                return self._build_nutrition_plan(parsed, profile, chunks, llm_response)

            attempt += 1
            if attempt > max_retries:
                break

            logger.warning(
                f"JSON parse failed (attempt {attempt}). Retrying with correction prompt."
            )
            correction = CORRECTION_PROMPT.format(previous_response=raw_content[:500])
            retry_response = await self.llm.generate(
                user_prompt=correction,
                system_prompt=self.system_prompt,
                temperature=0.1,
            )
            raw_content = retry_response.content

        logger.error("All retry attempts failed. Returning fallback nutrition plan.")
        return self._fallback_plan(profile, chunks, llm_response)

    def _try_parse_json(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

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
                daily_targets=DailyTargets(
                    **{
                        k: v
                        for k, v in daily_targets_data.items()
                        if k in DailyTargets.model_fields and v is not None
                    }
                ),
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
