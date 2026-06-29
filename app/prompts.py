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
    markers_text = (
        "\n".join(f"- {m.gene} | {m.variant} | {m.status}" for m in markers)
        or "No markers successfully extracted."
    )

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
