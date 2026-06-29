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
    requires_doctor_review: bool = True
    llm_model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
