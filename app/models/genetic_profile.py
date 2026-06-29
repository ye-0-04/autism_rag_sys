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

        return " | ".join(parts[:5])

    def has_actionable_markers(self) -> bool:
        return any(
            m.status in ["HOMOZYGOUS_VARIANT", "HETEROZYGOUS"] for m in self.markers
        )
