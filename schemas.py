"""
schemas.py  –  Pydantic request & response models for the Plant-Disease API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Response models ───────────────────────────────────────────────────────────

class TopKItem(BaseModel):
    """A single entry in the top-k prediction list."""
    class_name:  str   = Field(..., alias="class")
    confidence:  float

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """Returned by POST /predict and POST /predict/batch (per-image)."""
    predicted_class: str
    class_id:        int
    confidence:      float
    top_k:           list[TopKItem]
    all_probs:       dict[str, float]


class BatchPredictItem(BaseModel):
    """One row in a batch-inference response."""
    index:           int
    predicted_class: str
    confidence:      float


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictItem]


class HealthResponse(BaseModel):
    status:  str
    model:   str
    classes: int
    device:  str