"""
routes.py  –  FastAPI router for plant-disease prediction endpoints.

Endpoints
---------
GET  /              health-check
POST /predict       single-image inference  (multipart file upload)
POST /predict/batch multi-image inference   (multipart file upload, ≤ N files)
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from config import CFG
from predictor import PlantDiseasePredictor, get_predictor
from schemas import (
    BatchPredictItem,
    BatchPredictResponse,
    HealthResponse,
    PredictResponse,
    TopKItem,
)

router = APIRouter()

# ── helpers ───────────────────────────────────────────────────────────────────

MAX_BATCH = CFG.get("MAX_BATCH_SIZE", 16)


def _bytes_to_rgb(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into an RGB NumPy array."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid image.",
        )
    return np.array(img)


def _build_predict_response(raw: dict) -> PredictResponse:
    """Map the predictor's raw dict to the Pydantic response model."""
    return PredictResponse(
        predicted_class=raw["class"],
        class_id=raw["class_id"],
        confidence=round(raw["confidence"], 4),
        top_k=[
            TopKItem(**{"class": item["class"], "confidence": round(item["confidence"], 4)})
            for item in raw["top_k"]
        ],
        all_probs={k: round(v, 6) for k, v in raw["all_probs"].items()},
    )


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=HealthResponse, tags=["health"])
def health_check(predictor: PlantDiseasePredictor = Depends(get_predictor)):
    """Return server & model status."""
    return HealthResponse(
        status="running",
        model="YOLOv7Classifier",
        classes=predictor.nc,
        device=str(predictor.device),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    summary="Single-image plant-disease prediction",
)
async def predict_single(
    file:      UploadFile                = File(..., description="Plant-leaf image (JPEG / PNG)"),
    use_tta:   bool                      = True,
    top_k:     int                       = 3,
    predictor: PlantDiseasePredictor     = Depends(get_predictor),
):
    """
    Upload a single plant-leaf image and receive a prediction.

    - **use_tta**: enable test-time augmentation (slightly slower, more accurate)
    - **top_k**: number of top classes to return (default 3)
    """
    img  = _bytes_to_rgb(await file.read())
    raw  = predictor.predict(img, use_tta=use_tta, top_k=top_k)
    return _build_predict_response(raw)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    summary="Batch plant-disease prediction",
)
async def predict_batch(
    files:     list[UploadFile]          = File(..., description=f"Up to {MAX_BATCH} images"),
    use_tta:   bool                      = False,
    predictor: PlantDiseasePredictor     = Depends(get_predictor),
):
    """
    Upload multiple plant-leaf images and receive a prediction for each.

    Images are processed sequentially (TTA off by default for speed).
    """
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH}.",
        )

    images = [_bytes_to_rgb(await f.read()) for f in files]
    raws   = predictor.predict_batch(images, use_tta=use_tta)

    results = [
        BatchPredictItem(
            index=i,
            predicted_class=r["class"],
            confidence=round(r["confidence"], 4),
        )
        for i, r in enumerate(raws)
    ]
    return BatchPredictResponse(results=results)
