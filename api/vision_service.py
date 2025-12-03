"""
FastAPI microservice that exposes screenshot dark-pattern detection.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.vision import VisionAnalyzer

logger = logging.getLogger("vision_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Ethical Eye Vision Service",
    description="Multimodal detection of dark patterns from screenshots.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = VisionAnalyzer()


class Detection(BaseModel):
    label: str
    score: float
    reason: str
    bbox: List[int]
    text: str | None = None
    heuristics: dict | None = None


class VisionResponse(BaseModel):
    detections: List[Detection]
    metadata: dict


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/vision/analyze", response_model=VisionResponse)
async def analyze_screenshot(file: UploadFile = File(...)) -> VisionResponse:
    try:
        contents = await file.read()
        detections, metadata = analyzer.analyze(contents)
        return VisionResponse(detections=detections, metadata=metadata)
    except Exception as exc:  # pragma: no cover - FastAPI handles stack traces
        logger.exception("Vision analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.vision_service:app", host="0.0.0.0", port=8000, reload=False)

