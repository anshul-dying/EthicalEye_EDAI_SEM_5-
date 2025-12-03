"""
Configuration helpers for the vision microservice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    """Runtime configuration for the vision analyzer."""

    model_name: str = os.getenv("VISION_CLIP_MODEL", "openai/clip-vit-base-patch32")
    device: str | None = os.getenv("VISION_DEVICE")
    max_regions: int = int(os.getenv("VISION_MAX_REGIONS", 18))
    confidence_threshold: float = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", 0.35))
    enable_ocr: bool = os.getenv("VISION_ENABLE_OCR", "1") not in {"0", "false", "False"}
    enable_heuristics: bool = (
        os.getenv("VISION_ENABLE_HEURISTICS", "1") not in {"0", "false", "False"}
    )
    enable_text_classifier: bool = (
        os.getenv("VISION_ENABLE_TEXT_CLASSIFIER", "0") not in {"0", "false", "False"}
    )
    detector_min_area_ratio: float = float(os.getenv("VISION_DETECTOR_MIN_AREA", 0.0025))


def get_config() -> VisionConfig:
    """Return a cached config instance."""

    # dataclasses are lightweight; caching unnecessary but available for clarity.
    return VisionConfig()

