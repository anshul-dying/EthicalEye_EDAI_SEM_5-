"""
Lightweight OCR helper built on pytesseract with graceful fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytesseract
from PIL import Image


@dataclass
class OcrResult:
    text: str
    confidence: float


class OcrExtractor:
    """Extract readable text snippets for downstream analysis."""

    def __init__(self) -> None:
        self._available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract(self, image: Image.Image | np.ndarray) -> OcrResult:
        if not self._available:
            return OcrResult(text="", confidence=0.0)

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)

        try:
            raw_text = pytesseract.image_to_string(
                pil_image, config="--oem 3 --psm 6"
            )
            cleaned = " ".join(raw_text.split())
            confidence = min(len(cleaned) / 64.0, 1.0)
            return OcrResult(text=cleaned, confidence=confidence)
        except Exception:
            return OcrResult(text="", confidence=0.0)

