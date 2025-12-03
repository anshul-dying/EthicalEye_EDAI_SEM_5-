"""
Orchestrates preprocessing, CLIP scoring, and heuristics fusion.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .clip_scorer import ClipScorer
from .config import VisionConfig, get_config
from .detector import VisionDetector
from .ocr import OcrExtractor


class VisionAnalyzer:
    """High-level pipeline entry point for screenshot analysis."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or get_config()
        self.detector = VisionDetector(min_area_ratio=self.config.detector_min_area_ratio)
        self.ocr = OcrExtractor()
        self.clip = ClipScorer(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self.logger = logging.getLogger("vision.analyzer")
        self.text_classifier = self._init_text_classifier()

    def analyze(self, image_bytes: bytes) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        """Return detections and metadata for the supplied screenshot bytes."""

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        start = time.perf_counter()

        layout_stats = self.detector.layout_stats(bgr)
        proposals = self.detector.propose_regions(
            bgr_image=bgr, max_regions=self.config.max_regions
        )

        detections: List[Dict[str, object]] = []

        for proposal in proposals:
            clip_result = self.clip.rank_crop(proposal.crop)
            heuristics = self._compute_heuristics(proposal, layout_stats, bgr_image=bgr)
            ocr_result = self.ocr.extract(proposal.crop) if self.config.enable_ocr else None
            text_analysis = self._score_text(ocr_result.text) if ocr_result else None

            fused = self._fuse_scores(
                clip_result, heuristics, ocr_result, text_analysis
            )
            if fused["score"] < self.config.confidence_threshold:
                continue

            detections.append(
                {
                    "label": fused["label"],
                    "score": fused["score"],
                    "reason": fused["reason"],
                    "bbox": list(proposal.bbox),
                    "text": ocr_result.text if ocr_result else "",
                    "heuristics": heuristics,
                    "text_category": text_analysis.get("category") if text_analysis else None,
                    "text_confidence": text_analysis.get("confidence") if text_analysis else None,
                }
            )

        runtime = (time.perf_counter() - start) * 1000
        metadata = {
            "model": self.config.model_name,
            "device": self.clip.device,
            "regions_evaluated": len(proposals),
            "detections": len(detections),
            "runtime_ms": runtime,
        }
        return detections, metadata

    def _compute_heuristics(
        self,
        proposal,
        layout_stats: Dict[str, float],
        bgr_image: np.ndarray,
    ) -> Dict[str, float]:
        """Return supplemental heuristics used during fusion."""

        if not self.config.enable_heuristics:
            return {}

        x, y, w, h = proposal.bbox
        crop = bgr_image[y : y + h, x : x + w]
        contrast = self.detector.color_contrast(crop)

        heuristics = {
            "area_ratio": proposal.area_ratio,
            "button_like": 1.0 if proposal.is_button_like else 0.0,
            "color_contrast": contrast,
            "layout_edge_density": layout_stats["edge_density"],
        }
        return heuristics

    @staticmethod
    def _fuse_scores(clip_result, heuristics, ocr_result, text_analysis) -> Dict[str, object]:
        """Combine CLIP output, heuristics, and OCR-derived hints."""

        score = clip_result["score"]
        label = clip_result["label"]

        if label == "Color Manipulation":
            score += 0.25 * heuristics.get("color_contrast", 0.0)
        if label == "Misdirection Button" and heuristics.get("button_like"):
            score += 0.15
        if heuristics.get("area_ratio", 0.0) > 0.2 and label == "Disguised Ad":
            score += 0.1

        if ocr_result and ocr_result.text:
            lowered = ocr_result.text.lower()
            if any(term in lowered for term in ("only left", "hurry", "deal ends")):
                score += 0.15
                label = "Fake Scarcity"
            if any(term in lowered for term in ("no thanks", "continue", "download")):
                score += 0.1
                label = "Misdirection Button"

        if text_analysis and text_analysis.get("is_dark_pattern"):
            if text_analysis["confidence"] > score:
                label = text_analysis["category"]
                score = text_analysis["confidence"]

        score = float(min(score, 0.99))
        reason = VisionAnalyzer._compose_reason(
            label, score, heuristics, ocr_result, text_analysis
        )

        return {"label": label, "score": score, "reason": reason}

    @staticmethod
    def _compose_reason(label: str, score: float, heuristics, ocr_result, text_analysis) -> str:
        cues = []
        if heuristics.get("button_like"):
            cues.append("button-like shape")
        if heuristics.get("color_contrast", 0.0) > 0.45:
            cues.append("high color contrast")
        if heuristics.get("area_ratio", 0.0) > 0.18:
            cues.append("large footprint")
        if ocr_result and ocr_result.text:
            cues.append(f"OCR text: '{ocr_result.text[:40]}'")
        if text_analysis and text_analysis.get("category"):
            cues.append(f"text classified as {text_analysis['category']}")

        cue_text = "; ".join(cues) if cues else "visual features"
        return f"{label} with {score:.0%} confidence based on {cue_text}"

    def _init_text_classifier(self):
        if not self.config.enable_text_classifier:
            return None
        try:
            from api.ethical_eye_api import EthicalEyeAPI

            self.logger.info("Loading text classifier for OCR fusion...")
            return EthicalEyeAPI()
        except Exception as exc:
            self.logger.warning("Text classifier unavailable: %s", exc)
            return None

    def _score_text(self, text: str | None):
        if not text or not self.text_classifier:
            return None
        analysis = self.text_classifier.analyze_text(text)
        return {
            "category": analysis.get("category"),
            "confidence": float(analysis.get("confidence", 0.0)),
            "is_dark_pattern": analysis.get("is_dark_pattern", False),
        }

