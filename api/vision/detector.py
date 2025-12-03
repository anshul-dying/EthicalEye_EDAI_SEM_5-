"""
Image preprocessing utilities for visual dark-pattern detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class RegionProposal:
    """Representation of a UI region worthy of analysis."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_ratio: float
    aspect_ratio: float
    mean_color: Tuple[float, float, float]
    is_button_like: bool
    crop: Image.Image


class VisionDetector:
    """Construct heuristics-based region proposals and layout stats."""

    def __init__(self, min_area_ratio: float = 0.0025) -> None:
        self.min_area_ratio = min_area_ratio

    def propose_regions(
        self,
        bgr_image: np.ndarray,
        max_regions: int = 18,
    ) -> List[RegionProposal]:
        """Return cropped regions likely to contain UI elements."""

        height, width = bgr_image.shape[:2]
        min_area = width * height * self.min_area_ratio

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        proposals: List[RegionProposal] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or w < 32 or h < 32:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.2 or aspect_ratio > 8.5:
                continue

            crop = bgr_image[y : y + h, x : x + w]
            mean_color = tuple(map(float, crop.mean(axis=(0, 1))))

            proposal = RegionProposal(
                bbox=(x, y, w, h),
                area_ratio=area / float(width * height),
                aspect_ratio=aspect_ratio,
                mean_color=mean_color,
                is_button_like=self._looks_like_button(w, h, aspect_ratio),
                crop=_convert_crop_to_pil(crop),
            )
            proposals.append(proposal)

        proposals.sort(key=lambda p: p.area_ratio, reverse=True)
        return proposals[:max_regions]

    def layout_stats(self, bgr_image: np.ndarray) -> Dict[str, float]:
        """Compute coarse layout statistics used for heuristics."""

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        median_sat = float(np.median(hsv[:, :, 1]) / 255.0)
        median_val = float(np.median(hsv[:, :, 2]) / 255.0)

        edges = cv2.Canny(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = float(edges.mean() / 255.0)

        return {
            "median_saturation": median_sat,
            "median_value": median_val,
            "edge_density": edge_density,
        }

    def color_contrast(self, crop: np.ndarray) -> float:
        """Return a normalized color contrast estimate for a crop."""

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0

        sat_range = float(saturation.max() - saturation.min())
        val_range = float(value.max() - value.min())
        return (sat_range + val_range) / 2.0

    @staticmethod
    def _looks_like_button(width: int, height: int, aspect_ratio: float) -> bool:
        perimeter = 2 * (width + height)
        area = width * height
        solidity = area / float(perimeter ** 2 + 1e-6)
        return 0.18 <= aspect_ratio <= 4.5 and solidity > 0.002


def _convert_crop_to_pil(crop: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

