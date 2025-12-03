"""
CLIP-based similarity scoring between UI crops and textual prompts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class ClipPrompt:
    label: str
    prompts: Sequence[str]


PROMPT_BANK: List[ClipPrompt] = [
    ClipPrompt(
        label="Disguised Ad",
        prompts=(
            "a sponsored advertisement disguised as normal content",
            "a fake article that is actually an advertisement",
            "an image showing a deceptive ad banner on a website",
        ),
    ),
    ClipPrompt(
        label="Fake Scarcity",
        prompts=(
            "a popup showing only a few items left or high demand",
            "an e-commerce screenshot with false scarcity messaging",
        ),
    ),
    ClipPrompt(
        label="Misdirection Button",
        prompts=(
            "a bright colored button tricking the user",
            "a misleading download button on a cluttered page",
        ),
    ),
    ClipPrompt(
        label="Confirm-Shaming",
        prompts=(
            "a dialog that shames the user for declining an offer",
            "a guilt-tripping opt-out button",
        ),
    ),
    ClipPrompt(
        label="Color Manipulation",
        prompts=(
            "a high contrast color highlight drawing attention unfairly",
            "an interface using intense red or green to bias a choice",
        ),
    ),
]


class ClipScorer:
    """Wrapper around Hugging Face CLIP for region scoring."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        prompt_bank: Iterable[ClipPrompt] = PROMPT_BANK,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.prompts: List[ClipPrompt] = list(prompt_bank)

    def score_crop(self, crop: Image.Image) -> Dict[str, float]:
        """Return probability scores per prompt label for the supplied image."""

        text_inputs = [prompt for bank in self.prompts for prompt in bank.prompts]
        if not text_inputs:
            return {}

        with torch.no_grad():
            inputs = self.processor(
                text=text_inputs,
                images=crop,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().tolist()[0]

        # Collapse prompt-level probabilities into label-level averages.
        label_scores: Dict[str, List[float]] = {}
        idx = 0
        for bank in self.prompts:
            scores = []
            for _ in bank.prompts:
                scores.append(probs[idx])
                idx += 1
            label_scores[bank.label] = scores

        return {label: float(sum(scores) / len(scores)) for label, scores in label_scores.items()}

    def rank_crop(self, crop: Image.Image) -> Dict[str, object]:
        """Return the best-matching label and metadata for a crop."""

        start = time.perf_counter()
        scores = self.score_crop(crop)
        if not scores:
            return {
                "label": "Unknown",
                "score": 0.0,
                "scores": {},
                "latency_ms": (time.perf_counter() - start) * 1000,
            }

        label, score = max(scores.items(), key=lambda pair: pair[1])
        return {
            "label": label,
            "score": score,
            "scores": scores,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

