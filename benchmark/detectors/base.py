"""Common interfaces for the three hallucination-detection paradigms studied
in Methodology Section III (SelfCheckGPT, RAG Verification, LLM-as-a-Judge)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DetectionResult:
    is_hallucination: bool
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class HallucinationDetector:
    name: str = "base"

    def detect(self, prompt: str, generated_text: str, **kwargs) -> DetectionResult:
        raise NotImplementedError
