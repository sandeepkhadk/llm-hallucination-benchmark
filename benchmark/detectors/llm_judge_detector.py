"""
LLM-as-a-Judge paradigm (Methodology III-A/C): a judge model reads the
generator's output and flags unsupported claims. Cost concentrates in a full
autoregressive forward pass, heavily impacting TTFT/throughput, with modest,
predictable space complexity (single-sequence KV-cache, plus judge weights
if not shared with the generator).
"""
from __future__ import annotations

import re
import time

from detectors.base import DetectionResult, HallucinationDetector
from engine.inference_engine import InferenceEngine

JUDGE_PROMPT_TEMPLATE = """You are a fact-checking judge. Given a question and a \
candidate answer, decide whether the answer contains hallucinated \
(unsupported or fabricated) content.

Question: {prompt}
Candidate answer: {answer}

Respond with exactly one line: "VERDICT: HALLUCINATION" or "VERDICT: FAITHFUL", \
followed by a one-sentence justification."""


class LLMAsJudgeDetector(HallucinationDetector):
    name = "llm_as_judge"

    def __init__(self, judge_engine: InferenceEngine, max_new_tokens: int = 64):
        self.judge_engine = judge_engine
        self.max_new_tokens = max_new_tokens

    def detect(self, prompt: str, generated_text: str, **kwargs) -> DetectionResult:
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, answer=generated_text)

        result = self.judge_engine.generate([judge_prompt], max_new_tokens=self.max_new_tokens, n=1)[0][0]

        verdict_match = re.search(r"VERDICT:\s*(HALLUCINATION|FAITHFUL)", result.text, re.IGNORECASE)
        is_hallucination = bool(verdict_match) and verdict_match.group(1).upper() == "HALLUCINATION"

        return DetectionResult(
            is_hallucination=is_hallucination,
            score=1.0 if is_hallucination else 0.0,
            metrics=dict(
                ttft_ms=result.ttft_ms,
                latency_ms=result.latency_ms,
                throughput_tps=result.throughput_tps,
                judge_output_tokens=result.output_tokens,
                raw_verdict=result.text.strip()[:200],
            ),
        )
