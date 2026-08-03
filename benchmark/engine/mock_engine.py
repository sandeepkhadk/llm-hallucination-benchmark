"""
Deterministic, dependency-free mock engine for exercising the benchmarking
pipeline (dataset prep -> detection -> stats -> plots) without requiring
model downloads or a GPU. NOT a substitute for the real vLLM/transformers
backends when producing reported results.
"""
from __future__ import annotations

import random
from typing import List

from engine.inference_engine import GenerationResult, InferenceEngine


class MockEngine(InferenceEngine):
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, prompts: List[str], max_new_tokens: int = 256, n: int = 1):
        results = []
        for prompt in prompts:
            per_prompt = []
            for _ in range(n):
                n_out = self._rng.randint(max(10, max_new_tokens // 4), max_new_tokens)
                ttft = self._rng.uniform(20, 80)
                latency = ttft + n_out * self._rng.uniform(5, 15)
                throughput = n_out / (latency / 1000)
                vocab = ["fact", "claim", "detail", "context", "answer", "however", "therefore"]
                text = " ".join(self._rng.choice(vocab) for _ in range(n_out // 5 + 1))
                per_prompt.append(
                    GenerationResult(
                        prompt=prompt,
                        text=text,
                        ttft_ms=ttft,
                        latency_ms=latency,
                        throughput_tps=throughput,
                        prompt_tokens=len(prompt.split()),
                        output_tokens=n_out,
                    )
                )
            results.append(per_prompt)
        return results
