"""
Unified inference-engine interface with TTFT/latency/throughput measurement
(Methodology III-A/D). vLLM (v0.4.0-compatible offline API) is the primary,
methodology-faithful backend; see `transformers_engine.py` and
`mock_engine.py` for CPU-friendly alternatives used during development.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationResult:
    prompt: str
    text: str
    ttft_ms: float
    latency_ms: float
    throughput_tps: float
    prompt_tokens: int
    output_tokens: int
    seed: Optional[int] = None  # random seed used for this generation, if any (Methodology III-B)


class InferenceEngine:
    """Common interface implemented by every backend (vLLM, transformers, mock)."""

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        n: int = 1,
        seeds: Optional[List[List[int]]] = None,
    ) -> List[List[GenerationResult]]:
        """`seeds`, if given, is a list aligned with `prompts`, each entry a
        list of `n` per-sample seeds, applied immediately before that
        prompt/sample's generation call for reproducibility (Methodology
        III-B). `seeds=None` leaves generation unseeded."""
        raise NotImplementedError

    def warmup(self, prompt: str, n_runs: int, max_new_tokens: int = 32) -> None:
        """Runs `n_runs` throwaway generations to mitigate cold-start GPU
        initialization effects (Methodology III-C)."""
        for _ in range(n_runs):
            self.generate([prompt], max_new_tokens=max_new_tokens, n=1)


class VLLMEngine(InferenceEngine):
    """Wraps vLLM's offline `LLM` API (Methodology III-D: vLLM v0.4.0)."""

    def __init__(self, model_name: str, dtype: str = "float16", **engine_kwargs):
        from vllm import LLM, SamplingParams  # heavy optional dependency

        self._SamplingParams = SamplingParams
        self.model_name = model_name
        self.llm = LLM(model=model_name, dtype=dtype, **engine_kwargs)

    def generate(self, prompts, max_new_tokens=256, n=1, seeds=None):
        # Not used for the reported --lite/--paper runs (Methodology III-B
        # uses the transformers backend); seeds is applied best-effort as a
        # single per-prompt seed (vLLM applies it to all n completions of
        # that prompt) for interface parity with TransformersEngine.
        if seeds is not None:
            sp = [self._SamplingParams(max_tokens=max_new_tokens, n=n, seed=seeds[i][0]) for i in range(len(prompts))]
        else:
            sp = self._SamplingParams(max_tokens=max_new_tokens, n=n)
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        total_latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for i, prompt_out in enumerate(outputs):
            per_prompt = []
            n_prompt_tokens = len(prompt_out.prompt_token_ids)
            prompt_seed = seeds[i][0] if seeds is not None else None
            for completion in prompt_out.outputs:
                n_out_tokens = len(completion.token_ids)
                # vLLM's offline (non-streaming) API doesn't expose a native
                # first-token timestamp; approximate TTFT as the per-token
                # share of total latency. Use vLLM's AsyncLLMEngine for a
                # directly measured, streaming-based TTFT.
                ttft_ms = total_latency_ms / max(n_out_tokens, 1)
                throughput = n_out_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0
                per_prompt.append(
                    GenerationResult(
                        prompt=prompt_out.prompt,
                        text=completion.text,
                        ttft_ms=ttft_ms,
                        latency_ms=total_latency_ms,
                        throughput_tps=throughput,
                        prompt_tokens=n_prompt_tokens,
                        output_tokens=n_out_tokens,
                        seed=prompt_seed,
                    )
                )
            results.append(per_prompt)
        return results
