"""
SelfCheckGPT paradigm (Methodology III-A/C): samples N additional generations
and measures agreement. Time complexity O(N*L); space complexity dominated by
the KV-cache for N concurrent sequences during decoding, O(N*L*d_model).
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from detectors.base import DetectionResult, HallucinationDetector
from engine.inference_engine import InferenceEngine
from engine.memory_profiler import estimate_kv_cache_mb, peak_vram_mb, reset_peak_vram


def _lexical_overlap_scorer(main_text: str, samples: List[str]) -> float:
    """Jaccard-overlap consistency proxy, used as a light-weight stand-in for
    the NLI/BERTScore scorer in Manakul et al. [2]. Pass a custom
    `consistency_scorer` (e.g., an NLI contradiction-rate scorer) for a
    higher-fidelity reproduction of the original method."""
    main_tokens = set(main_text.lower().split())
    if not main_tokens:
        return 0.0
    scores = []
    for s in samples:
        s_tokens = set(s.lower().split())
        union = main_tokens | s_tokens
        inter = main_tokens & s_tokens
        scores.append(len(inter) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


class SelfCheckGPTDetector(HallucinationDetector):
    name = "selfcheckgpt"

    def __init__(
        self,
        engine: InferenceEngine,
        n_samples: int = 5,
        consistency_scorer: Optional[Callable[[str, List[str]], float]] = None,
        consistency_threshold: float = 0.5,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ):
        self.engine = engine
        self.n_samples = n_samples
        self.consistency_scorer = consistency_scorer or _lexical_overlap_scorer
        self.consistency_threshold = consistency_threshold
        self._model_shape = dict(num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, dtype_bytes=dtype_bytes)

    def detect(self, prompt: str, generated_text: str, max_new_tokens: int = 256, **kwargs) -> DetectionResult:
        reset_peak_vram()
        start = time.perf_counter()
        sample_results = self.engine.generate([prompt], max_new_tokens=max_new_tokens, n=self.n_samples)[0]
        elapsed_s = time.perf_counter() - start

        samples_text = [r.text for r in sample_results]
        consistency = self.consistency_scorer(generated_text, samples_text)
        is_hallucination = consistency < self.consistency_threshold

        seq_len = max((r.output_tokens for r in sample_results), default=max_new_tokens)
        kv_cache_mb = estimate_kv_cache_mb(batch_size=self.n_samples, seq_len=seq_len, **self._model_shape)
        vram_spike_mb = peak_vram_mb()
        total_out_tokens = sum(r.output_tokens for r in sample_results)
        throughput = total_out_tokens / elapsed_s if elapsed_s > 0 else 0.0

        return DetectionResult(
            is_hallucination=is_hallucination,
            score=1.0 - consistency,
            metrics=dict(
                consistency_score=consistency,
                n_samples=self.n_samples,
                sampling_latency_ms=elapsed_s * 1000,
                throughput_tps=throughput,
                kv_cache_mb=kv_cache_mb,
                peak_vram_mb=vram_spike_mb,
            ),
        )
