"""
Main benchmarking harness implementing Methodology Section III-C (Data
Collection Procedure).

For each sampled prompt and each (batch_size, paradigm) combination:
  1) Baseline generation (records TTFT + throughput)
  2) Detection execution for the paradigm under test
  3) Metric logging (peak VRAM, KV-cache, latency averaged over warm-up runs)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import Config
from detectors.base import HallucinationDetector
from engine.inference_engine import InferenceEngine
from engine.memory_profiler import peak_vram_mb, reset_peak_vram


def _bucket_for_length(n_tokens: int, buckets: Dict[str, Tuple[int, int]]) -> str:
    for name, (lo, hi) in buckets.items():
        if lo <= n_tokens < hi:
            return name
    return "long"


def run_benchmark(
    dataset: List[Dict],
    generator_engine: InferenceEngine,
    detectors: Dict[str, HallucinationDetector],
    config: Config,
) -> pd.DataFrame:
    rows = []

    # Warm-up: mitigate cold-start GPU initialization effects. Intentionally
    # left unseeded -- warm-up outputs are discarded and are not part of the
    # measured, reproducible results (Methodology III-C).
    if dataset:
        generator_engine.warmup(dataset[0]["prompt"], config.experiment.n_warmup_runs)

    start_time = time.perf_counter()
    total_prompts = len(dataset)
    base_seed = config.experiment.seed

    for batch_size in config.experiment.batch_sizes:
        for start_idx in range(0, len(dataset), batch_size):
            batch = dataset[start_idx : start_idx + batch_size]
            prompts = [item["prompt"] for item in batch]

            done = start_idx + len(batch)
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / done) * (total_prompts - done) if done else 0
            print(
                f"[run_benchmark] batch_size={batch_size} prompt {done}/{total_prompts} "
                f"| elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min",
                flush=True,
            )

            # 1) Baseline generation. Prompt i (its absolute position in
            # `dataset`, independent of batch_size) is seeded base_seed + i,
            # applied per-prompt rather than once globally, so the same
            # prompt is generated from the same random state at every model
            # scale (--lite / --paper), keeping the cross-scale F1
            # comparison clean of seed drift.
            reset_peak_vram()
            t0 = time.perf_counter()
            baseline_seeds = [[base_seed + start_idx + j] for j in range(len(batch))]
            baseline_results = generator_engine.generate(
                prompts, max_new_tokens=300, n=1, seeds=baseline_seeds
            )
            time.perf_counter() - t0
            baseline_vram_mb = peak_vram_mb()

            for local_idx, (item, gen_list) in enumerate(zip(batch, baseline_results)):
                gen = gen_list[0]
                bucket = _bucket_for_length(gen.output_tokens, config.experiment.sequence_length_buckets)
                prompt_index = start_idx + local_idx
                base_row = dict(
                    prompt_id=item["id"],
                    source=item["source"],
                    hallucination_label=item["hallucination_label"],
                    reference_answer=item.get("reference_answer", ""),
                    hallucinated_answer=item.get("hallucinated_answer", ""),
                    generated_text=gen.text,
                    batch_size=batch_size,
                    seq_len_bucket=bucket,
                    seed=gen.seed,
                    baseline_ttft_ms=gen.ttft_ms,
                    baseline_latency_ms=gen.latency_ms,
                    baseline_throughput_tps=gen.throughput_tps,
                    baseline_peak_vram_mb=baseline_vram_mb,
                )

                # 2) Detection execution, per paradigm
                for paradigm_name, detector in detectors.items():
                    detection = detector.detect(
                        prompt=item["prompt"], generated_text=gen.text, prompt_index=prompt_index
                    )
                    row = dict(base_row)
                    row["paradigm"] = paradigm_name
                    row["predicted_hallucination"] = detection.is_hallucination
                    row["detection_score"] = detection.score
                    for k, v in detection.metrics.items():
                        row[f"metric_{k}"] = v
                    rows.append(row)

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_results.csv"
    df.to_csv(out_path, index=False)
    return out_path
