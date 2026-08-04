"""
Configuration for the hallucination-detection computational benchmark.

Mirrors the independent/dependent variables and models/dataset defined in
Methodology Section III-B (Research Design).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ModelConfig:
    generator_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    judge_model_same_size: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    judge_model_lightweight: str = "meta-llama/Llama-3.2-1B-Instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # Optional stronger judge, loaded sequentially (after the generator is freed)
    # for a second, higher-quality judging pass. Empty means unused.
    judge_model_strong: str = ""


@dataclass
class DatasetConfig:
    total_prompts: int = 500
    truthfulqa_fraction: float = 0.5  # stratified split between HaluEval / TruthfulQA
    seed: int = 42


@dataclass
class ExperimentConfig:
    paradigms: List[str] = field(
        default_factory=lambda: ["selfcheckgpt", "rag_verification", "llm_as_judge"]
    )
    # Independent variable: output sequence length bucket (in tokens).
    sequence_length_buckets: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"short": (0, 100), "medium": (100, 300), "long": (300, 10_000)}
    )
    # Independent variable: batch size.
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    n_selfcheck_samples: int = 5  # N in SelfCheckGPT's O(N*L) time complexity
    n_warmup_runs: int = 5
    fp16: bool = True
    # Base seed for reproducible, per-prompt generation seeding (default
    # matches dataset.seed). Prompt i is seeded base_seed + i for baseline
    # generation, and base_seed + i*100 + sample_idx for SelfCheckGPT's N
    # samples, so the same prompt is generated from the same random state
    # across model scales (--lite / --paper).
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output_dir: str = "results"
