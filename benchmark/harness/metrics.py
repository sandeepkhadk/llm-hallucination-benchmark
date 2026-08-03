"""
F1-score computation against ground-truth hallucination labels
(Methodology III-B, primary accuracy metric).

Determining whether a *generated* response is itself a hallucination
requires comparing it against the reference answer. We use a lexical-overlap
heuristic: the generation is labeled a hallucination when it is closer to
the dataset's `hallucinated_answer` than to its `reference_answer`. This is
a pragmatic stand-in for human annotation or a stronger NLI-based labeler.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _overlap(a: str, b: str) -> float:
    a_tokens = set(str(a).lower().split())
    b_tokens = set(str(b).lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def label_ground_truth(row: pd.Series) -> bool:
    gen = row.get("generated_text", "")
    ref = row.get("reference_answer", "")
    hallucinated = row.get("hallucinated_answer", "")
    return _overlap(gen, hallucinated) > _overlap(gen, ref)


def compute_accuracy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-paradigm F1 / precision / recall against ground-truth labels."""
    df = df.copy()
    if "ground_truth_hallucination" not in df.columns:
        df["ground_truth_hallucination"] = df.apply(label_ground_truth, axis=1)

    rows = []
    for paradigm, group in df.groupby("paradigm"):
        y_true = group["ground_truth_hallucination"]
        y_pred = group["predicted_hallucination"]
        rows.append(
            dict(
                paradigm=paradigm,
                f1=f1_score(y_true, y_pred, zero_division=0),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                n=len(group),
            )
        )
    return pd.DataFrame(rows)


# Per-paradigm columns that make up the *added* latency of running detection,
# on top of the shared baseline generation (which is identical across
# paradigms for a given prompt and thus useless for paradigm comparisons).
_OVERHEAD_COLUMNS = {
    "selfcheckgpt": ["metric_sampling_latency_ms"],
    "rag_verification": ["metric_embedding_latency_ms", "metric_retrieval_latency_ms"],
    "llm_as_judge": ["metric_latency_ms"],
    "llm_as_judge_strong": ["metric_latency_ms"],
}


def compute_detection_overhead(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `detection_overhead_ms` column: the paradigm-specific latency
    cost of hallucination detection, summed from each paradigm's own metric
    columns (Methodology III-D). Unlike `baseline_ttft_ms`/
    `baseline_throughput_tps`, this varies by paradigm and is what
    stats tests should actually compare."""
    df = df.copy()
    overhead = pd.Series(0.0, index=df.index)
    for paradigm, cols in _OVERHEAD_COLUMNS.items():
        mask = df["paradigm"] == paradigm
        available = [c for c in cols if c in df.columns]
        if available:
            overhead.loc[mask] = df.loc[mask, available].sum(axis=1)
    df["detection_overhead_ms"] = overhead
    return df

