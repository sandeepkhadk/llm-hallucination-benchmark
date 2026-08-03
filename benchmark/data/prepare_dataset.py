"""
Builds the 500-prompt stratified evaluation set from HaluEval and TruthfulQA
(Methodology Section III-B, "Models and Dataset").

Real datasets are pulled from the HuggingFace Hub. If a dataset cannot be
downloaded (no internet access, gated repo, renamed config, etc.) the loader
falls back to synthetic mock data so the rest of the pipeline can still be
exercised end-to-end via `--mock`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

try:
    from datasets import load_dataset

    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

# Known HuggingFace Hub locations for HaluEval; tried in order since the
# community has published it under a couple of different repo/config names.
# `notrichardren/HaluEval`'s "qa" config carries `right_answer` +
# `hallucinated_answer` together per row (what we need); `pminervini/HaluEval`'s
# "qa_samples" only has a single `answer` tagged yes/no per row, so it's a
# lower-fidelity fallback (see the field mapping below). Each repo also uses a
# different split name on the Hub, so it's tracked alongside repo/config.
HALUEVAL_CANDIDATES = [
    ("notrichardren/HaluEval", "qa", "train"),
    ("pminervini/HaluEval", "qa_samples", "data"),
]
# "truthful_qa" (no namespace) was deprecated on the HF Hub in favor of the
# namespaced repo id below; kept as a fallback in case the namespace changes again.
TRUTHFULQA_CANDIDATES = ["truthfulqa/truthful_qa", "truthful_qa"]
TRUTHFULQA_CONFIG = "generation"


def _tag_intrinsic_extrinsic(source: str) -> str:
    """Per Huang et al. [1]: intrinsic hallucinations contradict the given
    input/knowledge, extrinsic ones add ungrounded information. HaluEval's
    hallucinated samples are constructed against a supplied knowledge
    snippet (intrinsic); TruthfulQA's incorrect answers are common,
    ungrounded misconceptions (extrinsic)."""
    return "intrinsic" if source == "halueval" else "extrinsic"


def _load_halueval(n: int, seed: int) -> List[Dict]:
    if not _HAS_DATASETS:
        raise RuntimeError("`datasets` package not installed")

    ds, last_err = None, None
    for repo, cfg, split in HALUEVAL_CANDIDATES:
        try:
            ds = load_dataset(repo, cfg, split=split)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load HaluEval from any known repo/config: {last_err}")

    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for i, r in enumerate(rows):
        question = r.get("question") or r.get("knowledge") or r.get("dialogue_history") or ""
        # `notrichardren/HaluEval` ships both answers per row directly. The
        # `pminervini/HaluEval` fallback only has one `answer` tagged yes/no,
        # so only one of reference/hallucinated_answer will be populated there.
        if "hallucination" in r and "answer" in r:
            is_hallucinated = str(r.get("hallucination", "")).strip().lower() == "yes"
            hallucinated = r.get("answer", "") if is_hallucinated else ""
            right = r.get("answer", "") if not is_hallucinated else ""
        else:
            hallucinated = r.get("hallucinated_answer") or r.get("hallucinated_summary") or ""
            right = r.get("right_answer") or ""
        out.append(
            {
                "id": f"haluevalqa-{i}",
                "source": "halueval",
                "prompt": question,
                "reference_answer": right,
                "hallucinated_answer": hallucinated,
                "hallucination_label": _tag_intrinsic_extrinsic("halueval"),
            }
        )
    return out


def _load_truthfulqa(n: int, seed: int) -> List[Dict]:
    if not _HAS_DATASETS:
        raise RuntimeError("`datasets` package not installed")

    ds, last_err = None, None
    for repo in TRUTHFULQA_CANDIDATES:
        try:
            ds = load_dataset(repo, TRUTHFULQA_CONFIG, split="validation")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load TruthfulQA from any known repo: {last_err}")

    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for i, r in enumerate(rows):
        best_answer = r.get("best_answer") or (r.get("correct_answers") or [""])[0]
        incorrect = (r.get("incorrect_answers") or [""])[0]
        out.append(
            {
                "id": f"truthfulqa-{i}",
                "source": "truthfulqa",
                "prompt": r["question"],
                "reference_answer": best_answer,
                "hallucinated_answer": incorrect,
                "hallucination_label": _tag_intrinsic_extrinsic("truthfulqa"),
            }
        )
    return out


def _mock_dataset(n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        source = "halueval" if i % 2 == 0 else "truthfulqa"
        out.append(
            {
                "id": f"mock-{i}",
                "source": source,
                "prompt": f"[MOCK] Explain fact #{i} about topic {rng.randint(1, 50)}.",
                "reference_answer": f"[MOCK reference answer #{i}]",
                "hallucinated_answer": f"[MOCK hallucinated answer #{i}]",
                "hallucination_label": _tag_intrinsic_extrinsic(source),
            }
        )
    return out


def build_dataset(total: int, truthfulqa_fraction: float, seed: int, mock: bool = False) -> List[Dict]:
    """Stratified sample of `total` prompts split between HaluEval and TruthfulQA."""
    if mock:
        return _mock_dataset(total, seed)

    n_truthfulqa = round(total * truthfulqa_fraction)
    n_halueval = total - n_truthfulqa

    try:
        haluqa = _load_halueval(n_halueval, seed)
    except Exception as e:  # noqa: BLE001
        print(f"[prepare_dataset] Falling back to mock HaluEval data: {e}")
        haluqa = _mock_dataset(n_halueval, seed)

    try:
        tqa = _load_truthfulqa(n_truthfulqa, seed)
    except Exception as e:  # noqa: BLE001
        print(f"[prepare_dataset] Falling back to mock TruthfulQA data: {e}")
        tqa = _mock_dataset(n_truthfulqa, seed + 1)

    combined = haluqa + tqa
    random.Random(seed).shuffle(combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--truthfulqa-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Use synthetic data, no downloads required")
    parser.add_argument("--out", type=str, default="results/dataset.jsonl")
    args = parser.parse_args()

    rows = build_dataset(args.total, args.truthfulqa_fraction, args.seed, mock=args.mock)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()
