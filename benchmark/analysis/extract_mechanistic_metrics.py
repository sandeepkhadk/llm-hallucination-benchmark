"""
extract_mechanistic_metrics.py

Post-hoc derivation of mechanistic per-paradigm metrics from an existing
benchmark results CSV, so Table VII / Section IV can report:
    - generator baseline output length (tokens)
    - RAG: n_claims, n_retrieved (top-k), corpus size V
    - LLM-as-a-Judge: judge prompt length (tokens), judge verdict length (tokens)
    - SelfCheckGPT: n_samples, mean tokens per additional sample
    - failure rate per paradigm

No model re-run required: everything is computed from already-saved columns
(generated_text, latency/throughput fields) plus one offline tokenization
pass and a reconstructed judge prompt template.

USAGE
    python extract_mechanistic_metrics.py results.csv \
        --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
        --judge-template judge_template.txt \
        --topk 3 \
        --out table_mechanistic.csv

If --tokenizer is omitted, token counts are approximated via a whitespace
split (rough, but keeps the script runnable without downloading a model).
If --judge-template is omitted, a placeholder template is used and you
should verify it matches the actual prompt your harness constructed
(Section III-E of the paper describes it as a single fixed verdict prompt
used verbatim across all 100 prompts).
"""

import argparse
import sys
import pandas as pd
import numpy as np


DEFAULT_JUDGE_TEMPLATE = (
    "You are evaluating whether the following response contains "
    "hallucinated (unsupported or fabricated) information.\n\n"
    "Response:\n{response}\n\n"
    "Does this response contain hallucinated content? Answer Yes or No, "
    "then briefly justify your answer."
)


def get_tokenizer(name):
    if name is None:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"[warn] could not load tokenizer '{name}' ({e}); "
              f"falling back to whitespace token counts.", file=sys.stderr)
        return None


def count_tokens(text, tokenizer):
    if not isinstance(text, str) or not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # fallback: whitespace split, rough proxy only
    return len(text.split())


def load_judge_template(path):
    if path is None:
        return DEFAULT_JUDGE_TEMPLATE
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="path to the raw results CSV")
    ap.add_argument("--tokenizer", default=None,
                     help="HF tokenizer name/path used by the generator/judge "
                          "(e.g. Qwen/Qwen2.5-0.5B-Instruct). If omitted, "
                          "falls back to whitespace-token approximation.")
    ap.add_argument("--judge-template", default=None,
                     help="path to a text file containing the exact judge "
                          "prompt template, with {response} as the "
                          "placeholder for the generator's output. If "
                          "omitted, a placeholder template is used -- "
                          "replace it with your real one before trusting "
                          "the judge_prompt_tokens numbers.")
    ap.add_argument("--topk", type=int, default=1,
                     help="the k used in the RAG FAISS search "
                          "(IndexFlatIP.search(..., k=?)). Check your RAG "
                          "verification code for the actual value passed; "
                          "this script cannot infer it from the CSV alone.")
    ap.add_argument("--out", default="table_mechanistic.csv",
                     help="output CSV path for the summary table")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    tokenizer = get_tokenizer(args.tokenizer)
    judge_template = load_judge_template(args.judge_template)

    # ---- 1. Generator baseline output length (tokens) -------------------
    # Prefer direct tokenization of the saved generated_text (more accurate
    # than the throughput*time estimate, which has rounding noise).
    if "generated_text" in df.columns:
        df["generator_output_tokens"] = df["generated_text"].apply(
            lambda t: count_tokens(t, tokenizer)
        )
    else:
        df["generator_output_tokens"] = (
            df["baseline_throughput_tps"] * (df["baseline_latency_ms"] / 1000.0)
        )

    # ---- 2. RAG: n_claims, n_retrieved, V --------------------------------
    is_rag = df["paradigm"] == "rag_verification"
    df.loc[is_rag, "n_retrieved_topk"] = args.topk  # constant per your k
    # metric_n_claims and metric_index_size are already logged columns

    # ---- 3. Judge: prompt tokens + verdict tokens ------------------------
    is_judge = df["paradigm"] == "llm_as_judge"
    if is_judge.any():
        df.loc[is_judge, "judge_prompt_tokens"] = df.loc[is_judge, "generated_text"].apply(
            lambda resp: count_tokens(judge_template.format(response=resp), tokenizer)
        )
    # metric_judge_output_tokens is already the verdict length column

    # ---- 4. SelfCheckGPT: n_samples + per-sample token estimate ---------
    is_scg = df["paradigm"] == "selfcheckgpt"
    if is_scg.any():
        total_sample_tokens = (
            df.loc[is_scg, "metric_throughput_tps"]
            * (df.loc[is_scg, "metric_sampling_latency_ms"] / 1000.0)
        )
        df.loc[is_scg, "selfcheck_sample_tokens_mean"] = (
            total_sample_tokens / df.loc[is_scg, "metric_n_samples"]
        )

    # ---- 5. Failure rate per paradigm ------------------------------------
    # A "failure" = the paradigm's core output field is missing/unparseable
    # for that row. Adjust the column checked per-paradigm to match your
    # harness's actual failure signal (e.g. malformed judge verdicts that
    # don't parse to yes/no, RAG rows with no claims extracted, etc.)
    def row_failed(row):
        p = row["paradigm"]
        if p == "selfcheckgpt":
            return pd.isna(row.get("metric_consistency_score"))
        if p == "rag_verification":
            return pd.isna(row.get("detection_score")) or pd.isna(row.get("metric_n_claims"))
        if p == "llm_as_judge":
            return pd.isna(row.get("predicted_hallucination")) or pd.isna(row.get("metric_raw_verdict"))
        return False

    df["failed"] = df.apply(row_failed, axis=1)

    # ---- Build summary table (mean +/- std per paradigm) -----------------
    summary_specs = {
        "selfcheckgpt": ["generator_output_tokens", "metric_n_samples",
                          "selfcheck_sample_tokens_mean", "detection_overhead_ms"],
        "rag_verification": ["generator_output_tokens", "metric_n_claims",
                              "n_retrieved_topk", "metric_index_size",
                              "metric_vector_store_mb", "detection_overhead_ms"],
        "llm_as_judge": ["generator_output_tokens", "judge_prompt_tokens",
                       "metric_judge_output_tokens", "detection_overhead_ms"],
    }

    rows = []
    for paradigm, cols in summary_specs.items():
        sub = df[df["paradigm"] == paradigm]
        if sub.empty:
            continue
        row = {"paradigm": paradigm, "n_rows": len(sub),
               "failure_rate": sub["failed"].mean()}
        for c in cols:
            if c in sub.columns:
                row[f"{c}_mean"] = sub[c].mean()
                row[f"{c}_std"] = sub[c].std()
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved full summary to {args.out}")

    # Also dump the row-level augmented CSV in case you want it
    augmented_path = args.out.replace(".csv", "_row_level.csv")
    df.to_csv(augmented_path, index=False)
    print(f"Saved row-level augmented CSV to {augmented_path}")


if __name__ == "__main__":
    main()