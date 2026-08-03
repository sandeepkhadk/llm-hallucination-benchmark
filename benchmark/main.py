"""
Entry point for the hallucination-detection computational benchmark
(implements Methodology Section III end-to-end).

Usage:
    python main.py --mock                  # fast, no downloads, sanity-checks the pipeline
    python main.py --backend vllm           # full run with vLLM + real models (needs a real GPU)
    python main.py --backend transformers   # CPU/GPU fallback via HF transformers
    python main.py --lite                  # real (non-mock) run with tiny CPU-friendly models,
                                            # for machines with ~8GB RAM and a 1-2GB GPU
    python main.py --paper                 # research-paper-scale run (1.5B shared generator/judge,
                                            # BGE-small embedder) sized for a free single-T4 Colab GPU
    python main.py --paper --strong-judge  # as above, plus a second judging pass with a 3B model,
                                            # loaded only after the primary engines are freed
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import Config
from data.prepare_dataset import build_dataset
from harness.run_benchmark import run_benchmark, save_results
from harness.metrics import compute_accuracy_metrics, compute_detection_overhead
from analysis.stats_analysis import run_all_stats
from analysis.visualize import plot_pareto_frontier, plot_metric_distribution


def build_engines(config: Config, backend: str, mock: bool):
    if mock:
        from engine.mock_engine import MockEngine

        generator = MockEngine(seed=config.dataset.seed)
        judge = generator
        return generator, judge

    # If the judge is configured to be the same model as the generator, reuse
    # the already-loaded weights instead of loading a second copy (0 extra VRAM).
    shares_weights = config.model.judge_model_lightweight == config.model.generator_model

    if backend == "vllm":
        from engine.inference_engine import VLLMEngine

        generator = VLLMEngine(config.model.generator_model)
        judge = generator if shares_weights else VLLMEngine(config.model.judge_model_lightweight)
    else:
        from engine.transformers_engine import TransformersEngine

        generator = TransformersEngine(config.model.generator_model)
        judge = generator if shares_weights else TransformersEngine(config.model.judge_model_lightweight)
    return generator, judge


def build_detectors(config: Config, generator_engine, judge_engine, corpus, mock: bool):
    from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
    from detectors.llm_judge_detector import LLMAsJudgeDetector

    if mock:
        from detectors.mock_rag_detector import MockRAGVerificationDetector

        rag_detector = MockRAGVerificationDetector(corpus=corpus)
    else:
        from detectors.rag_verification_detector import RAGVerificationDetector

        rag_detector = RAGVerificationDetector(config.model.embedding_model, corpus=corpus)

    return {
        "selfcheckgpt": SelfCheckGPTDetector(generator_engine, n_samples=config.experiment.n_selfcheck_samples),
        "rag_verification": rag_detector,
        "llm_as_judge": LLMAsJudgeDetector(judge_engine),
    }


def apply_lite_overrides(config: Config) -> None:
    """Swap in tiny CPU-friendly models/settings for low-RAM/low-VRAM machines (~8GB RAM, 1-2GB GPU)."""
    config.model.generator_model = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.judge_model_same_size = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.judge_model_lightweight = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.embedding_model = "BAAI/bge-small-en-v1.5"
    config.experiment.batch_sizes = [1]
    config.experiment.n_selfcheck_samples = 3
    config.experiment.n_warmup_runs = 1
    config.experiment.fp16 = False


def apply_paper_overrides(config: Config) -> None:
    """Research-paper-scale preset sized for a free-tier single T4 (16GB):
    generator and primary judge share one 1.5B model (weights loaded once, 0
    extra VRAM); an optional stronger 3B judge can be run afterwards via
    --strong-judge, loaded only once the primary engines are freed."""
    config.model.generator_model = "Qwen/Qwen2.5-1.5B-Instruct"
    config.model.judge_model_same_size = "Qwen/Qwen2.5-1.5B-Instruct"
    config.model.judge_model_lightweight = "Qwen/Qwen2.5-1.5B-Instruct"
    config.model.judge_model_strong = "Qwen/Qwen2.5-3B-Instruct"
    config.model.embedding_model = "BAAI/bge-small-en-v1.5"
    config.experiment.batch_sizes = [1, 4]
    config.experiment.n_selfcheck_samples = 5
    config.experiment.n_warmup_runs = 3
    config.experiment.fp16 = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--lite", action="store_true", help="real run with tiny models, forces --backend transformers")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="research-paper-scale preset (1.5B generator/judge, sharing weights) sized for a free single-T4 GPU, forces --backend transformers",
    )
    parser.add_argument(
        "--strong-judge",
        action="store_true",
        help="after the main run, free the primary engines and re-judge with model.judge_model_strong (requires --paper)",
    )
    parser.add_argument("--total-prompts", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    config = Config()
    config.output_dir = args.output_dir

    if args.lite and not args.mock:
        apply_lite_overrides(config)
        args.backend = "transformers"
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--lite: using {config.model.generator_model} / {config.model.embedding_model} on {device}")

    if args.paper and not args.mock:
        apply_paper_overrides(config)
        args.backend = "transformers"
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--paper: using {config.model.generator_model} / {config.model.embedding_model} on {device}")

    if args.strong_judge and not config.model.judge_model_strong:
        raise SystemExit("--strong-judge requires model.judge_model_strong to be set (use --paper)")

    dataset = build_dataset(
        total=args.total_prompts,
        truthfulqa_fraction=config.dataset.truthfulqa_fraction,
        seed=config.dataset.seed,
        mock=args.mock,
    )
    corpus = [item["reference_answer"] for item in dataset if item.get("reference_answer")]

    generator_engine, judge_engine = build_engines(config, args.backend, args.mock)
    detectors = build_detectors(config, generator_engine, judge_engine, corpus, mock=args.mock)

    df = run_benchmark(dataset, generator_engine, detectors, config)
    df = compute_detection_overhead(df)
    out_path = save_results(df, config.output_dir)
    print(f"Raw results saved to {out_path}")

    accuracy_df = compute_accuracy_metrics(df)
    accuracy_path = f"{config.output_dir}/accuracy_metrics.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"Accuracy metrics saved to {accuracy_path}")

    throughput_col = "metric_throughput_tps" if "metric_throughput_tps" in df.columns else "baseline_throughput_tps"
    compute_df = df.groupby("paradigm").agg(
        mean_throughput_tps=(throughput_col, "mean"),
        mean_detection_overhead_ms=("detection_overhead_ms", "mean"),
    ).reset_index()

    ttests, anovas = run_all_stats(df, metrics=["baseline_ttft_ms", "baseline_throughput_tps", "detection_overhead_ms"])
    ttests.to_csv(f"{config.output_dir}/paired_ttests.csv", index=False)
    anovas.to_csv(f"{config.output_dir}/anova_results.csv", index=False)

    # detection_overhead_ms is unit-consistent (ms) across all paradigms, unlike
    # throughput_tps (tokens/sec for generation-based detectors vs. claims/sec for RAG).
    plot_pareto_frontier(
        accuracy_df,
        compute_df,
        throughput_col="mean_detection_overhead_ms",
        output_path=f"{config.output_dir}/plots/pareto_frontier.png",
        lower_is_better=True,
    )
    plot_metric_distribution(df, metric="baseline_ttft_ms", output_path=f"{config.output_dir}/plots/ttft_distribution.png")

    if args.strong_judge and not args.mock:
        run_strong_judge_pass(config, dataset, df)

    print("Benchmark complete.")


def run_strong_judge_pass(config: Config, dataset, df) -> None:
    """Re-judge the already-generated text with a stronger, sequentially-loaded
    judge model (config.model.judge_model_strong). Reuses the text generated by
    the primary pass, so the (now-freed) generator/primary-judge weights don't
    need to stay resident while the stronger model is loaded."""
    import gc

    import torch

    from detectors.llm_judge_detector import LLMAsJudgeDetector
    from engine.transformers_engine import TransformersEngine

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"--strong-judge: loading {config.model.judge_model_strong}...")
    strong_engine = TransformersEngine(config.model.judge_model_strong)
    strong_detector = LLMAsJudgeDetector(strong_engine)

    prompt_by_id = {item["id"]: item["prompt"] for item in dataset}
    judge_rows = df[df["paradigm"] == "llm_as_judge"]

    rows = []
    for _, row in judge_rows.iterrows():
        detection = strong_detector.detect(
            prompt=prompt_by_id[row["prompt_id"]], generated_text=row["generated_text"]
        )
        new_row = row.to_dict()
        new_row["paradigm"] = "llm_as_judge_strong"
        new_row["predicted_hallucination"] = detection.is_hallucination
        new_row["detection_score"] = detection.score
        for k, v in detection.metrics.items():
            new_row[f"metric_{k}"] = v
        rows.append(new_row)

    strong_df = pd.DataFrame(rows)
    strong_df = compute_detection_overhead(strong_df)
    strong_path = f"{config.output_dir}/strong_judge_results.csv"
    strong_df.to_csv(strong_path, index=False)
    print(f"Strong-judge results saved to {strong_path}")

    strong_accuracy_df = compute_accuracy_metrics(strong_df)
    strong_accuracy_path = f"{config.output_dir}/strong_judge_accuracy_metrics.csv"
    strong_accuracy_df.to_csv(strong_accuracy_path, index=False)
    print(f"Strong-judge accuracy metrics saved to {strong_accuracy_path}")


if __name__ == "__main__":
    main()
