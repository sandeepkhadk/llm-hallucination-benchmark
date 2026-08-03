# Hallucination Detection Computational Benchmark

A controlled, reproducible benchmark that compares three LLM hallucination-detection
paradigms — **SelfCheckGPT**, **RAG Verification**, and **LLM-as-a-Judge** — not just on
detection accuracy (F1/precision/recall) but on their **computational cost**: latency
overhead, throughput, peak VRAM, KV-cache size, and vector-store memory. It implements the
Methodology section (Section III) of `hallucination.pdf`, end-to-end: dataset construction →
baseline generation → per-paradigm detection → statistical significance testing (paired
t-tests, ANOVA) → Pareto-frontier / distribution visualizations → a generated Word report.

## Why this exists

Most hallucination-detection comparisons report accuracy only. In production, the paradigm
you can afford to run at all depends on its **time and space complexity**:
SelfCheckGPT is `O(N·L)` (N extra generations of length L), RAG Verification is `O(C·V)`
(claims × corpus size, dominated by embedding + vector search), and LLM-as-a-Judge is
`O(P + L_judge)` (judge prompt + output length). This benchmark measures all three axes on
the same dataset/hardware so the accuracy-vs-cost trade-off (see `results/plots/pareto_frontier.png`)
is empirically grounded rather than assumed.

## Folder structure

```
benchmark/
├── main.py                                # CLI entry point: wires config → data → engines → detectors → harness → analysis
├── config.py                              # ModelConfig / DatasetConfig / ExperimentConfig (independent variables)
├── requirements.txt                       # Full dependency set (vLLM, torch, transformers, sentence-transformers, faiss, ...)
├── README.md
├── Hallucination_Benchmark_Colab.ipynb    # One-click Google Colab runner (upload/clone → install → run → download results)
├── .gitignore                             # Ignores __pycache__/, results/, *.log
│
├── data/
│   ├── __init__.py
│   └── prepare_dataset.py                 # Builds the stratified prompt sample (HaluEval + TruthfulQA),
│                                           # falls back to synthetic mock data per-source if downloads fail
│
├── engine/
│   ├── __init__.py
│   ├── inference_engine.py                # GenerationResult dataclass + InferenceEngine base + VLLMEngine (primary backend)
│   ├── transformers_engine.py             # TransformersEngine — CPU/GPU-portable fallback backend
│   ├── mock_engine.py                     # MockEngine — dependency-free fake timings for pipeline sanity-checks
│   └── memory_profiler.py                 # KV-cache / vector-store / peak-VRAM measurement utilities
│
├── detectors/
│   ├── __init__.py
│   ├── base.py                            # DetectionResult dataclass + HallucinationDetector abstract base
│   ├── selfcheckgpt_detector.py            # SelfCheckGPT paradigm (N-sample consistency check)
│   ├── rag_verification_detector.py        # RAG Verification paradigm (BGE embeddings + FAISS retrieval)
│   ├── mock_rag_detector.py                # MockRAGVerificationDetector — pure-Python stand-in used only in --mock mode
│   └── llm_judge_detector.py               # LLM-as-a-Judge paradigm (judge model scores the generated answer)
│
├── harness/
│   ├── __init__.py
│   ├── run_benchmark.py                   # Main data-collection loop (baseline generation → per-paradigm detection)
│   └── metrics.py                         # Ground-truth labeling, F1/precision/recall, and detection_overhead_ms
│
├── analysis/
│   ├── __init__.py
│   ├── stats_analysis.py                  # Paired t-tests + ANOVA across paradigms (Methodology III-D)
│   └── visualize.py                       # Pareto frontier (F1 vs cost) + metric-distribution plots
│
└── results/                                # Generated at runtime (git-ignored), created by `main.py`
    ├── raw_results.csv                     # One row per (prompt, paradigm, batch_size, seq_len_bucket) run
    ├── accuracy_metrics.csv                # F1/precision/recall per paradigm
    ├── strong_judge_accuracy_metrics.csv   # Same, for the optional --strong-judge rerun
    ├── strong_judge_results.csv            # Raw rows for the --strong-judge rerun
    ├── paired_ttests.csv                   # Pairwise significance tests between paradigms
    ├── anova_results.csv                   # One-way ANOVA across all three paradigms
    ├── Benchmark_Report.docx               # Generated report: methodology, results, conclusions, complexity analysis
    └── plots/
        ├── pareto_frontier.png             # F1 vs detection_overhead_ms trade-off (log-scale, lower=cheaper)
        └── ttft_distribution.png           # TTFT distribution boxplot per paradigm
```

## Module responsibilities

- `config.py` — models, dataset, and experiment configuration (independent
  variables: paradigm, output length bucket, batch size). Key classes:
  `ModelConfig` (generator/judge/embedding/strong-judge model names),
  `DatasetConfig` (total prompts, 50/50 HaluEval/TruthfulQA split, seed=42),
  `ExperimentConfig` (paradigms list, sequence-length buckets, batch sizes,
  N=5 SelfCheckGPT samples, 5 warm-up runs).
- `data/prepare_dataset.py` — builds the stratified prompt sample from
  HaluEval + TruthfulQA (falls back to synthetic mock data if downloads fail).
- `engine/` — inference backends (`VLLMEngine` primary, `TransformersEngine`
  fallback, `MockEngine` for dependency-free pipeline testing) and
  `memory_profiler.py` (VRAM / KV-cache / vector-store memory measurement).
- `detectors/` — the three detection paradigms, each returning a
  `DetectionResult` plus the metrics specified in Methodology III-C.
  `mock_rag_detector.py` mirrors `RAGVerificationDetector`'s output shape
  with pure-Python Jaccard similarity so `--mock` runs need no model downloads.
- `harness/run_benchmark.py` — the main data-collection loop (baseline
  generation → detection execution → metric logging), and `metrics.py` for
  F1/precision/recall against ground-truth hallucination labels plus the
  unit-consistent `detection_overhead_ms` metric used for cost comparisons.
- `analysis/` — paired t-tests + ANOVA (`stats_analysis.py`) and Pareto
  frontier / distribution plots (`visualize.py`).
- `main.py` — end-to-end entry point; `build_engines()` selects the
  generator/judge backend (reusing one engine for both when they share the
  same model, so no extra VRAM is used), `build_detectors()` selects real vs.
  mock detectors based on the `--mock` flag.

## Quick start (no downloads, sanity-check the pipeline)

```bash
pip install pandas scipy scikit-learn matplotlib seaborn
python main.py --mock --total-prompts 40
```

This runs the full pipeline (dataset → detection → stats → plots) with a
deterministic `MockEngine` so you can verify the harness end-to-end before
committing to real model downloads.

## Full run

```bash
pip install -r requirements.txt
python main.py --backend vllm --total-prompts 500
```

Requires a CUDA GPU for `vllm`/`torch`. Use `--backend transformers` for a
CPU/GPU-portable (but slower, less methodology-faithful) alternative.

## Low-resource real run (`--lite`)

If you only have ~8GB RAM and a 1-2GB GPU (or no GPU), `--backend vllm` and the
paper's 8B-parameter models are not feasible — vLLM itself needs a real CUDA
GPU with several GB of free VRAM just for the 8B model's weights and KV cache.
`--lite` swaps in tiny CPU-friendly models instead, so you can still get a
*real* (non-mock) end-to-end run:

```bash
pip install -r requirements.txt   # torch + transformers + sentence-transformers + faiss are enough; vllm not required
python main.py --lite --total-prompts 40
```

`--lite` forces `--backend transformers`, uses `Qwen/Qwen2.5-0.5B-Instruct` as
generator/judge and `BAAI/bge-small-en-v1.5` as the embedding model, and
reduces batch sizes / SelfCheckGPT samples for a CPU-friendly runtime. Results
will not match the paper's reported accuracy/throughput numbers (much smaller
models) but exercise the same real code paths as the full run.

## Research-paper-scale run (`--paper`), sized for a free single-T4 GPU

A middle ground between `--lite` and the full 8B `--backend vllm` run —
chosen so a free Google Colab T4 (16GB VRAM) can run real, non-mock models
without OOM-ing:

| Role | Model | ~VRAM (fp16) | Notes |
|---|---|---|---|
| Generator | `Qwen2.5-1.5B-Instruct` | ~3 GB | Generates answers |
| Judge (primary) | Same 1.5B model (weights shared) | +0 GB | Reuses the already-loaded generator weights — no separate model load |
| Judge (optional, stronger) | `Qwen2.5-3B-Instruct` | ~6 GB | Loaded only after the primary engines are freed, so it never coexists with the generator in VRAM |
| Embedder (RAG) | `BGE-small-en-v1.5` | ~130 MB | Lightweight retrieval embeddings |

```bash
pip install -r requirements.txt
python main.py --paper --total-prompts 100                 # 1.5B generator + judge, T4-friendly
python main.py --paper --strong-judge --total-prompts 100  # + a second judging pass with the 3B model
```

`--paper` forces `--backend transformers` and sets `batch_sizes=[1, 4]`. Note
that `run_benchmark.py` re-runs the full dataset once per batch size (it's an
independent variable, not a speed optimization), so `--paper`'s two batch
sizes roughly double the wall-clock time versus `--lite`'s single batch size —
budget accordingly (e.g. ~100-150 prompts fits comfortably in a few hours on a
free T4; 500 prompts does not, see "Limitations" below).

`--strong-judge` requires `--paper` (it uses `config.model.judge_model_strong`,
which is empty by default) and writes its own
`strong_judge_results.csv`/`strong_judge_accuracy_metrics.csv`, so you can
compare the cheap shared-weight judge against the stronger 3B judge on
identical generated text without regenerating anything.

## Running on Google Colab

`Hallucination_Benchmark_Colab.ipynb` is a self-contained notebook that: (1)
uploads/clones this `benchmark/` folder onto a Colab GPU runtime, (2) installs
dependencies (without touching Colab's preinstalled GPU-matched `torch`), (3)
lets you pick `--mock` / `--lite` / `--paper` / `--paper --strong-judge`, and
(4) displays and zips up the results for download. Open it in Colab, set
`Runtime → Change runtime type → T4 GPU`, and run the cells top to bottom.

Results are written to `results/`: `raw_results.csv`, `accuracy_metrics.csv`,
`strong_judge_results.csv` / `strong_judge_accuracy_metrics.csv` (only with
`--strong-judge`), `paired_ttests.csv`, `anova_results.csv`, and `plots/`.

## Reports and analysis

- `results/plots/pareto_frontier.png` — F1 accuracy vs. `detection_overhead_ms`
  (log-scale x-axis, lower = cheaper) for all three paradigms; the
  Pareto-dominant paradigm sits toward the top-left.
- `results/plots/ttft_distribution.png` — sanity-check that baseline
  time-to-first-token is paradigm-independent (detection happens *after*
  generation, so this should look identical across paradigms).
- `results/Benchmark_Report.docx` — a generated Word report covering the
  models/config used, procedure, accuracy + statistical-significance tables,
  conclusions, a Big-O time/space complexity analysis per paradigm, and an
  "Experimental Scale and Limitations" section reconciling the theoretical
  complexity terms with the measured `detection_overhead_ms`/memory numbers.

## Limitations

The full methodology specifies 8B-parameter models over 500 prompts across
three batch sizes — infeasible on free-tier compute (a single 16GB GPU with no
guaranteed multi-hour session). `--lite` and `--paper` substitute smaller
open models to keep the pipeline runnable end-to-end; this changes the
*absolute* latency/memory numbers but not the *asymptotic* relationships
(`O(N·L)`, `O(C·V)`, `O(P + L_judge)`) or the expected relative ordering of
paradigms by cost. For production-scale absolute figures, re-run with the
default `config.py` models on a GPU with enough VRAM for an 8B model plus
concurrent judge/embedding models (e.g. a single A100 40GB).
