# Hallucination Detection Computational Benchmark

A controlled, reproducible benchmark that compares three LLM hallucination-detection
paradigms — **SelfCheckGPT**, **RAG Verification**, and **LLM-as-a-Judge** — not just on
detection accuracy (F1/precision/recall) but on their **computational cost**: latency
overhead, throughput, peak VRAM, KV-cache size, and vector-store memory. This repository
contains both the benchmarking harness (`benchmark/`) and the accompanying IEEE-conference
paper (`main.tex`) that reports the results of running it end-to-end: dataset construction →
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

## Repository layout

```
hallucination/
├── main.tex                     # IEEE-conference paper (Sections I-V): methodology, results, discussion
├── README.md                    # This file
├── .gitignore                   # Ignores __pycache__/, *.pyc, *.log, research_papers/, .venv/
│
├── benchmark/                   # The benchmarking harness (code). See benchmark/README.md for full details.
│   ├── main.py, config.py, requirements.txt, Hallucination_Benchmark_Colab.ipynb
│   ├── data/                    # Dataset construction (HaluEval + TruthfulQA)
│   ├── engine/                  # vLLM / transformers / mock inference backends + memory profiling
│   ├── detectors/               # SelfCheckGPT, RAG verification, LLM-as-a-judge implementations
│   ├── harness/                 # Data-collection loop + accuracy/overhead metrics
│   ├── analysis/                # Paired t-tests, ANOVA, Pareto/TTFT plots
│   └── README.md
│
├── results/                     # Checked-in output of the `--lite` (Qwen2.5-0.5B) run — main.tex Sections IV-A/B/C
│   ├── raw_results.csv, accuracy_metrics.csv, paired_ttests.csv, anova_results.csv
│   └── plots/                   # pareto_frontier.png, ttft_distribution.png
│
├── results_paper/               # Checked-in output of the `--paper` (Qwen2.5-1.5B) run — main.tex Section IV-D
│   ├── raw_results.csv, accuracy_metrics.csv, paired_ttests.csv, anova_results.csv
│   ├── strong_judge_results.csv, strong_judge_accuracy_metrics.csv   # --strong-judge (3B judge) ablation
│   └── plots/
│
├── research_paper_/              # Self-contained LaTeX build directory used to actually compile the paper
│   ├── main.tex                 # copy of the root main.tex, kept alongside the assets pdflatex/bibtex need
│   ├── references.bib
│   ├── hallucination.pdf        # last compiled output
│   ├── plots/                   # copies of results/plots/*.png (referenced via \includegraphics{plots/...})
│   └── plots_paper/             # copies of results_paper/plots/*.png (referenced via \includegraphics{plots_paper/...})
│
└── research_papers/              # Git-ignored: cited reference PDFs (FActScore, SelfCheckGPT, RAGAS, etc.), not tracked
    └── references/
```

`main.tex` at the repo root is the single source of truth for the paper's text and is what
should be edited going forward; because `pdflatex`/`bibtex` also need `references.bib` and
the `plots/`/`plots_paper/` image folders sitting next to `main.tex`, compiling means copying
the updated `main.tex` into `research_paper_/` (or copying `references.bib` and the plots
folders next to the root `main.tex`) before running the build.

See `benchmark/README.md` for the harness's internal module responsibilities and its own
(git-ignored) `results/` output layout when run standalone from inside `benchmark/`.

## Quick start (no downloads, sanity-check the pipeline)

```bash
cd benchmark
pip install pandas scipy scikit-learn matplotlib seaborn
python main.py --mock --total-prompts 40
```

This runs the full pipeline (dataset → detection → stats → plots) with a
deterministic `MockEngine` so you can verify the harness end-to-end before
committing to real model downloads.

## Full run

```bash
cd benchmark
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
cd benchmark
pip install -r requirements.txt   # torch + transformers + sentence-transformers + faiss are enough; vllm not required
python main.py --lite --total-prompts 40 --output-dir ../results
```

`--lite` forces `--backend transformers`, uses `Qwen/Qwen2.5-0.5B-Instruct` as
generator/judge and `BAAI/bge-small-en-v1.5` as the embedding model, and
reduces batch sizes / SelfCheckGPT samples for a CPU-friendly runtime. This is
the run whose checked-in output lives in the repo-root `results/` folder and
is reported in main.tex Sections IV-A through IV-C.

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
cd benchmark
pip install -r requirements.txt
python main.py --paper --total-prompts 100 --output-dir ../results_paper                 # 1.5B generator + judge, T4-friendly
python main.py --paper --strong-judge --total-prompts 100 --output-dir ../results_paper  # + a second judging pass with the 3B model
```

`--paper` forces `--backend transformers` and sets `batch_sizes=[1, 4]`. Note
that `run_benchmark.py` re-runs the full dataset once per batch size (it's an
independent variable, not a speed optimization), so `--paper`'s two batch
sizes roughly double the wall-clock time versus `--lite`'s single batch size —
budget accordingly (e.g. ~100-150 prompts fits comfortably in a few hours on a
free T4; 500 prompts does not, see "Limitations" below). This is the run whose
checked-in output lives in the repo-root `results_paper/` folder and is
reported in main.tex Section IV-D.

`--strong-judge` requires `--paper` (it uses `config.model.judge_model_strong`,
which is empty by default) and writes its own
`strong_judge_results.csv`/`strong_judge_accuracy_metrics.csv`, so you can
compare the cheap shared-weight judge against the stronger 3B judge on
identical generated text without regenerating anything.

## Running on Google Colab

`benchmark/Hallucination_Benchmark_Colab.ipynb` is a self-contained notebook that: (1)
uploads/clones the `benchmark/` folder onto a Colab GPU runtime, (2) installs
dependencies (without touching Colab's preinstalled GPU-matched `torch`), (3)
lets you pick `--mock` / `--lite` / `--paper` / `--paper --strong-judge`, and
(4) displays and zips up the results for download. Open it in Colab, set
`Runtime → Change runtime type → T4 GPU`, and run the cells top to bottom. The
`results.zip`/`results_paper.zip` it produces are what get extracted into the
repo-root `results/`/`results_paper/` folders checked into this repository.

## Reports and analysis

- `results/plots/pareto_frontier.png` / `results_paper/plots/pareto_frontier.png` — F1
  accuracy vs. `detection_overhead_ms` (log-scale x-axis, lower = cheaper) for all three
  paradigms; the Pareto-dominant paradigm sits toward the top-left.
- `results/plots/ttft_distribution.png` / `results_paper/plots/ttft_distribution.png` —
  sanity-check that baseline time-to-first-token is paradigm-independent (detection happens
  *after* generation, so this should look identical across paradigms).
- `results/Benchmark_Report.docx` (generated at runtime, git-ignored) — a Word report
  covering the models/config used, procedure, accuracy + statistical-significance tables,
  conclusions, a Big-O time/space complexity analysis per paradigm, and an "Experimental
  Scale and Limitations" section reconciling the theoretical complexity terms with the
  measured `detection_overhead_ms`/memory numbers.
- `main.tex` — the write-up of all of the above, including the `--strong-judge` 3B-judge
  ablation (Section IV-D) and the cross-scale Discussion/Recommendation (Section IV-E).

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
