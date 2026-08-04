"""
Pareto-frontier visualizations of accuracy-vs-compute trade-offs
(Methodology III-D): F1-score vs Throughput, via Matplotlib/Seaborn.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pareto_frontier(
    accuracy_df: pd.DataFrame,
    compute_df: pd.DataFrame,
    throughput_col: str,
    output_path: str,
    lower_is_better: bool = False,
) -> None:
    merged = accuracy_df.merge(compute_df, on="paradigm")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=merged, x=throughput_col, y="f1", hue="paradigm", s=140, ax=ax)
    for _, row in merged.iterrows():
        # Centered, straight-up offset so edge points (e.g. the cheapest/most
        # expensive paradigm on a log axis) don't drift their label sideways
        # into the legend or off the axes.
        ax.annotate(
            row["paradigm"],
            (row[throughput_col], row["f1"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    ax.set_xlabel(f"{throughput_col} ({'lower = cheaper' if lower_is_better else 'higher = faster'})")
    ax.set_ylabel("F1-score (higher = more accurate)")
    ax.set_title("Accuracy vs. Compute Pareto Frontier")
    if lower_is_better:
        ax.set_xscale("log")  # overhead can span multiple orders of magnitude across paradigms
        ax.invert_xaxis()
    # Extra left/right breathing room so edge-point labels never touch the axes border.
    ax.margins(x=0.2)
    # Legend placed outside the axes so it never overlaps a data point/label.
    ax.legend(title="paradigm", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    # Headroom above the topmost points keeps annotations from clipping at the axes edge.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.06 * (y_hi - y_lo))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_distribution(df: pd.DataFrame, metric: str, output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="paradigm", y=metric, ax=ax)
    ax.set_title(f"Distribution of {metric} by paradigm")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
