"""
Statistical significance testing for accuracy-vs-compute trade-offs
(Methodology III-D): paired t-tests and ANOVA across paradigms via SciPy.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import pandas as pd
from scipy import stats


def paired_ttests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired t-test of `metric` between every pair of paradigms, matched on
    prompt_id/batch_size/seq_len_bucket so each pair compares the same
    underlying workload."""
    pivot = df.pivot_table(index=["prompt_id", "batch_size", "seq_len_bucket"], columns="paradigm", values=metric)
    paradigms = list(pivot.columns)
    rows = []
    for a, b in combinations(paradigms, 2):
        paired = pivot[[a, b]].dropna()
        if len(paired) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(paired[a], paired[b])
        rows.append(dict(paradigm_a=a, paradigm_b=b, metric=metric, t_stat=t_stat, p_value=p_value, n=len(paired)))
    return pd.DataFrame(rows)


def one_way_anova(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One-way ANOVA of `metric` across all paradigms."""
    groups = [g[metric].dropna().values for _, g in df.groupby("paradigm")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return pd.DataFrame([dict(metric=metric, f_stat=None, p_value=None)])
    f_stat, p_value = stats.f_oneway(*groups)
    return pd.DataFrame([dict(metric=metric, f_stat=f_stat, p_value=p_value)])


def run_all_stats(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ttest_frames = [paired_ttests(df, m) for m in metrics]
    anova_frames = [one_way_anova(df, m) for m in metrics]
    ttests = pd.concat(ttest_frames, ignore_index=True) if ttest_frames else pd.DataFrame()
    anovas = pd.concat(anova_frames, ignore_index=True) if anova_frames else pd.DataFrame()
    return ttests, anovas
