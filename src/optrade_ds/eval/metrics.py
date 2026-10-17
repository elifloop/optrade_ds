"""Simple cross-sectional evaluation: forward-return rank information coefficient (IC)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ra, rb = a.rank(), b.rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_ic(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "fwd_ret_5",
    min_symbols: int = 3,
) -> pd.Series:
    """Per-date Spearman correlation between scores and forward returns."""
    data = df.dropna(subset=[score_col, target_col])
    out = {}
    for date, g in data.groupby("date"):
        if len(g) < min_symbols:
            continue
        ic = _spearman(g[score_col], g[target_col])
        if not np.isnan(ic):
            out[date] = ic
    return pd.Series(out, dtype=float, name="rank_ic")


def evaluate_scores(
    df: pd.DataFrame, score_col: str = "score", target_col: str = "fwd_ret_5"
) -> dict:
    """Summary metrics: mean/std rank IC, IC information ratio, hit rate, number of dates."""
    ic = rank_ic(df, score_col, target_col)
    n = int(ic.size)
    if n == 0:
        return {"rank_ic_mean": None, "rank_ic_std": None, "ic_ir": None,
                "ic_hit_rate": None, "n_dates": 0, "target": target_col}
    std = float(ic.std(ddof=1)) if n > 1 else 0.0
    return {
        "rank_ic_mean": round(float(ic.mean()), 6),
        "rank_ic_std": round(std, 6),
        "ic_ir": round(float(ic.mean()) / std, 6) if std > 0 else None,
        "ic_hit_rate": round(float((ic > 0).mean()), 6),
        "n_dates": n,
        "target": target_col,
    }
