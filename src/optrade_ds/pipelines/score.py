"""Score pipeline: latest artifact -> latest features per symbol -> recommendations rows."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from optrade_ds.data.writer import RECOMMENDATION_COLUMNS, RecommendationWriter
from optrade_ds.features.technical import compute_features, latest_features
from optrade_ds.models.base import latest_artifact_path
from optrade_ds.models.baseline import BaselineMomentumRanker

log = logging.getLogger(__name__)


def build_rationale(row: pd.Series) -> str:
    """Human readable summary, e.g. ``20d momentum +4.2%, 60d momentum +9.8%, vol 28% ann., ...``."""
    return (
        f"20d momentum {row['mom_20'] * 100:+.1f}%, "
        f"60d momentum {row['mom_60'] * 100:+.1f}%, "
        f"5d {row['mom_5'] * 100:+.1f}%, "
        f"vol {row['vol_20'] * 100:.0f}% ann., "
        f"RSI {row['rsi_14']:.0f}, "
        f"{abs(row['trend_50']) * 100:.1f}% {'above' if row['trend_50'] >= 0 else 'below'} 50d avg"
    )


def score_latest(prices: pd.DataFrame, model: BaselineMomentumRanker) -> pd.DataFrame:
    """Return one recommendation row per scorable symbol, sorted by score desc.

    Columns: symbol, score, rationale, model_version, as_of (= max price date in ``prices``).
    """
    if prices.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    as_of = pd.Timestamp(prices["date"].max()).date()
    latest = latest_features(compute_features(prices))
    if latest.empty:
        log.warning("no symbols with enough history to score")
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    out = pd.DataFrame(
        {
            "symbol": latest["symbol"].values,
            "score": model.score(latest).round(6).values,
            "rationale": latest.apply(build_rationale, axis=1).values,
            "model_version": model.version,
            "as_of": as_of,
        }
    )
    return out.sort_values("score", ascending=False).reset_index(drop=True)[RECOMMENDATION_COLUMNS]


def score_pipeline(
    prices: pd.DataFrame,
    model_dir: str | Path,
    writer: RecommendationWriter | None,
    artifact: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(artifact) if artifact else latest_artifact_path(model_dir)
    model = BaselineMomentumRanker.load(path)
    log.info("loaded artifact %s (version %s)", path, model.version)
    recs = score_latest(prices, model)
    if writer is not None and not recs.empty:
        n = writer.write(recs)
        log.info("inserted %d recommendation rows", n)
    return recs
