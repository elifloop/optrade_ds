"""Baseline momentum ranker: weighted z-score of momentum/trend features minus a volatility penalty.

    score = sum_i w_i * z(feature_i) - vol_penalty * z(vol_20)

z-scores use the mean/std of each feature over the training set (stored in the artifact), so a
saved model scores any number of symbols deterministically. Weights are either configured
(``DEFAULT_WEIGHTS`` or user supplied) or learned as the non-negative per-feature rank IC against
forward returns, normalized to sum to 1 (falls back to configured weights if no feature has
positive IC).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

import numpy as np
import pandas as pd

from optrade_ds.eval.metrics import rank_ic
from optrade_ds.features.technical import FEATURE_COLUMNS
from optrade_ds.models.base import BaseRanker, new_version

DEFAULT_WEIGHTS: dict[str, float] = {
    "mom_5": 0.10,
    "mom_20": 0.35,
    "mom_60": 0.30,
    "trend_50": 0.15,
    "rsi_14": 0.10,
}
RISK_FEATURE = "vol_20"
DEFAULT_VOL_PENALTY = 0.25


class BaselineMomentumRanker(BaseRanker):
    name = "baseline"
    model_type = "baseline_momentum"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        vol_penalty: float = DEFAULT_VOL_PENALTY,
        learn_weights: bool = False,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        unknown = set(self.weights) - set(FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"unknown features in weights: {sorted(unknown)}")
        self.vol_penalty = float(vol_penalty)
        self.learn_weights = learn_weights
        self.feature_stats: dict[str, dict[str, float]] = {}
        self.training: dict[str, Any] = {}
        self.version: str | None = None
        self.created_at: str | None = None

    @property
    def features(self) -> list[str]:
        return [*self.weights.keys(), RISK_FEATURE]

    def _check_fitted(self) -> None:
        if not self.feature_stats:
            raise RuntimeError("model is not fitted")

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> Self:
        if features.empty:
            raise ValueError("cannot fit on empty features (not enough price history?)")
        stats = {}
        for f in self.features:
            col = features[f].astype(float)
            std = float(col.std(ddof=0))
            stats[f] = {"mean": float(col.mean()), "std": std if std > 0 else 1.0}
        self.feature_stats = stats

        weight_mode = "configured"
        if self.learn_weights and target is not None:
            df = features.assign(_target=target.values)
            learned = {}
            for f in self.weights:
                ic = rank_ic(df, score_col=f, target_col="_target")
                learned[f] = max(float(ic.mean()), 0.0) if ic.size else 0.0
            total = sum(learned.values())
            if total > 0:
                self.weights = {f: round(w / total, 6) for f, w in learned.items()}
                weight_mode = "learned"
            else:
                weight_mode = "configured(fallback: no positive IC)"

        now = datetime.now(UTC)
        self.version = new_version(now)
        self.created_at = now.isoformat()
        dates = pd.to_datetime(features["date"])
        self.training = {
            "n_rows": int(len(features)),
            "n_symbols": int(features["symbol"].nunique()),
            "symbols": sorted(features["symbol"].unique().tolist()),
            "date_min": dates.min().date().isoformat(),
            "date_max": dates.max().date().isoformat(),
            "weight_mode": weight_mode,
        }
        return self

    def zscores(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(
            {
                f: (features[f].astype(float) - s["mean"]) / s["std"]
                for f, s in self.feature_stats.items()
            },
            index=features.index,
        )

    def score(self, features: pd.DataFrame) -> pd.Series:
        z = self.zscores(features)
        raw = sum(w * z[f] for f, w in self.weights.items())
        score = raw - self.vol_penalty * z[RISK_FEATURE]
        return pd.Series(np.asarray(score, dtype=float), index=features.index, name="score")

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "features": self.features,
            "weights": self.weights,
            "risk_feature": RISK_FEATURE,
            "vol_penalty": self.vol_penalty,
            "feature_stats": self.feature_stats,
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("model_type") != cls.model_type:
            raise ValueError(f"not a {cls.model_type} artifact: {data.get('model_type')!r}")
        model = cls(weights=data["weights"], vol_penalty=data["vol_penalty"])
        model.feature_stats = data["feature_stats"]
        model.training = data.get("training", {})
        model.version = data["version"]
        model.created_at = data.get("created_at")
        return model
