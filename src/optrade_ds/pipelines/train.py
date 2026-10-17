"""Train pipeline: prices -> features -> fit -> eval -> save artifact."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from optrade_ds.eval.metrics import evaluate_scores
from optrade_ds.features.technical import add_forward_returns, compute_features
from optrade_ds.models.baseline import BaselineMomentumRanker

log = logging.getLogger(__name__)


def train_model(
    prices: pd.DataFrame,
    learn_weights: bool = False,
    horizon: int = 5,
    holdout_frac: float = 0.2,
) -> BaselineMomentumRanker:
    """Fit the baseline ranker on all data and attach eval metrics to ``model.training``.

    Metrics: in-sample rank IC on all rows, plus out-of-sample rank IC of a model fit on the
    first ``1 - holdout_frac`` of dates and evaluated on the remaining dates.
    """
    feats = compute_features(prices)
    if feats.empty:
        raise ValueError("no symbols have enough price history to compute features")
    target_col = f"fwd_ret_{horizon}"
    data = add_forward_returns(feats, prices, horizon)
    labelled = data.dropna(subset=[target_col])

    if learn_weights:
        # Weight learning needs labels, so fit on rows that have a forward return.
        model = BaselineMomentumRanker(learn_weights=True).fit(labelled, labelled[target_col])
    else:
        model = BaselineMomentumRanker().fit(data)

    metrics: dict = {"in_sample": evaluate_scores(labelled.assign(score=model.score(labelled)),
                                                   target_col=target_col)}
    dates = sorted(labelled["date"].unique())
    cut = int(len(dates) * (1 - holdout_frac))
    if 0 < cut < len(dates):
        train_part = labelled[labelled["date"] < dates[cut]]
        test_part = labelled[labelled["date"] >= dates[cut]]
        oos = BaselineMomentumRanker(learn_weights=learn_weights).fit(
            train_part, train_part[target_col]
        )
        metrics["holdout"] = evaluate_scores(
            test_part.assign(score=oos.score(test_part)), target_col=target_col
        )
    model.training["metrics"] = metrics
    model.training["horizon_days"] = horizon
    return model


def train_pipeline(
    prices: pd.DataFrame, model_dir: str | Path, learn_weights: bool = False, horizon: int = 5
) -> tuple[BaselineMomentumRanker, Path]:
    model = train_model(prices, learn_weights=learn_weights, horizon=horizon)
    path = model.save(model_dir)
    log.info("saved artifact %s", path)
    log.info("metrics: %s", model.training.get("metrics"))
    return model, path
