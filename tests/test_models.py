import json

import pandas as pd
import pytest

from optrade_ds.eval import evaluate_scores, rank_ic
from optrade_ds.features import compute_features, latest_features
from optrade_ds.models import BaselineMomentumRanker, latest_artifact_path
from optrade_ds.pipelines import train_model


def test_uptrend_ranks_above_downtrend(prices):
    feats = compute_features(prices)
    model = BaselineMomentumRanker().fit(feats)
    latest = latest_features(feats)
    scores = dict(zip(latest["symbol"], model.score(latest), strict=True))
    assert scores["UP"] > scores["FLAT"] > scores["DOWN"]
    assert max(scores, key=scores.get) == "UP"
    assert min(scores, key=scores.get) == "DOWN"


def test_artifact_roundtrip(prices, tmp_path):
    model = train_model(prices)
    path = model.save(tmp_path)
    assert path.name == f"baseline-{model.version}.json"
    data = json.loads(path.read_text())
    for key in ("version", "created_at", "features", "weights", "training", "feature_stats"):
        assert key in data
    assert "metrics" in data["training"]
    loaded = BaselineMomentumRanker.load(path)
    assert loaded.version == model.version
    assert loaded.weights == model.weights
    latest = latest_features(compute_features(prices))
    pd.testing.assert_series_equal(loaded.score(latest), model.score(latest))
    assert latest_artifact_path(tmp_path) == path


def test_latest_artifact_picks_newest(prices, tmp_path):
    feats = compute_features(prices)
    a = BaselineMomentumRanker().fit(feats)
    a.version = "20250101T000000000000Z"
    b = BaselineMomentumRanker().fit(feats)
    b.version = "20260101T000000000000Z"
    b.save(tmp_path)
    a.save(tmp_path)
    assert latest_artifact_path(tmp_path).name == "baseline-20260101T000000000000Z.json"


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_artifact_path(tmp_path)


def test_learned_weights_are_normalized(prices):
    model = train_model(prices, learn_weights=True)
    assert sum(model.weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(w >= 0 for w in model.weights.values())


def test_rank_ic_perfect_and_eval_summary():
    df = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 4,
        "score": [1, 2, 3, 4, 4, 3, 2, 1],
        "fwd_ret_5": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
    })
    ic = rank_ic(df)
    assert ic.tolist() == pytest.approx([1.0, -1.0])
    summary = evaluate_scores(df)
    assert summary["n_dates"] == 2
    assert summary["rank_ic_mean"] == pytest.approx(0.0)
    assert evaluate_scores(df.iloc[0:0])["n_dates"] == 0


def test_fit_on_empty_raises():
    with pytest.raises(ValueError):
        BaselineMomentumRanker().fit(compute_features(pd.DataFrame(columns=["symbol", "date", "close"])))
