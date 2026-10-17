import datetime as dt

import pandas as pd
import pytest

from optrade_ds.cli import main
from optrade_ds.data import InMemoryRecommendationWriter, load_prices_csv
from optrade_ds.data.writer import RECOMMENDATION_COLUMNS
from optrade_ds.features import MIN_HISTORY
from optrade_ds.pipelines import score_latest, score_pipeline, train_pipeline

from .conftest import make_prices


def test_score_output_schema(prices, tmp_path):
    model, _ = train_pipeline(prices, tmp_path)
    recs = score_latest(prices, model)
    assert list(recs.columns) == RECOMMENDATION_COLUMNS
    assert len(recs) == prices["symbol"].nunique()
    assert recs["score"].is_monotonic_decreasing
    assert (recs["model_version"] == model.version).all()
    assert (recs["as_of"] == prices["date"].max().date()).all()
    assert recs["rationale"].str.contains(r"20d momentum [+-]\d+\.\d%").all()
    assert recs["rationale"].str.contains(r"vol \d+% ann\.").all()
    assert recs.iloc[0]["symbol"] == "UP"


def test_score_pipeline_writes_via_fake_writer(prices, tmp_path):
    train_pipeline(prices, tmp_path)
    writer = InMemoryRecommendationWriter()
    recs = score_pipeline(prices, tmp_path, writer)
    assert len(writer.rows) == len(recs) == 5
    sym, score, rationale, version, as_of = writer.rows[0]
    assert isinstance(score, float) and isinstance(as_of, dt.date)
    assert isinstance(rationale, str) and version


def test_score_skips_short_history_symbol(prices, tmp_path):
    train_pipeline(prices, tmp_path)
    short = make_prices({"NEW": 0.01}, n_days=MIN_HISTORY - 10)
    short["date"] = short["date"] + pd.offsets.BDay(200 - (MIN_HISTORY - 10))
    combined = pd.concat([prices, short], ignore_index=True)
    writer = InMemoryRecommendationWriter()
    recs = score_pipeline(combined, tmp_path, writer)
    assert "NEW" not in set(recs["symbol"])
    assert len(writer.rows) == 5


def test_score_all_short_history_writes_nothing(tmp_path, prices):
    train_pipeline(prices, tmp_path)
    writer = InMemoryRecommendationWriter()
    recs = score_pipeline(make_prices({"A": 0.0}, n_days=10), tmp_path, writer)
    assert recs.empty and writer.rows == []


def test_train_rejects_insufficient_history(tmp_path):
    with pytest.raises(ValueError):
        train_pipeline(make_prices({"A": 0.0, "B": 0.0}, n_days=20), tmp_path)


def test_cli_all_with_csv(prices, tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    prices.to_csv(csv, index=False)
    assert len(load_prices_csv(csv)) == len(prices)
    rc = main(["all", "--csv", str(csv), "--model-dir", str(tmp_path / "models")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would insert 5 recommendations" in out
    assert len(list((tmp_path / "models").glob("baseline-*.json"))) == 1


def test_cli_score_without_artifact_fails(prices, tmp_path):
    csv = tmp_path / "prices.csv"
    prices.to_csv(csv, index=False)
    assert main(["score", "--csv", str(csv), "--model-dir", str(tmp_path / "empty")]) == 1
