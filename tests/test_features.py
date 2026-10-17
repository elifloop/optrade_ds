import numpy as np
import pandas as pd
import pytest

from optrade_ds.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY,
    add_forward_returns,
    compute_features,
    latest_features,
)


def _linear(symbol="LIN", n=80, step=1.0):
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100.0 + step * np.arange(n)
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": close})


def test_momentum_and_trend_values():
    p = _linear()
    f = compute_features(p)
    last = f.iloc[-1]
    close = p["close"].to_numpy()
    assert last["mom_5"] == pytest.approx(close[-1] / close[-6] - 1)
    assert last["mom_20"] == pytest.approx(close[-1] / close[-21] - 1)
    assert last["mom_60"] == pytest.approx(close[-1] / close[-61] - 1)
    assert last["trend_50"] == pytest.approx(close[-1] / close[-50:].mean() - 1)
    # Monotonic uptrend: only gains -> RSI 100.
    assert last["rsi_14"] == pytest.approx(100.0)
    assert set(FEATURE_COLUMNS) <= set(f.columns)


def test_volatility_annualized():
    p = _linear()
    f = compute_features(p)
    logret = np.log(p["close"] / p["close"].shift(1))
    expected = logret.iloc[-20:].std() * np.sqrt(252)
    assert f.iloc[-1]["vol_20"] == pytest.approx(expected)


def test_rsi_downtrend_and_flat():
    down = compute_features(_linear(step=-0.5)).iloc[-1]
    assert down["rsi_14"] == pytest.approx(0.0)
    flat = compute_features(_linear(step=0.0)).iloc[-1]
    assert flat["rsi_14"] == pytest.approx(50.0)
    assert flat["vol_20"] == pytest.approx(0.0)


def test_first_complete_row_is_at_min_history():
    f = compute_features(_linear(n=MIN_HISTORY))
    assert len(f) == 1
    assert f.iloc[0]["date"] == pd.bdate_range("2025-01-01", periods=MIN_HISTORY)[-1]


def test_insufficient_history_symbols_skipped(prices):
    short = _linear(symbol="SHORT", n=MIN_HISTORY - 1)
    f = compute_features(pd.concat([prices, short], ignore_index=True))
    assert "SHORT" not in set(f["symbol"])
    assert set(f["symbol"]) == set(prices["symbol"])
    assert compute_features(short).empty
    assert compute_features(short.iloc[0:0]).empty


def test_latest_and_forward_returns(prices):
    f = compute_features(prices)
    latest = latest_features(f)
    assert len(latest) == prices["symbol"].nunique()
    assert (latest["date"] == prices["date"].max()).all()
    fr = add_forward_returns(f, prices, horizon=5)
    assert fr.groupby("symbol")["fwd_ret_5"].apply(lambda s: s.tail(5).isna().all()).all()
    assert fr["fwd_ret_5"].notna().sum() > 0
