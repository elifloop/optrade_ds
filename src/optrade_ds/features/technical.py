"""Per-symbol technical features computed from daily closes.

Features (all computed per symbol on rows sorted by date):

- ``mom_5`` / ``mom_20`` / ``mom_60``: simple N-trading-day returns ``close / close.shift(N) - 1``.
- ``vol_20``: annualized realized volatility, ``std(daily log returns, 20d) * sqrt(252)``.
- ``rsi_14``: RSI-like oscillator in [0, 100] using 14-day simple averages of gains/losses.
- ``trend_50``: distance from the 50-day simple moving average, ``close / SMA50 - 1``.

Short histories: a row only gets complete features once ``MIN_HISTORY`` closes exist for its
symbol (the 60-day return needs 61 observations). Rows with any missing feature are dropped
by :func:`compute_features`, so symbols with fewer than ``MIN_HISTORY`` rows simply produce no
rows and are skipped (logged at INFO) instead of raising.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MOMENTUM_WINDOWS = (5, 20, 60)
VOL_WINDOW = 20
RSI_WINDOW = 14
TREND_WINDOW = 50
TRADING_DAYS = 252

FEATURE_COLUMNS = ["mom_5", "mom_20", "mom_60", "vol_20", "rsi_14", "trend_50"]
MIN_HISTORY = max(*MOMENTUM_WINDOWS, VOL_WINDOW, RSI_WINDOW, TREND_WINDOW) + 1


def _rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    total = gain + loss
    rsi = 100.0 * gain / total
    # Flat window (no moves at all) -> neutral 50.
    return rsi.where(total > 0, 50.0).where(gain.notna())


def _symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"].astype(float)
    out = pd.DataFrame({"symbol": g["symbol"].values, "date": g["date"].values}, index=g.index)
    for n in MOMENTUM_WINDOWS:
        out[f"mom_{n}"] = close / close.shift(n) - 1.0
    log_ret = np.log(close / close.shift(1))
    out["vol_20"] = log_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(
        TRADING_DAYS
    )
    out["rsi_14"] = _rsi(close)
    sma = close.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    out["trend_50"] = close / sma - 1.0
    out["close"] = close.values
    return out


def compute_features(prices: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Compute features for every (symbol, date). Returns columns symbol, date, close, features.

    Symbols with fewer than ``MIN_HISTORY`` rows are skipped (documented above).
    """
    cols = ["symbol", "date", "close", *FEATURE_COLUMNS]
    if prices.empty:
        return pd.DataFrame(columns=cols)
    prices = prices.sort_values(["symbol", "date"])
    counts = prices.groupby("symbol").size()
    short = counts[counts < MIN_HISTORY]
    if not short.empty:
        log.info(
            "skipping %d symbol(s) with < %d rows of history: %s",
            len(short),
            MIN_HISTORY,
            ", ".join(f"{s}({n})" for s, n in short.items()),
        )
    keep = counts[counts >= MIN_HISTORY].index
    prices = prices[prices["symbol"].isin(keep)]
    if prices.empty:
        return pd.DataFrame(columns=cols)
    parts = [_symbol_features(g) for _, g in prices.groupby("symbol", sort=True)]
    feats = pd.concat(parts, ignore_index=True)[cols]
    if dropna:
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    return feats.reset_index(drop=True)


def add_forward_returns(
    features: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5
) -> pd.DataFrame:
    """Attach ``fwd_ret_<horizon>`` (future close / close - 1) for evaluation. NaN at the tail."""
    p = prices.sort_values(["symbol", "date"])[["symbol", "date", "close"]].copy()
    p[f"fwd_ret_{horizon}"] = (
        p.groupby("symbol")["close"].shift(-horizon) / p["close"] - 1.0
    )
    return features.merge(p[["symbol", "date", f"fwd_ret_{horizon}"]], on=["symbol", "date"])


def latest_features(features: pd.DataFrame) -> pd.DataFrame:
    """Last complete feature row per symbol."""
    if features.empty:
        return features
    idx = features.groupby("symbol")["date"].idxmax()
    return features.loc[idx].sort_values("symbol").reset_index(drop=True)
