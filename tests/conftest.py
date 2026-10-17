from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_prices(drifts: dict[str, float], n_days: int = 200, vol: float = 0.01,
                seed: int = 0, start: float = 100.0) -> pd.DataFrame:
    """Synthetic daily prices: geometric random walk with per-symbol daily drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    frames = []
    for sym, drift in drifts.items():
        rets = drift + vol * rng.standard_normal(n_days)
        close = start * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1_000_000,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices({"UP": 0.004, "FLAT": 0.0, "DOWN": -0.004, "MILD": 0.001, "NOISY": 0.0})
