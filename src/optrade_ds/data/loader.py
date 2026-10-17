"""Load daily prices (long format: one row per symbol/date) from Postgres or CSV."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from optrade_ds.config import normalize_database_url

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop rows without a close, de-duplicate and sort by (symbol, date)."""
    missing = {"symbol", "date", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"price data missing required columns: {sorted(missing)}")
    out = df.copy()
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PRICE_COLUMNS]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out.dropna(subset=["close"])
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load prices from a CSV with at least ``symbol,date,close`` columns (offline dev)."""
    return normalize_prices(pd.read_csv(path))


def load_prices_db(
    database_url: str, symbols: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load prices from the ``prices`` table using psycopg 3."""
    import psycopg

    query = "SELECT symbol, date, open, high, low, close, volume FROM prices"
    params: list = []
    if symbols:
        query += " WHERE symbol = ANY(%s)"
        params.append([s.upper() for s in symbols])
    query += " ORDER BY symbol, date"
    with psycopg.connect(normalize_database_url(database_url)) as conn, conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return normalize_prices(pd.DataFrame(rows, columns=PRICE_COLUMNS))
