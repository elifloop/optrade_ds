"""Writers for scored recommendations."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from optrade_ds.config import normalize_database_url

RECOMMENDATION_COLUMNS = ["symbol", "score", "rationale", "model_version", "as_of"]


class RecommendationWriter(Protocol):
    def write(self, recs: pd.DataFrame) -> int: ...


def _records(recs: pd.DataFrame) -> list[tuple]:
    return [
        (
            str(r.symbol),
            float(r.score),
            None if pd.isna(r.rationale) else str(r.rationale),
            str(r.model_version),
            pd.Timestamp(r.as_of).date(),
        )
        for r in recs[RECOMMENDATION_COLUMNS].itertuples(index=False)
    ]


class PostgresRecommendationWriter:
    """Insert rows into the ``recommendations`` table (append-only, per contract)."""

    def __init__(self, database_url: str) -> None:
        self.database_url = normalize_database_url(database_url)

    def write(self, recs: pd.DataFrame) -> int:
        import psycopg

        rows = _records(recs)
        if not rows:
            return 0
        with psycopg.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO recommendations (symbol, score, rationale, model_version, as_of) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
        return len(rows)


class InMemoryRecommendationWriter:
    """Collects rows in memory; useful for tests and dry runs."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def write(self, recs: pd.DataFrame) -> int:
        rows = _records(recs)
        self.rows.extend(rows)
        return len(rows)
