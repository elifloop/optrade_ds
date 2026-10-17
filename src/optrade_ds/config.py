"""Environment-driven configuration (see shared contract)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_DATABASE_URL = "postgresql://optrade:optrade@localhost:5432/optrade"
DEFAULT_MODEL_DIR = "./artifacts"


def normalize_database_url(url: str) -> str:
    """Accept plain ``postgresql://`` plus SQLAlchemy-style variants for psycopg 3."""
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str
    model_dir: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=normalize_database_url(
                os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
            ),
            model_dir=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        )
