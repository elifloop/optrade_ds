"""Ranker interface and artifact helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd


def new_version(now: datetime | None = None) -> str:
    """Timestamp-based, lexically sortable version string, e.g. ``20261016T064500123456Z``."""
    now = now or datetime.now(UTC)
    return now.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def latest_artifact_path(model_dir: str | Path, prefix: str = "baseline") -> Path:
    """Newest ``<prefix>-<version>.json`` in ``model_dir`` (versions sort lexically by time)."""
    paths = sorted(Path(model_dir).glob(f"{prefix}-*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no {prefix}-*.json artifacts found in {model_dir}; run `optrade-ds train` first"
        )
    return paths[-1]


class BaseRanker(ABC):
    """A model that maps per-(symbol, date) features to a cross-sectional score (higher=better)."""

    name: str = "base"
    version: str | None = None

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> Self: ...

    @abstractmethod
    def score(self, features: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...

    def artifact_name(self) -> str:
        if not self.version:
            raise ValueError("model has no version; call fit() first")
        return f"{self.name}-{self.version}.json"

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / self.artifact_name()
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))
        return out

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_dict(json.loads(Path(path).read_text()))
