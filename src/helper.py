"""
helpers.py — utility functions for data access, text/features, numeric
time-series operations, and statistics/validation.

Sections
--------
01 — Data access and storage
02 — Text and feature operations
03 — Numeric and time-series operations
04 — Statistics and validation
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy import stats


# =====================================================================
# 01 — DATA ACCESS AND STORAGE
# =====================================================================

def fetch_records(source, query=None, time_range=None):
    """Retrieve records matching the specified criteria.

    Parameters
    ----------
    source : list[dict] | pd.DataFrame | str | Path
        In-memory records, a path to a ``.json`` / ``.jsonl`` / ``.csv``
        file, or an ``http(s)://`` URL (fetched with GET).
    query : dict | str | callable | None
        ``dict``  — field/value equality filter (also sent as URL params
                    for HTTP sources),
        ``str``   — a pandas ``DataFrame.query`` expression,
        callable  — predicate applied per record,
        ``None``  — no filter.
    time_range : tuple | dict | None
        ``(start, end)`` applied to the ``timestamp`` field, or
        ``{"field": name, "start": ..., "end": ...}``.

    Returns
    -------
    list[dict]
        Matching records.
    """
    # --- load records -------------------------------------------------
    if isinstance(source, pd.DataFrame):
        records = source.to_dict("records")
    elif isinstance(source, (list, tuple)):
        records = list(source)
    elif isinstance(source, (str, Path)) and str(source).startswith(("http://", "https://")):
        params = query if isinstance(query, dict) else None
        resp = requests.get(str(source), params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records = payload if isinstance(payload, list) else [payload]
        if params is not None:
            query = None  # already applied server-side
    elif isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text())
            records = payload if isinstance(payload, list) else [payload]
        elif suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif suffix == ".csv":
            records = list(csv.DictReader(path.open(newline="")))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # --- apply query --------------------------------------------------
    if query is None:
        pass
    elif isinstance(query, dict):
        records = [r for r in records if all(r.get(k) == v for k, v in query.items())]
    elif isinstance(query, str):
        records = pd.DataFrame(records).query(query).to_dict("records")
    elif callable(query):
        records = [r for r in records if query(r)]
    else:
        raise TypeError(f"Unsupported query type: {type(query)}")

    # --- apply time range ---------------------------------------------
    if time_range is not None:
        if isinstance(time_range, dict):
            field = time_range.get("field", "timestamp")
            start, end = time_range.get("start"), time_range.get("end")
        else:
            field = "timestamp"
            start, end = time_range
        start = parse_timestamp(start) if start is not None else None
        end = parse_timestamp(end) if end is not None else None
        filtered = []
        for r in records:
            if field not in r:
                continue
            ts = parse_timestamp(r[field])
            if (start is None or ts >= start) and (end is None or ts <= end):
                filtered.append(r)
        records = filtered

    return records


def parse_timestamp(value, format=None, timezone="UTC"):
    """Parse and normalize a timestamp to a timezone-aware datetime.

    Parameters
    ----------
    value : datetime | int | float | str
        Datetime object, epoch seconds/milliseconds, or a timestamp string.
    format : str | None
        ``strptime`` format for string input; ``None`` lets pandas infer.
    timezone : str
        Target timezone name (e.g. ``"UTC"``, ``"America/New_York"``).
        Naive inputs are assumed to already be in this timezone.

    Returns
    -------
    datetime
        Timezone-aware datetime in the requested timezone.
    """
    tz = ZoneInfo(timezone) if isinstance(timezone, str) else timezone

    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # treat large numbers as epoch milliseconds
        ts = datetime.fromtimestamp(value / 1000.0 if value > 1e11 else value,
                                    tz=dt_timezone.utc)
    elif isinstance(value, str):
        ts = datetime.strptime(value, format) if format else pd.Timestamp(value).to_pydatetime()
    elif isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
    else:
        raise TypeError(f"Cannot parse timestamp from {type(value)}")

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz)
    return ts.astimezone(tz)


def validate_records(records, schema):
    """Check required fields, types, and constraints.

    Parameters
    ----------
    records : list[dict]
    schema : dict
        ``{field: {"type": type | tuple[type], "required": bool,
                  "min": num, "max": num, "allowed": iterable,
                  "pattern": regex_str, "check": callable}}``

    Returns
    -------
    (dict, list[dict])
        ``({"valid": n_valid, "invalid": n_invalid, "errors": {idx: [msgs]}},
           [invalid_records])``
    """
    import re

    errors = {}
    for i, rec in enumerate(records):
        msgs = []
        for field, rules in schema.items():
            present = field in rec and rec[field] is not None
            if rules.get("required", False) and not present:
                msgs.append(f"missing required field '{field}'")
                continue
            if not present:
                continue
            v = rec[field]
            expected = rules.get("type")
            if expected is not None and not isinstance(v, expected):
                msgs.append(f"'{field}' has type {type(v).__name__}, expected {expected}")
                continue
            if "min" in rules and v < rules["min"]:
                msgs.append(f"'{field}'={v} below min {rules['min']}")
            if "max" in rules and v > rules["max"]:
                msgs.append(f"'{field}'={v} above max {rules['max']}")
            if "allowed" in rules and v not in rules["allowed"]:
                msgs.append(f"'{field}'={v} not in allowed values")
            if "pattern" in rules and not re.fullmatch(rules["pattern"], str(v)):
                msgs.append(f"'{field}' does not match pattern")
            if "check" in rules and not rules["check"](v):
                msgs.append(f"'{field}' failed custom check")
        if msgs:
            errors[i] = msgs

    invalid_idx = set(errors)
    invalid = [r for i, r in enumerate(records) if i in invalid_idx]
    results = {
        "valid": len(records) - len(invalid),
        "invalid": len(invalid),
        "errors": errors,
    }
    return results, invalid


def append_only_write(store, records, key):
    """Append new records without silently overwriting existing history.

    Records whose ``key`` value already exists in the store are skipped.

    Parameters
    ----------
    store : str | Path | dict
        ``".jsonl"`` / ``".csv"`` file path, or
        ``{"path": "file.db", "table": "name"}`` for SQLite.
    records : list[dict]
    key : str
        Field used as the unique key for deduplication.

    Returns
    -------
    dict
        ``{"status": "ok", "written": n, "skipped": n, "total": n}``
    """
    if isinstance(records, dict):
        records = [records]
    records = list(records)

    if isinstance(store, dict):  # SQLite store
        path, table = store["path"], store["table"]
        conn = sqlite3.connect(path)
        try:
            cols = {k for r in records for k in r} | {key}
            cols = sorted(cols)
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                + ", ".join(f'"{c}" TEXT' for c in cols)
                + f', PRIMARY KEY ("{key}"))'
            )
            written = 0
            for r in records:
                cur = conn.execute(
                    f'INSERT OR IGNORE INTO "{table}" ('
                    + ", ".join(f'"{c}"' for c in cols) + ") VALUES ("
                    + ", ".join("?" for _ in cols) + ")",
                    [json.dumps(r.get(c)) if isinstance(r.get(c), (dict, list)) else r.get(c)
                     for c in cols],
                )
                written += cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return {"status": "ok", "written": written,
                "skipped": len(records) - written, "total": len(records)}

    path = Path(store)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        existing = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    existing.add(json.loads(line).get(key))
        new = [r for r in records if r.get(key) not in existing]
        with path.open("a") as f:
            for r in new:
                f.write(json.dumps(r, default=str) + "\n")

    elif suffix == ".csv":
        existing = set()
        if path.exists():
            with path.open(newline="") as f:
                existing = {row.get(key) for row in csv.DictReader(f)}
        new = [r for r in records if str(r.get(key)) not in existing]
        if new:
            write_header = not path.exists() or path.stat().st_size == 0
            fieldnames = list(dict.fromkeys(k for r in new for k in r))
            with path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerows(new)
    else:
        raise ValueError(f"Unsupported store type: {suffix} "
                         "(use .jsonl, .csv, or {'path':..., 'table':...} for SQLite)")

    return {"status": "ok", "written": len(new),
            "skipped": len(records) - len(new), "total": len(records)}


# =====================================================================
# 02 — TEXT AND FEATURE OPERATIONS
# =====================================================================

def encode_text(text, model="hashing"):
    """Convert text into a numeric embedding vector.

    Parameters
    ----------
    text : str
    model : str | object
        Identifier or model object. If an object with ``.encode``,
        ``.transform``, or ``.__call__`` is supplied it is used directly.
        ``"hashing"`` (default) uses a deterministic
        ``HashingVectorizer``-based embedding — no downloads required.

    Returns
    -------
    np.ndarray
        1-D numeric vector.
    """
    if not isinstance(model, str):
        if hasattr(model, "encode"):
            return np.asarray(model.encode(text)).ravel()
        if hasattr(model, "transform"):
            return np.asarray(model.transform([text])).ravel()
        if callable(model):
            return np.asarray(model(text)).ravel()
        raise TypeError("model object must expose .encode, .transform, or be callable")

    if model == "hashing":
        from sklearn.feature_extraction.text import HashingVectorizer
        vec = HashingVectorizer(n_features=256, norm="l2", alternate_sign=False)
        return vec.transform([text]).toarray().ravel()

    # try sentence-transformers style identifiers if the package exists
    try:
        from sentence_transformers import SentenceTransformer
        return np.asarray(SentenceTransformer(model).encode(text)).ravel()
    except ImportError:
        raise ValueError(
            f"Unknown model identifier {model!r}; install sentence-transformers "
            "or use model='hashing' or pass a model object.")


def apply_model(model, inputs):
    """Execute a model against supplied inputs.

    Dispatches on the model interface: ``.predict_proba`` / ``.predict``
    (sklearn-style), ``.forward`` (torch nn.Module), or any callable.

    Returns
    -------
    np.ndarray
        Predictions or scores.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(inputs))
    if hasattr(model, "predict"):
        return np.asarray(model.predict(inputs))

    try:
        import torch
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
                return model(x).detach().cpu().numpy()
    except ImportError:
        pass

    if callable(model):
        return np.asarray([model(x) for x in inputs]) if isinstance(inputs, list) \
            else np.asarray(model(inputs))

    raise TypeError(f"Don't know how to apply model of type {type(model)}")


def count_records(records, group_by=None, time_window=None):
    """Count records within specified groups and windows.

    Parameters
    ----------
    records : list[dict] | pd.DataFrame
    group_by : str | list[str] | None
        Grouping key field(s).
    time_window : str | dict | None
        Pandas frequency string (``"1h"``, ``"1D"``, ...) applied to the
        ``timestamp`` field, or ``{"field": name, "window": freq}``.

    Returns
    -------
    pd.DataFrame
        One row per group/window with a ``count`` column.
    """
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    group_by = [group_by] if isinstance(group_by, str) else (group_by or [])

    keys = list(group_by)
    if time_window is not None:
        if isinstance(time_window, dict):
            field, freq = time_window.get("field", "timestamp"), time_window["window"]
        else:
            field, freq = "timestamp", time_window
        df = df.copy()
        df[field] = pd.to_datetime(df[field])
        keys.append(pd.Grouper(key=field, freq=freq))

    if not keys:
        return pd.DataFrame({"count": [len(df)]})
    return df.groupby(keys).size().reset_index(name="count")


def compute_vector_similarity(vector_a, vector_b, metric="cosine"):
    """Compute a similarity measure between two numeric vectors.

    Parameters
    ----------
    vector_a, vector_b : array-like
    metric : str
        ``"cosine"`` (default), ``"dot"``, ``"euclidean"`` (returns
        similarity = 1 / (1 + distance)), or ``"pearson"``.

    Returns
    -------
    float
    """
    a = np.asarray(vector_a, dtype=float).ravel()
    b = np.asarray(vector_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    if metric == "cosine":
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0
    if metric == "dot":
        return float(np.dot(a, b))
    if metric == "euclidean":
        return float(1.0 / (1.0 + np.linalg.norm(a - b)))
    if metric == "pearson":
        return float(stats.pearsonr(a, b).statistic)
    raise ValueError(f"Unknown metric {metric!r}")


# =====================================================================
# 03 — NUMERIC AND TIME-SERIES OPERATIONS
# =====================================================================

def align_time_series(series_a, series_b, time_key="timestamp", how="inner"):
    """Match observations according to their timestamps.

    Parameters
    ----------
    series_a, series_b : list[dict] | pd.Series | pd.DataFrame
    time_key : str
        Timestamp field name for list-of-dict input (ignored when the
        inputs are pandas objects indexed by time).
    how : str
        ``"inner"`` (only matching timestamps, default), ``"outer"``,
        ``"left"``, or ``"right"``.

    Returns
    -------
    tuple
        Aligned datasets — for pandas input, a merged ``DataFrame``;
        for list-of-dict input, ``(aligned_a, aligned_b)`` lists of
        records sorted by time.
    """
    if isinstance(series_a, (pd.Series, pd.DataFrame)):
        a = series_a.to_frame("a") if isinstance(series_a, pd.Series) else series_a.add_suffix("_a")
        b = series_b.to_frame("b") if isinstance(series_b, pd.Series) else series_b.add_suffix("_b")
        return a.join(b, how=how)

    def index_by_time(records):
        return {parse_timestamp(r[time_key]): r for r in records}

    map_a, map_b = index_by_time(series_a), index_by_time(series_b)
    keys_a, keys_b = set(map_a), set(map_b)
    if how == "inner":
        times = sorted(keys_a & keys_b)
    elif how == "outer":
        times = sorted(keys_a | keys_b)
    elif how == "left":
        times = sorted(keys_a)
    elif how == "right":
        times = sorted(keys_b)
    else:
        raise ValueError(f"Unknown join type {how!r}")

    return ([map_a.get(t) for t in times], [map_b.get(t) for t in times])


def compute_windowed_change(values, start_offset, end_offset):
    """Compute changes between specified positions in a series.

    For each index ``i`` the change is::

        values[i + end_offset] - values[i + start_offset]

    Only indices where both positions are valid are returned.

    Parameters
    ----------
    values : array-like
        Ordered numeric values.
    start_offset, end_offset : int
        Relative positions (may be negative).

    Returns
    -------
    np.ndarray
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    lo = max(0, -start_offset, -end_offset)
    hi = min(n, n - start_offset, n - end_offset)
    idx = np.arange(lo, hi)
    return v[idx + end_offset] - v[idx + start_offset]


def subtract_aligned_series(series_a, series_b):
    """Subtract corresponding observations of two aligned numeric series.

    Accepts array-likes or pandas objects (which align on their index).
    """
    if isinstance(series_a, (pd.Series, pd.DataFrame)) or \
       isinstance(series_b, (pd.Series, pd.DataFrame)):
        return series_a - series_b
    return np.asarray(series_a, dtype=float) - np.asarray(series_b, dtype=float)


def apply_adjustments(values, adjustments):
    """Apply specified adjustments to numeric values.

    Parameters
    ----------
    values : array-like
    adjustments : scalar | array-like | dict | callable
        scalar / array-like — added to (all or element-wise) values,
        ``{index: delta}`` — added at the given positions,
        ``{"op": "add"|"multiply"|"replace", "values": ...}`` — a rule,
        callable — applied element-wise.

    Returns
    -------
    np.ndarray
    """
    v = np.asarray(values, dtype=float).copy()

    if callable(adjustments):
        return np.array([adjustments(x) for x in v], dtype=float)

    if isinstance(adjustments, dict):
        if "op" in adjustments:
            op, adj = adjustments["op"], np.asarray(adjustments["values"], dtype=float)
            if op == "add":
                return v + adj
            if op == "multiply":
                return v * adj
            if op == "replace":
                out = v.copy()
                out[:] = np.broadcast_to(adj, v.shape)
                return out
            raise ValueError(f"Unknown adjustment op {op!r}")
        for idx, delta in adjustments.items():  # {index: delta}
            v[idx] += delta
        return v

    return v + np.asarray(adjustments, dtype=float)


# =====================================================================
# 04 — STATISTICS AND VALIDATION
# =====================================================================

def split_by_time(records, boundaries, time_key="timestamp"):
    """Split records by time without random shuffling.

    Parameters
    ----------
    records : list[dict] | pd.DataFrame
    boundaries : array-like
        Sorted boundary timestamps; ``len(boundaries) + 1`` segments are
        produced.
    time_key : str
        Timestamp field for list-of-dict input.

    Returns
    -------
    list
        Chronologically ordered segments (list-of-dicts or DataFrames,
        matching the input type).
    """
    bounds = [parse_timestamp(b) for b in boundaries]
    bounds_v = [datetime.min.replace(tzinfo=dt_timezone.utc)] + bounds \
        + [datetime.max.replace(tzinfo=dt_timezone.utc)]

    def norm(t):
        return parse_timestamp(t)

    if isinstance(records, pd.DataFrame):
        df = records.copy()
        df[time_key] = pd.to_datetime(df[time_key], utc=True)
        df = df.sort_values(time_key)
        segments = []
        for i in range(len(bounds_v) - 1):
            mask = (df[time_key] >= pd.Timestamp(bounds_v[i])) & \
                   (df[time_key] < pd.Timestamp(bounds_v[i + 1]))
            segments.append(df[mask])
        return segments

    ordered = sorted(records, key=lambda r: norm(r[time_key]))
    segments = [[] for _ in range(len(bounds_v) - 1)]
    for r in ordered:
        t = norm(r[time_key])
        for i in range(len(bounds_v) - 1):
            if bounds_v[i] <= t < bounds_v[i + 1]:
                segments[i].append(r)
                break
    return segments


def compute_rank_correlation(values_a, values_b):
    """Spearman rank-correlation coefficient between two numeric sequences.

    Returns
    -------
    float
        rho in [-1, 1]: strength and direction of rank association.
    """
    return float(stats.spearmanr(values_a, values_b).statistic)


def aggregate_by_quantile(records, score, value, buckets=4):
    """Group observations by score quantiles and aggregate a value field.

    Parameters
    ----------
    records : list[dict] | pd.DataFrame
    score : str
        Field used to define quantile buckets.
    value : str
        Field aggregated within each bucket.
    buckets : int
        Number of quantile buckets (e.g. 4 = quartiles, 10 = deciles).

    Returns
    -------
    pd.DataFrame
        One row per bucket with count/mean/median/std/min/max/sum.
    """
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    df = df.copy()
    df["bucket"] = pd.qcut(df[score], q=buckets, duplicates="drop")
    return df.groupby("bucket", observed=True)[value].agg(
        ["count", "mean", "median", "std", "min", "max", "sum"]).reset_index()


def run_permutation_test(observed_statistic, data, statistic_function,
                         iterations=1000, seed=None):
    """Permutation test: build a null distribution and compare statistics.

    Parameters
    ----------
    observed_statistic : float
        The statistic computed on the real (unpermuted) data.
    data : array-like | tuple(array-like, array-like)
        A single dataset (permuted before each call to
        ``statistic_function``), or two groups whose labels are permuted
        — the standard two-sample scheme.
    statistic_function : callable
        Called as ``statistic_function(permuted)`` for a single dataset
        or ``statistic_function(group_a, group_b)`` for the two-sample
        scheme.
    iterations : int
    seed : int | None

    Returns
    -------
    dict
        ``{"observed": float, "null_distribution": np.ndarray,
           "p_value": float}`` — ``p_value`` is the fraction of permuted
        statistics at least as extreme as the observed one.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(iterations)

    if isinstance(data, tuple) and len(data) == 2:
        a, b = np.asarray(data[0]), np.asarray(data[1])
        pooled = np.concatenate([a, b])
        n_a = len(a)
        for i in range(iterations):
            perm = rng.permutation(pooled)
            null[i] = statistic_function(perm[:n_a], perm[n_a:])
    else:
        arr = np.asarray(data)
        for i in range(iterations):
            null[i] = statistic_function(rng.permutation(arr))

    p_value = float(np.mean(np.abs(null) >= np.abs(observed_statistic)))
    return {"observed": float(observed_statistic),
            "null_distribution": null, "p_value": p_value}


def evaluate_conditions(values, conditions):
    """Evaluate whether supplied values satisfy specified conditions.

    Parameters
    ----------
    values : array-like
    conditions : callable | str | dict | list
        Condition spec(s):
          * callable — ``cond(v) -> bool``
          * str — comparison like ``">= 5"``, ``"== 'x'"``,
            ``"between 1 and 5"``, ``"in [1, 2, 3]"``
          * dict — ``{"field": name, "op": ">", "value": 5}`` for
            dict records.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_values,)`` for a single condition or
        ``(n_values, n_conditions)`` for a list.
    """
    import ast
    import operator as op_mod

    OPS = {">": op_mod.gt, ">=": op_mod.ge, "<": op_mod.lt, "<=": op_mod.le,
           "==": op_mod.eq, "!=": op_mod.ne}

    def to_predicate(cond):
        if callable(cond):
            return cond
        if isinstance(cond, str):
            s = cond.strip()
            if s.lower().startswith("between"):
                parts = s.split()
                lo, hi = ast.literal_eval(parts[1]), ast.literal_eval(parts[3])
                return lambda v, lo=lo, hi=hi: lo <= v <= hi
            if s.lower().startswith("in "):
                allowed = ast.literal_eval(s[3:].strip())
                return lambda v, allowed=allowed: v in allowed
            for sym in (">=", "<=", "!=", "==", ">", "<"):
                if s.startswith(sym):
                    rhs = ast.literal_eval(s[len(sym):].strip())
                    fn = OPS[sym]
                    return lambda v, fn=fn, rhs=rhs: fn(v, rhs)
            raise ValueError(f"Cannot parse condition {cond!r}")
        if isinstance(cond, dict):
            field, fn, target = cond["field"], OPS[cond["op"]], cond["value"]
            return lambda v, field=field, fn=fn, target=target: fn(v[field], target)
        raise TypeError(f"Unsupported condition type: {type(cond)}")

    single = not isinstance(conditions, (list, tuple)) or callable(conditions)
    conds = [conditions] if single else list(conditions)
    preds = [to_predicate(c) for c in conds]

    vals = list(values) if not isinstance(values, np.ndarray) else values.tolist()
    result = np.array([[p(v) for p in preds] for v in vals], dtype=bool)
    return result[:, 0] if single else result


# =====================================================================
# smoke test
# =====================================================================

if __name__ == "__main__":
    recs = [
        {"id": 1, "timestamp": "2024-01-01", "text": "good", "score": 0.9, "y": 10},
        {"id": 2, "timestamp": "2024-02-01", "text": "bad", "score": 0.2, "y": 4},
        {"id": 3, "timestamp": "2024-03-01", "text": "ok", "score": 0.5, "y": 7},
    ]

    print("fetch:", len(fetch_records(recs, {"text": "good"})),
          len(fetch_records(recs, "score >= 0.5")),
          len(fetch_records(recs, None, ("2024-01-15", "2024-12-31"))))
    print("ts:", parse_timestamp("2024-01-01 12:00", "%Y-%m-%d %H:%M", "UTC"))
    schema = {"id": {"type": int, "required": True},
              "score": {"type": (int, float), "min": 0, "max": 1}}
    print("validate:", validate_records(recs, schema)[0])
    print("count:\n", count_records(recs, "text", {"field": "timestamp", "window": "ME"}))
    print("sim:", round(compute_vector_similarity([1, 0], [1, 0]), 3))
    print("embed shape:", encode_text("hello world").shape)
    a = [{"timestamp": "2024-01-01", "v": 1}, {"timestamp": "2024-02-01", "v": 2}]
    b = [{"timestamp": "2024-02-01", "v": 5}, {"timestamp": "2024-03-01", "v": 6}]
    print("align:", align_time_series(a, b))
    print("win change:", compute_windowed_change([1, 3, 6, 10], 0, 1))
    print("subtract:", subtract_aligned_series([5, 7, 9], [1, 2, 3]))
    print("adjust:", apply_adjustments([1, 2, 3], {1: 10}))
    print("split sizes:", [len(s) for s in split_by_time(recs, ["2024-02-15"])])
    print("spearman:", compute_rank_correlation([1, 2, 3], [3, 2, 1]))
    print("quantiles:\n", aggregate_by_quantile(recs, "score", "y", 2))
    print("perm:", run_permutation_test(
        5.0, ([1, 2, 3], [5, 6, 7]), lambda x, y: x.mean() - y.mean(),
        iterations=200, seed=0)["p_value"])
    print("conditions:", evaluate_conditions([1, 5, 9], [">= 5", "< 8"]))
