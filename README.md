# optrade-datascience

Data science layer of **optrade** (step 2 of the shared data flow). Reads daily prices from
Postgres (filled by `optrade-data`), computes technical features, trains a baseline ranker,
saves a JSON artifact, and inserts scored rows into the `recommendations` table served by
`optrade-backend`.

> Not investment advice. The baseline model is a transparent heuristic for plumbing/demo purposes.

## Layout

```
src/optrade_ds/
  config.py            DATABASE_URL / MODEL_DIR from env (plain postgresql:// accepted)
  data/loader.py       load_prices_db (psycopg 3) / load_prices_csv (offline dev)
  data/writer.py       PostgresRecommendationWriter, InMemoryRecommendationWriter (tests/dry-run)
  features/technical.py  mom_5/20/60, vol_20 (annualized), rsi_14, trend_50
  models/base.py       BaseRanker (fit/score/save/load), artifact versioning helpers
  models/baseline.py   BaselineMomentumRanker
  eval/metrics.py      forward-return rank IC, IC IR, hit rate
  pipelines/           train / score pipelines
  cli.py               `optrade-ds train | score | all`
notebooks/             exploration notebook skeleton
tests/                 pytest on synthetic prices (no DB needed)
```

## Quick start

```bash
uv sync
cp .env.example .env   # adjust DATABASE_URL; export the vars in your shell
export DATABASE_URL=postgresql://optrade:optrade@localhost:5432/optrade MODEL_DIR=./artifacts

uv run optrade-ds train            # load -> features -> fit -> eval -> artifacts/baseline-<version>.json
uv run optrade-ds score            # latest artifact -> score latest date per symbol -> INSERT recommendations
uv run optrade-ds all              # train then score (Docker default)

# Offline: CSV with symbol,date,close[,open,high,low,volume]; scoring never writes to DB
uv run optrade-ds all --csv prices.csv
```

Useful flags: `--symbols AAPL,MSFT`, `--learn-weights`, `--horizon 5`, `--artifact PATH`,
`--dry-run` (score without writing), `--model-dir`, `--database-url`, `-v`.

## Features

Computed per symbol on date-sorted closes:

| feature | definition |
|---|---|
| `mom_5`, `mom_20`, `mom_60` | `close / close.shift(N) - 1` |
| `vol_20` | std of 20 daily log returns × √252 (annualized) |
| `rsi_14` | 100 × avg gain / (avg gain + avg loss), 14-day simple averages; 50 if flat |
| `trend_50` | `close / SMA50 - 1` |

**Short histories:** a symbol needs `MIN_HISTORY = 61` rows before it gets a complete feature
row. Symbols with fewer rows are skipped (logged at INFO) during both training and scoring —
they simply get no recommendation rather than causing an error. If *no* symbol has enough
history, `train` fails with a clear message and `score` writes nothing.

## Model

`BaselineMomentumRanker`:

```
score = Σ w_i · z(feature_i) − vol_penalty · z(vol_20)
```

z-scores use training-set mean/std per feature (stored in the artifact). Default weights:
`mom_20 0.35, mom_60 0.30, trend_50 0.15, mom_5 0.10, rsi_14 0.10`, `vol_penalty 0.25`.
With `--learn-weights`, weights = non-negative mean rank IC of each feature vs. the forward
return, normalized to sum to 1 (falls back to defaults if none positive).

Artifact `${MODEL_DIR}/baseline-<version>.json` (version = UTC timestamp
`YYYYMMDDTHHMMSSffffffZ`, lexically sortable; `score` picks the newest) contains `version`,
`created_at`, `features`, `weights`, `vol_penalty`, `feature_stats`, and `training`
(rows, symbols, date range, weight mode, and eval `metrics`).

## Evaluation

`train` reports forward-return (default 5-day) **rank IC** — per-date Spearman correlation
between score and forward return — as in-sample and on a time-based holdout (last 20% of dates,
model refit on the first 80%). Summary: mean, std, IC IR, hit rate, number of dates.

## Recommendations output

One row per scorable symbol inserted into `recommendations`
(`symbol, score, rationale, model_version, as_of`), where `as_of` = max price date loaded and
rationale looks like
`20d momentum +4.2%, 60d momentum +9.8%, 5d +1.1%, vol 28% ann., RSI 61, 3.0% above 50d avg`.
Rows are appended; the backend serves the latest `as_of` + latest `model_version`.

## Docker

```bash
docker build -t optrade-ds .
docker run --rm -e DATABASE_URL=postgresql://optrade:optrade@host.docker.internal:5432/optrade \
  -v "$PWD/artifacts:/app/artifacts" optrade-ds          # default CMD: all
docker run --rm ... optrade-ds train
```

In compose (owned by `optrade-platform`) this runs as `ds-train` under the `jobs` profile.

## Development

```bash
uv run ruff check .
uv run pytest -q
```
