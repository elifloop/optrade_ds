"""``optrade-ds`` command line: train | score | all."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from optrade_ds.config import Settings
from optrade_ds.data.loader import load_prices_csv, load_prices_db
from optrade_ds.data.writer import PostgresRecommendationWriter
from optrade_ds.pipelines.score import score_pipeline
from optrade_ds.pipelines.train import train_pipeline

log = logging.getLogger("optrade_ds")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        prices = load_prices_csv(args.csv)
    else:
        prices = load_prices_db(args.database_url, args.symbols)
    if args.symbols:
        prices = prices[prices["symbol"].isin([s.upper() for s in args.symbols])]
    log.info("loaded %d price rows for %d symbols", len(prices), prices["symbol"].nunique())
    return prices


def _train(args: argparse.Namespace, prices: pd.DataFrame) -> None:
    model, path = train_pipeline(
        prices, args.model_dir, learn_weights=args.learn_weights, horizon=args.horizon
    )
    print(f"trained model {model.version} -> {path}")
    print(json.dumps(model.training.get("metrics", {}), indent=2))


def _score(args: argparse.Namespace, prices: pd.DataFrame) -> None:
    writer = None if args.dry_run or args.csv else PostgresRecommendationWriter(args.database_url)
    recs = score_pipeline(prices, args.model_dir, writer, artifact=args.artifact)
    if recs.empty:
        print("no recommendations produced (insufficient history?)")
        return
    action = "would insert" if writer is None else "inserted"
    print(f"{action} {len(recs)} recommendations (model {recs['model_version'].iloc[0]}, "
          f"as_of {recs['as_of'].iloc[0]}):")
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(recs.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--database-url", default=settings.database_url,
                        help="Postgres URL (default: $DATABASE_URL)")
    common.add_argument("--model-dir", default=settings.model_dir,
                        help="artifact directory (default: $MODEL_DIR or ./artifacts)")
    common.add_argument("--csv", help="load prices from CSV (symbol,date,close,...) instead of DB;"
                        " implies --dry-run for scoring")
    common.add_argument("--symbols", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                        help="comma separated subset of symbols")
    common.add_argument("-v", "--verbose", action="store_true")

    train_opts = argparse.ArgumentParser(add_help=False)
    train_opts.add_argument("--learn-weights", action="store_true",
                            help="learn feature weights from rank IC instead of configured defaults")
    train_opts.add_argument("--horizon", type=int, default=5, help="forward-return horizon (days)")

    score_opts = argparse.ArgumentParser(add_help=False)
    score_opts.add_argument("--artifact", help="explicit artifact path (default: latest)")
    score_opts.add_argument("--dry-run", action="store_true", help="print but do not write to DB")

    p = argparse.ArgumentParser(prog="optrade-ds", description="optrade data science pipelines")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("train", parents=[common, train_opts], help="fit and save a baseline model")
    sub.add_parser("score", parents=[common, score_opts], help="score latest prices with latest model")
    sub.add_parser("all", parents=[common, train_opts, score_opts], help="train then score")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = "DEBUG" if args.verbose else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        prices = _load(args)
        if args.command in ("train", "all"):
            _train(args, prices)
        if args.command in ("score", "all"):
            if args.command == "all":
                args.artifact = None
            _score(args, prices)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report cleanly
        log.error("%s failed: %s", args.command, exc)
        if args.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
