"""Precompute the serving artifacts for the kNN (continuum) recommender.

The live API can't load the 784 MB games file, so we precompute everything the
neighbourhood recommender needs at request time:

  data/models/knn_matrix.npy     scaled 18-feature matrix (N x 18, float32)
  data/knn_player_meta.parquet   row-aligned username + avg_rating
  data/knn_family_stats.parquet  per (player, colour, family): n, sum_score_diff

At request time the API scales the query, finds the k nearest rows of the
matrix, takes those usernames, filters the family-stats to them, sums n and
sum_score_diff per (colour, family), and applies the same shrinkage + ranking
as the offline recommender. No games file required.

Only the ~154k *indexed* players (those with a feature vector) appear in the
stats table — neighbours can only ever be drawn from the index — which keeps
the artifact small.

Run:
    uv run python scripts/build_knn_index.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_coach.recommender import effective_color_expr, family_expr  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    ap.add_argument("--games", type=Path, default=DATA_DIR / "games.parquet")
    ap.add_argument(
        "--recommendations", type=Path, default=DATA_DIR / "recommendations.json"
    )
    ap.add_argument("--scaler", type=Path, default=MODELS_DIR / "scaler.joblib")
    args = ap.parse_args()
    console = Console()

    feature_columns = json.loads(args.recommendations.read_text())["feature_columns"]
    scaler = joblib.load(args.scaler)
    feats = pl.read_parquet(args.features)
    console.print(f"Loaded {feats.height:,} indexed players, {len(feature_columns)} features")

    # ── 1. Scaled matrix + aligned meta ─────────────────────────────────────
    matrix = scaler.transform(feats.select(feature_columns).to_numpy()).astype(np.float32)
    np.save(MODELS_DIR / "knn_matrix.npy", matrix)
    feats.select(["username", "avg_rating"]).write_parquet(
        DATA_DIR / "knn_player_meta.parquet"
    )
    console.print(
        f"Wrote knn_matrix.npy {matrix.shape} "
        f"({matrix.nbytes/1e6:.1f} MB) + knn_player_meta.parquet"
    )

    # ── 2. Per-(player, colour, family) score-residual building blocks ───────
    index_users = feats.select("username")
    games = pl.read_parquet(
        args.games,
        columns=["username", "color", "result", "user_rating",
                 "opponent_rating", "opening_name"],
    )
    actual = (
        pl.when(pl.col("result") == "win").then(1.0)
        .when(pl.col("result") == "draw").then(0.5)
        .otherwise(0.0)
    )
    expected = 1.0 / (
        1.0 + 10.0 ** ((pl.col("opponent_rating") - pl.col("user_rating")) / 400.0)
    )

    # Keep only colour-appropriate rows (player's colour == opening's owner),
    # classified per full name with the SAME expression the games-based ranker
    # uses. Doing the filter here means the API just pools the survivors and
    # gets a byte-identical answer to the offline recommender.
    stats = (
        games.join(index_users, on="username", how="inner")
        .filter(pl.col("opening_name").is_not_null())
        .with_columns(
            family=family_expr(),
            sd=(actual - expected),
            effective_color=effective_color_expr(),
        )
        .filter(pl.col("color") == pl.col("effective_color"))
        .group_by(["username", "color", "family"])
        .agg(n=pl.len(), sum_sd=pl.col("sd").sum())
    )
    stats.write_parquet(DATA_DIR / "knn_family_stats.parquet")
    size_mb = (DATA_DIR / "knn_family_stats.parquet").stat().st_size / 1e6
    console.print(
        f"Wrote knn_family_stats.parquet: {stats.height:,} rows "
        f"({stats['username'].n_unique():,} players) — {size_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
