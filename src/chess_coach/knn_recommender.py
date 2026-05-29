"""Continuum-native opening recommender: k-nearest players in style space.

Motivation
----------
The K-Means recommender (`recommender.py`) snaps a player to one of five hard
clusters, then recommends that cluster's best openings. But we established the
playstyle space is a *continuum*, not five discrete blobs (see DECISIONS § 4.3):
silhouette ~0.08, flat GMM BIC, HDBSCAN finds no density groups. Hard clustering
imposes boundaries the data doesn't have — two stylistically identical players
can land in different clusters if they sit either side of a centroid boundary.

This recommender drops the boundaries. Given a query player it finds the *k*
nearest players in the same 18-dim scaled style space, pools *their* games, and
ranks openings by the same skill-adjusted residual + Bayesian shrinkage. The
cohort is now a smooth neighbourhood around the query rather than a fixed cell —
which is what a continuum actually warrants.

It deliberately reuses the production artifacts (scaler, 18-feature order) so it
is a true drop-in alternative to the cluster lookup, directly comparable on the
same player.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rich.console import Console

from chess_coach.features import build_player_features
from chess_coach.recommender import (
    fetch_user_games_df,
    rank_openings_for_members,
    render_recommendations,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = DATA_DIR / "models"


@dataclass
class KnnIndex:
    """The scaled training matrix plus aligned player metadata."""

    feature_columns: list[str]
    usernames: np.ndarray  # (N,) str
    avg_rating: np.ndarray  # (N,) float
    matrix: np.ndarray  # (N, D) float32, already scaled
    scaler: object

    @property
    def n_players(self) -> int:
        return self.matrix.shape[0]


def load_index(
    features_path: Path = DATA_DIR / "features.parquet",
    recommendations_path: Path = DATA_DIR / "recommendations.json",
    scaler_path: Path = MODELS_DIR / "scaler.joblib",
) -> KnnIndex:
    """Build the kNN index from the production artifacts.

    The feature order is read from recommendations.json — the authoritative
    list the saved scaler was fit on — so this stays correct even though the
    working-tree FEATURE_COLUMNS may carry experimental extra features.
    """
    feature_columns = json.loads(recommendations_path.read_text())["feature_columns"]
    scaler = joblib.load(scaler_path)
    feats = pl.read_parquet(features_path)

    missing = [c for c in feature_columns if c not in feats.columns]
    if missing:
        raise ValueError(f"features file is missing production columns: {missing}")

    matrix = scaler.transform(feats.select(feature_columns).to_numpy()).astype(np.float32)
    return KnnIndex(
        feature_columns=feature_columns,
        usernames=feats["username"].to_numpy(),
        avg_rating=feats["avg_rating"].to_numpy(),
        matrix=matrix,
        scaler=scaler,
    )


def nearest_indices(
    query_scaled: np.ndarray, matrix: np.ndarray, k: int, exclude: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the k rows of `matrix` closest to `query_scaled` (Euclidean).

    `exclude` drops a specific row (leave-one-out, when the query *is* a
    training player). Returns (indices, distances) sorted nearest-first.
    """
    dists = np.linalg.norm(matrix - query_scaled, axis=1)
    if exclude is not None:
        dists[exclude] = np.inf
    take = min(k, np.isfinite(dists).sum())
    part = np.argpartition(dists, take - 1)[:take]
    order = part[np.argsort(dists[part])]
    return order, dists[order]


def recommend(
    index: KnnIndex,
    query_scaled: np.ndarray,
    games: pl.DataFrame,
    *,
    k: int = 2000,
    top_n: int = 5,
    min_games_in_opening: int = 100,
    shrinkage_k: int = 2000,
    exclude_idx: int | None = None,
) -> dict:
    """k-NN opening recommendations for one scaled query vector.

    Defaults are sized for a ~2000-player neighbourhood (~1.3% of the 154k
    population — local, but large enough for stable family samples). At that
    scale mainstream families gather a few thousand games, so shrinkage_k=2000
    weights them at >=50% while suppressing rare gambits, and the 100-game
    floor keeps a family out until there's real evidence. A smaller k tracks
    the individual more tightly but lets offbeat openings dominate (validated
    empirically: k=500 surfaced Van Geet/Van't Kruijs noise that k=2000 drops).
    """
    idx, dists = nearest_indices(query_scaled, index.matrix, k, exclude=exclude_idx)
    neighbour_names = index.usernames[idx]
    members = pl.DataFrame({"username": neighbour_names})

    white = rank_openings_for_members(
        games, members, "white",
        min_games_in_opening=min_games_in_opening, top_n=top_n, shrinkage_k=shrinkage_k,
    )
    black = rank_openings_for_members(
        games, members, "black",
        min_games_in_opening=min_games_in_opening, top_n=top_n, shrinkage_k=shrinkage_k,
    )
    return {
        "neighbours": neighbour_names,
        "neighbour_ratings": index.avg_rating[idx],
        "distances": dists,
        "white": white,
        "black": black,
    }


# ── Serving path (no games file) ────────────────────────────────────────────
# The live API can't hold the 784 MB games table. `build_knn_index.py`
# precomputes per-(player, colour, family) building blocks (n, sum_sd); here we
# pool them over the kNN neighbourhood and apply the identical shrinkage +
# ranking. Produces the same answer as the games path, just from a 33 MB table.


@dataclass
class ServingIndex:
    """Lightweight artifacts loaded once at API startup."""

    feature_columns: list[str]
    usernames: np.ndarray
    avg_rating: np.ndarray
    matrix: np.ndarray
    scaler: object
    family_stats: pl.DataFrame  # username, color, family, n, sum_sd


def load_serving_index(
    recommendations_path: Path = DATA_DIR / "recommendations.json",
    scaler_path: Path = MODELS_DIR / "scaler.joblib",
    matrix_path: Path = MODELS_DIR / "knn_matrix.npy",
    meta_path: Path = DATA_DIR / "knn_player_meta.parquet",
    stats_path: Path = DATA_DIR / "knn_family_stats.parquet",
) -> ServingIndex:
    feature_columns = json.loads(recommendations_path.read_text())["feature_columns"]
    meta = pl.read_parquet(meta_path)
    return ServingIndex(
        feature_columns=feature_columns,
        usernames=meta["username"].to_numpy(),
        avg_rating=meta["avg_rating"].to_numpy(),
        matrix=np.load(matrix_path),
        scaler=joblib.load(scaler_path),
        family_stats=pl.read_parquet(stats_path),
    )


def recommend_served(
    index: ServingIndex,
    query_scaled: np.ndarray,
    *,
    k: int = 2000,
    top_n: int = 5,
    min_games_in_opening: int = 100,
    shrinkage_k: int = 2000,
    exclude_idx: int | None = None,
) -> dict:
    """kNN recommendations from precomputed stats (matches `recommend`)."""
    idx, dists = nearest_indices(query_scaled, index.matrix, k, exclude=exclude_idx)
    neighbour_names = index.usernames[idx].tolist()

    # family_stats is already colour-filtered at build time (only rows where the
    # player's colour matches the opening's owner survive), so we just pool.
    pooled = (
        index.family_stats.filter(pl.col("username").is_in(neighbour_names))
        .group_by(["color", "family"])
        .agg(n=pl.col("n").sum(), sum_sd=pl.col("sum_sd").sum())
        .filter(pl.col("n") >= min_games_in_opening)
        .with_columns(
            score_residual=pl.col("n") / (pl.col("n") + shrinkage_k)
            * (pl.col("sum_sd") / pl.col("n"))
        )
    )

    def top(color: str) -> list[dict]:
        df = (
            pooled.filter(pl.col("color") == color)
            .sort("score_residual", descending=True)
            .head(top_n)
        )
        return [
            {"name": r["family"], "n": int(r["n"]),
             "score_residual": float(r["score_residual"])}
            for r in df.iter_rows(named=True)
        ]

    ratings = index.avg_rating[idx]
    return {
        "white": top("white"),
        "black": top("black"),
        "k": len(neighbour_names),
        "neighbour_rating_min": float(ratings.min()),
        "neighbour_rating_max": float(ratings.max()),
        "neighbour_rating_mean": float(ratings.mean()),
    }


def _query_from_username(
    index: KnnIndex, username: str, max_games: int
) -> tuple[np.ndarray, int | None]:
    """Build a scaled query vector for a live Lichess user.

    If the username is already in the training index we return its row and an
    exclude index (leave-one-out) so a player isn't recommended their own games.
    """
    hit = np.where(index.usernames == username)[0]
    if hit.size:
        i = int(hit[0])
        return index.matrix[i : i + 1], i

    user_games = fetch_user_games_df(username, max_games)
    feats = build_player_features(user_games, min_games=1)
    X = feats.select(index.feature_columns).to_numpy()
    return index.scaler.transform(X).astype(np.float32), None


def main() -> None:
    ap = argparse.ArgumentParser(description="kNN (continuum) opening recommender")
    ap.add_argument("username", help="Lichess username (live fetch unless --like)")
    ap.add_argument(
        "--like", action="store_true",
        help="Treat username as an existing training player; use its stored "
             "vector (offline, no Lichess call) with leave-one-out.",
    )
    ap.add_argument("--k", type=int, default=2000, help="neighbours to pool")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=100)
    ap.add_argument("--min-opening-games", type=int, default=100)
    ap.add_argument("--shrinkage-k", type=int, default=2000)
    ap.add_argument("--games", type=Path, default=DATA_DIR / "games.parquet")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    console = Console()
    console.print("Loading style index + games ...")
    index = load_index()
    games = pl.read_parquet(args.games)

    if args.like:
        hit = np.where(index.usernames == args.username)[0]
        if not hit.size:
            console.print(f"[red]{args.username} not in training index.[/red]")
            return
        i = int(hit[0])
        query, exclude = index.matrix[i : i + 1], i
        console.print(
            f"[dim]Using stored vector for {args.username} "
            f"(rating {index.avg_rating[i]:.0f}), leave-one-out.[/dim]"
        )
    else:
        console.print(f"Fetching games for [bold]{args.username}[/bold] ...")
        query, exclude = _query_from_username(index, args.username, args.max_games)

    out = recommend(
        index, query, games,
        k=args.k, top_n=args.top_n,
        min_games_in_opening=args.min_opening_games,
        shrinkage_k=args.shrinkage_k, exclude_idx=exclude,
    )

    console.print(
        f"\n[bold green]{args.username}[/bold green] -> "
        f"{args.k} nearest players in style space "
        f"(rating range {out['neighbour_ratings'].min():.0f}"
        f"-{out['neighbour_ratings'].max():.0f}, "
        f"mean {out['neighbour_ratings'].mean():.0f}; "
        f"dist {out['distances'].min():.2f}-{out['distances'].max():.2f})"
    )
    console.print(
        "[dim]Openings ranked over the pooled games of your style "
        "neighbourhood — no hard cluster boundary.[/dim]"
    )
    render_recommendations(
        console, args.username, None, out["white"], out["black"], verbose=args.verbose
    )


if __name__ == "__main__":
    main()
