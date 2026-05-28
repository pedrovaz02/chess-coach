"""Precompute recommendations.json for the web API.

Run this once after training (cluster.py). The web backend then just does a
dict lookup at request time instead of filtering 28k games per call.

Output structure:
    {
      "k": 5,
      "feature_columns": [...],
      "clusters": [
        {
          "id": 0,
          "name": "Blitz brawler",
          "blurb": "Lives in time pressure ...",
          "size": 39,
          "avg_rating": 2008,
          "feature_means": {feature: value, ...},
          "top_openings": {
            "white": [{"eco": "B30", "name": "...", "n": 30, "score_residual": 0.04}, ...],
            "black": [...]
          }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from rich.console import Console

from chess_coach.features import FEATURE_COLUMNS
from chess_coach.recommender import opening_rankings


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# Cluster names are tied to the K=5 fit with the current seed. If you retrain
# with different K or seed, regenerate these by inspecting the cluster summary
# heatmap (see notebooks/01_data_exploration.ipynb).
CLUSTER_PROFILES: dict[int, dict[str, str]] = {
    0: {
        "name": "Quick 1.e4 amateur",
        "blurb": "Plays 1.e4 84% of the time, decides games fast — 30% finish "
                 "before move 20 per side. Tactical short games dominate.",
    },
    1: {
        "name": "Underrated 1.e4 overperformer",
        "blurb": "Outperforming Elo by +0.11 score per game (and +0.15 in long "
                 "games). Classical 1.e4 player whose rating hasn't caught up.",
    },
    2: {
        "name": "1.e4 grinder",
        "blurb": "Higher-rated 1.e4 specialist (~1790). Longest games in the "
                 "dataset (75 moves avg), Sicilian defender as Black.",
    },
    3: {
        "name": "Queenside king-hunter",
        "blurb": "Aggressive attacking style: 41% queenside castles, castles "
                 "late (avg move 14), 38% mate rate — the highest. Goes for "
                 "the king.",
    },
    4: {
        "name": "1.d4 specialist",
        "blurb": "Closed-position devotee. Only 6% 1.e4 — almost exclusively "
                 "plays 1.d4 (48%) as White. Solid, positional style.",
    },
}


# Shrinkage strength + min sample for the head-to-head residual table.
# Lower min than the display recommendations (5000) so more families have
# entries for both colours, giving more matchup overlap.
_VS_SHRINKAGE_K = 30_000
_VS_MIN_GAMES = 800


def _family_residual_table(
    games: pl.DataFrame, clustered: pl.DataFrame, cluster: int
) -> dict[str, dict[str, float]]:
    """Per-(colour, family) shrunk score residual for one cluster.

    No colour-appropriate filtering (unlike the display recommendations):
    we want White *and* Black residuals for every family so a head-to-head
    can ask "how does cluster A do as White in family F vs how does cluster
    B do as Black in F". Returns {"white": {family: resid}, "black": {...}}.
    """
    members = clustered.filter(pl.col("cluster") == cluster).select("username")
    expected = 1.0 / (
        1.0 + (10.0 ** ((pl.col("opponent_rating") - pl.col("user_rating")) / 400.0))
    )
    actual = (
        pl.when(pl.col("result") == "win").then(1.0)
        .when(pl.col("result") == "draw").then(0.5)
        .otherwise(0.0)
    )
    family = pl.col("opening_name").str.split(":").list.get(0).str.strip_chars()

    out: dict[str, dict[str, float]] = {}
    base = (
        games.join(members, on="username", how="inner")
        .filter(pl.col("opening_name").is_not_null())
        .with_columns(score_diff=(actual - expected), family=family)
    )
    for color in ("white", "black"):
        agg = (
            base.filter(pl.col("color") == color)
            .group_by("family")
            .agg(n=pl.len(), raw=pl.col("score_diff").mean())
            .filter(pl.col("n") >= _VS_MIN_GAMES)
            .with_columns(
                resid=pl.col("n") / (pl.col("n") + _VS_SHRINKAGE_K) * pl.col("raw")
            )
        )
        out[color] = {
            row["family"]: round(float(row["resid"]), 4)
            for row in agg.iter_rows(named=True)
        }
    return out


def build_recommendations(
    games: pl.DataFrame,
    clustered: pl.DataFrame,
    top_n: int,
    min_opening_games: int,
) -> dict:
    cluster_ids = sorted(clustered["cluster"].unique().to_list())

    clusters_out = []
    for cid in cluster_ids:
        members = clustered.filter(pl.col("cluster") == cid)
        feature_means = {
            col: float(members[col].mean()) for col in FEATURE_COLUMNS
        }

        white = opening_rankings(
            games, clustered, cid, "white",
            min_games_in_opening=min_opening_games, top_n=top_n,
        )
        black = opening_rankings(
            games, clustered, cid, "black",
            min_games_in_opening=min_opening_games, top_n=top_n,
        )

        def serialise(df: pl.DataFrame) -> list[dict]:
            return [
                {
                    "name": row["name"],
                    "n": int(row["n"]),
                    "score_residual": float(row["score_residual"]),
                }
                for row in df.iter_rows(named=True)
            ]

        # Accuracy stats for this cluster — only over players with enough
        # analysed games for a stable per-player ACPL (>= 3). May be absent
        # if the dataset has no eval data (e.g. API-collected).
        accuracy = None
        if "acpl" in members.columns and "n_analyzed_games" in members.columns:
            analyzed = members.filter(pl.col("n_analyzed_games") >= 3)
            if analyzed.height > 0:
                accuracy = {
                    "avg_acpl": float(analyzed["acpl"].mean()),
                    "avg_blunder_rate": float(analyzed["blunder_rate"].mean()),
                    "n_players": int(analyzed.height),
                }

        profile = CLUSTER_PROFILES.get(
            cid, {"name": f"Cluster {cid}", "blurb": ""}
        )
        clusters_out.append({
            "id": cid,
            "name": profile["name"],
            "blurb": profile["blurb"],
            "size": int(members.height),
            "avg_rating": float(members["avg_rating"].mean()),
            "accuracy": accuracy,
            "family_residuals": _family_residual_table(games, clustered, cid),
            "feature_means": feature_means,
            "top_openings": {
                "white": serialise(white),
                "black": serialise(black),
            },
        })

    return {
        "k": len(cluster_ids),
        "feature_columns": FEATURE_COLUMNS,
        "clusters": clusters_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute recommendations.json")
    parser.add_argument("--games", type=Path, default=DATA_DIR / "games.parquet")
    parser.add_argument(
        "--clustered", type=Path, default=DATA_DIR / "players_clustered.parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "recommendations.json"
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-opening-games", type=int, default=5000)
    args = parser.parse_args()

    console = Console()
    games = pl.read_parquet(args.games)
    clustered = pl.read_parquet(args.clustered)
    console.print(
        f"Loaded {games.height:,} games and {clustered.height} clustered players"
    )

    payload = build_recommendations(
        games, clustered, top_n=args.top_n, min_opening_games=args.min_opening_games
    )
    args.output.write_text(json.dumps(payload, indent=2))
    console.print(f"Wrote [bold]{args.output}[/bold] (K={payload['k']})")

    for c in payload["clusters"]:
        nw = len(c["top_openings"]["white"])
        nb = len(c["top_openings"]["black"])
        console.print(
            f"  C{c['id']} {c['name']}: {nw} White / {nb} Black openings"
        )


if __name__ == "__main__":
    main()
