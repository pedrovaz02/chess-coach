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
        "name": "Blitz brawler",
        "blurb": "High mate rate, frequent time losses. Plays fast and "
                 "aggressively, lets the clock decide many games.",
    },
    1: {
        "name": "Balanced amateur",
        "blurb": "Lower rating band, no strong stylistic extremes. "
                 "Plays the position, doesn't force the game.",
    },
    2: {
        "name": "Sicilian devotee",
        "blurb": "When facing 1.e4 as Black, plays the Sicilian 90% of the "
                 "time. Sharp, asymmetrical positions.",
    },
    3: {
        "name": "Underrated overperformer",
        "blurb": "Outperforming the Elo expectation by ~12 score points "
                 "per game. Rating likely catching up to actual strength.",
    },
    4: {
        "name": "1.d4 specialist",
        "blurb": "Queen's-pawn player. ~42% of your White games start 1.d4 "
                 "(vs ~6% for the average player). Solid, positional.",
    },
}


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
                    "eco": row["opening_eco"],
                    "name": row["opening_name"],
                    "n": int(row["n"]),
                    "score_residual": float(row["score_residual"]),
                }
                for row in df.iter_rows(named=True)
            ]

        profile = CLUSTER_PROFILES.get(
            cid, {"name": f"Cluster {cid}", "blurb": ""}
        )
        clusters_out.append({
            "id": cid,
            "name": profile["name"],
            "blurb": profile["blurb"],
            "size": int(members.height),
            "avg_rating": float(members["avg_rating"].mean()),
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
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--min-opening-games", type=int, default=15)
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
