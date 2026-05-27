"""End-to-end pipeline: Lichess username → opening recommendations.

Steps:
    1. Fetch the user's games (live, via Lichess API).
    2. Build their feature vector with the same code paths as training.
    3. Project into the trained K-Means space, get cluster label.
    4. Look up the openings that work best for that cluster — broken down by
       color, filtered to openings with enough sample size to be meaningful.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import polars as pl
from rich.console import Console
from rich.table import Table

from chess_coach.collector import fetch_games, game_to_row
from chess_coach.features import FEATURE_COLUMNS, build_player_features


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = DATA_DIR / "models"


def fetch_user_games_df(username: str, max_games: int = 100) -> pl.DataFrame:
    games = fetch_games(username, max_games)
    rows = [r for r in (game_to_row(g, username) for g in games) if r is not None]
    if not rows:
        raise ValueError(f"No games found for {username}")
    return pl.DataFrame(rows)


def predict_cluster(user_features: pl.DataFrame, scaler, model) -> int:
    X = user_features.select(FEATURE_COLUMNS).to_numpy()
    X_scaled = scaler.transform(X)
    return int(model.predict(X_scaled)[0])


def opening_rankings(
    games: pl.DataFrame,
    clustered_players: pl.DataFrame,
    cluster: int,
    color: str,
    min_games_in_opening: int = 30,
    top_n: int = 10,
) -> pl.DataFrame:
    """Top openings for a (cluster, color), by win rate across all players in
    that cluster, filtered to openings with >= min_games_in_opening samples.
    """
    members = clustered_players.filter(pl.col("cluster") == cluster).select("username")

    return (
        games.join(members, on="username", how="inner")
        .filter(
            (pl.col("color") == color)
            & pl.col("opening_eco").is_not_null()
        )
        .group_by(["opening_eco", "opening_name"])
        .agg(
            n=pl.len(),
            win_rate=(pl.col("result") == "win").mean(),
            draw_rate=(pl.col("result") == "draw").mean(),
        )
        .filter(pl.col("n") >= min_games_in_opening)
        .sort("win_rate", descending=True)
        .head(top_n)
    )


def render_recommendations(
    console: Console, username: str, cluster: int, white: pl.DataFrame, black: pl.DataFrame
) -> None:
    console.print(
        f"\n[bold green]{username}[/bold green] → cluster [bold]{cluster}[/bold]\n"
    )

    for color, df in (("White", white), ("Black", black)):
        table = Table(title=f"Top openings for cluster {cluster} as {color}")
        table.add_column("ECO")
        table.add_column("Opening")
        table.add_column("Games", justify="right")
        table.add_column("Win %", justify="right")
        table.add_column("Draw %", justify="right")
        for row in df.iter_rows(named=True):
            table.add_row(
                row["opening_eco"],
                row["opening_name"],
                str(row["n"]),
                f"{row['win_rate']*100:.1f}",
                f"{row['draw_rate']*100:.1f}",
            )
        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend openings for a Lichess user")
    parser.add_argument("username", help="Lichess username")
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-opening-games", type=int, default=30)
    parser.add_argument(
        "--games", type=Path, default=DATA_DIR / "games.parquet"
    )
    parser.add_argument(
        "--clustered", type=Path, default=DATA_DIR / "players_clustered.parquet"
    )
    args = parser.parse_args()

    console = Console()
    model = joblib.load(MODELS_DIR / "kmeans.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    games = pl.read_parquet(args.games)
    clustered = pl.read_parquet(args.clustered)

    console.print(f"Fetching games for [bold]{args.username}[/bold]...")
    user_games = fetch_user_games_df(args.username, args.max_games)
    console.print(f"  {user_games.height} games")

    user_features = build_player_features(user_games, min_games=1)
    cluster = predict_cluster(user_features, scaler, model)

    white = opening_rankings(
        games, clustered, cluster, "white",
        min_games_in_opening=args.min_opening_games, top_n=args.top_n,
    )
    black = opening_rankings(
        games, clustered, cluster, "black",
        min_games_in_opening=args.min_opening_games, top_n=args.top_n,
    )
    render_recommendations(console, args.username, cluster, white, black)


if __name__ == "__main__":
    main()
