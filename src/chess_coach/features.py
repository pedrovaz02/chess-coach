"""Aggregate the games dataset into a per-player playstyle feature matrix.

Input:  data/games.parquet   (one row per game)
Output: data/features.parquet (one row per player, with the columns below)

Features (what we cluster on):
    score_residual        actual score (1/0.5/0) minus expected score from Elo
                          across all games. This is the skill-adjusted measure:
                          +0.05 means "you over-perform your rating by 5%",
                          0 means "you perform exactly at your rating", and is
                          orthogonal to raw strength.
    draw_rate             draws / total games
    avg_moves             mean ply count per game (long = positional grinder,
                          short = sharp tactical)
    white_score_residual  score_residual restricted to games as White
    black_score_residual  score_residual restricted to games as Black
    opening_diversity     unique ECO codes / total games (0 = narrow rep,
                          1 = never repeats)
    timeout_rate          % games ending in flag fall (time management)
    resign_rate           % games ending in resignation (fighting spirit)

Why skill adjustment matters:
    A 2700 player beats a 1500 player ~99% of the time regardless of style.
    Raw win_rate would put all GMs in one cluster and all amateurs in another,
    which is just measuring rating. score_residual removes that effect by
    comparing actual results to Elo's prediction.

Metadata (kept but not fed to model):
    n_games, avg_rating
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from rich.console import Console


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FEATURE_COLUMNS = [
    "score_residual",
    "draw_rate",
    "avg_moves",
    "white_score_residual",
    "black_score_residual",
    "opening_diversity",
    "timeout_rate",
    "resign_rate",
]


def _annotate_games(games: pl.DataFrame) -> pl.DataFrame:
    """Add per-game derived columns: actual_score and expected_score (Elo).

    actual_score: win=1, draw=0.5, loss=0  (standard chess scoring)
    expected_score: 1 / (1 + 10^((opp - user) / 400))  (Elo formula)
    """
    return games.with_columns(
        actual_score=pl.when(pl.col("result") == "win")
        .then(1.0)
        .when(pl.col("result") == "draw")
        .then(0.5)
        .otherwise(0.0),
        expected_score=1.0
        / (
            1.0
            + (10.0 ** ((pl.col("opponent_rating") - pl.col("user_rating")) / 400.0))
        ),
    ).with_columns(score_diff=pl.col("actual_score") - pl.col("expected_score"))


def build_player_features(games: pl.DataFrame, min_games: int = 20) -> pl.DataFrame:
    """Collapse a games table into one row per player.

    Players with fewer than `min_games` are dropped — too noisy to cluster on.
    """
    g = _annotate_games(games)

    is_white = pl.col("color") == "white"
    is_black = pl.col("color") == "black"
    white_n = is_white.sum()
    black_n = is_black.sum()

    features = (
        g.group_by("username")
        .agg(
            n_games=pl.len(),
            avg_rating=pl.col("user_rating").mean(),
            # Skill-adjusted: mean of (actual - expected) score across all games.
            score_residual=pl.col("score_diff").mean(),
            draw_rate=(pl.col("result") == "draw").mean(),
            avg_moves=pl.col("n_moves").mean(),
            white_score_residual=pl.when(white_n > 0)
            .then(pl.col("score_diff").filter(is_white).mean())
            .otherwise(None),
            black_score_residual=pl.when(black_n > 0)
            .then(pl.col("score_diff").filter(is_black).mean())
            .otherwise(None),
            opening_diversity=pl.col("opening_eco").n_unique() / pl.len(),
            timeout_rate=(pl.col("status") == "outoftime").mean(),
            resign_rate=(pl.col("status") == "resign").mean(),
        )
        .filter(pl.col("n_games") >= min_games)
        .with_columns(
            # If a player never played a colour, use overall residual.
            pl.col("white_score_residual").fill_null(pl.col("score_residual")),
            pl.col("black_score_residual").fill_null(pl.col("score_residual")),
        )
        .sort("username")
    )

    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-player feature matrix")
    parser.add_argument(
        "--games", type=Path, default=DATA_DIR / "games.parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "features.parquet"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Drop players with fewer than this many games",
    )
    args = parser.parse_args()

    console = Console()
    console.print(f"Reading [bold]{args.games}[/bold]...")
    games = pl.read_parquet(args.games)
    console.print(f"  {games.height:,} games, {games['username'].n_unique()} players")

    features = build_player_features(games, min_games=args.min_games)
    console.print(
        f"\nFeature matrix: [green]{features.height}[/green] players "
        f"× {len(FEATURE_COLUMNS)} features "
        f"(dropped {games['username'].n_unique() - features.height} below min_games={args.min_games})"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(args.output)
    console.print(f"Saved to [bold]{args.output}[/bold]")

    console.print("\n[bold]Summary stats:[/bold]")
    summary = features.select(FEATURE_COLUMNS).describe()
    console.print(summary)


if __name__ == "__main__":
    main()
