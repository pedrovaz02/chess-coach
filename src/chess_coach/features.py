"""Aggregate the games dataset into a per-player playstyle feature matrix.

Input:  data/games.parquet   (one row per game)
Output: data/features.parquet (one row per player, with the columns below)

Features (what we cluster on):
    score_residual        actual score (1/0.5/0) minus expected score from Elo
                          across all games. This is the skill-adjusted measure:
                          +0.05 means "you over-perform your rating by 5%",
                          0 means "you perform exactly at your rating", and is
                          orthogonal to raw strength.
    white_score_residual  score_residual restricted to games as White
    black_score_residual  score_residual restricted to games as Black
    draw_rate             draws / total games
    avg_moves             mean ply count per game (long = positional grinder,
                          short = sharp tactical)
    opening_diversity     unique ECO codes / total games (0 = narrow rep,
                          1 = never repeats)
    timeout_rate          % games ending in flag fall (time management)
    resign_rate           % games ending in resignation (fighting spirit)
    pct_e4_as_white       fraction of White games starting 1.e4 (ECO B/C)
    pct_d4_as_white       fraction of White games starting 1.d4 (ECO D/E)
    pct_sicilian_as_black against 1.e4 as Black, fraction playing Sicilian
                          (ECO B20-B99) — aggressive vs solid defenses
    mate_rate             games ending in checkmate (real sharpness, not the
                          opponent simply resigning early)
    short_game_rate       games shorter than 40 plies (~20 moves per side)
                          — tactical decisions over positional grinds

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
    "white_score_residual",
    "black_score_residual",
    "draw_rate",
    "avg_moves",
    "opening_diversity",
    "timeout_rate",
    "resign_rate",
    "pct_e4_as_white",
    "pct_d4_as_white",
    "pct_sicilian_as_black",
    "mate_rate",
    "short_game_rate",
]


def _annotate_games(games: pl.DataFrame) -> pl.DataFrame:
    """Add per-game derived columns:
        actual_score, expected_score, score_diff (Elo-based)
        eco_letter, eco_num (ECO code parsed into family + number)
    """
    eco_letter = pl.col("opening_eco").str.slice(0, 1)
    eco_num = pl.col("opening_eco").str.slice(1, 2).cast(pl.Int8, strict=False)

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
        eco_letter=eco_letter,
        eco_num=eco_num,
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

    # Opening family flags
    is_e4_opening = pl.col("eco_letter").is_in(["B", "C"])  # 1.e4 systems
    is_d4_opening = pl.col("eco_letter").is_in(["D", "E"])  # 1.d4 systems
    is_sicilian = (pl.col("eco_letter") == "B") & (pl.col("eco_num") >= 20)

    # Black-against-e4 games: opponent played e4, we're Black (ECO B or C)
    is_black_vs_e4 = is_black & is_e4_opening
    black_vs_e4_n = is_black_vs_e4.sum()

    features = (
        g.group_by("username")
        .agg(
            n_games=pl.len(),
            avg_rating=pl.col("user_rating").mean(),
            # ── Performance ───────────────────────────────────────────
            score_residual=pl.col("score_diff").mean(),
            white_score_residual=pl.when(white_n > 0)
            .then(pl.col("score_diff").filter(is_white).mean())
            .otherwise(None),
            black_score_residual=pl.when(black_n > 0)
            .then(pl.col("score_diff").filter(is_black).mean())
            .otherwise(None),
            # ── Result mix ────────────────────────────────────────────
            draw_rate=(pl.col("result") == "draw").mean(),
            mate_rate=(pl.col("status") == "mate").mean(),
            timeout_rate=(pl.col("status") == "outoftime").mean(),
            resign_rate=(pl.col("status") == "resign").mean(),
            # ── Game shape ────────────────────────────────────────────
            avg_moves=pl.col("n_moves").mean(),
            short_game_rate=(pl.col("n_moves") < 40).mean(),
            # ── Opening repertoire ────────────────────────────────────
            opening_diversity=pl.col("opening_eco").n_unique() / pl.len(),
            pct_e4_as_white=pl.when(white_n > 0)
            .then((is_white & is_e4_opening).sum() / white_n)
            .otherwise(None),
            pct_d4_as_white=pl.when(white_n > 0)
            .then((is_white & is_d4_opening).sum() / white_n)
            .otherwise(None),
            pct_sicilian_as_black=pl.when(black_vs_e4_n > 0)
            .then((is_black_vs_e4 & is_sicilian).sum() / black_vs_e4_n)
            .otherwise(None),
        )
        .filter(pl.col("n_games") >= min_games)
        .with_columns(
            pl.col("white_score_residual").fill_null(pl.col("score_residual")),
            pl.col("black_score_residual").fill_null(pl.col("score_residual")),
            # Players who never played a colour or never faced 1.e4 as Black:
            # fall back to dataset mean so they don't break clustering.
            pl.col("pct_e4_as_white").fill_null(
                pl.col("pct_e4_as_white").mean()
            ),
            pl.col("pct_d4_as_white").fill_null(
                pl.col("pct_d4_as_white").mean()
            ),
            pl.col("pct_sicilian_as_black").fill_null(
                pl.col("pct_sicilian_as_black").mean()
            ),
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
