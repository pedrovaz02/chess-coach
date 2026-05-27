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
import json
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


# Opening → color classifier. Built once from the Lichess openings TSVs by
# `scripts/build_openings_db.py`: each opening is classified by whose move
# was last in its canonical PGN. This is more accurate than a name regex —
# "Sicilian Defense: Najdorf, English Attack" is correctly White-led even
# though "Defense" is in the name, because Be3 is the defining last move.
_OPENINGS_PATH = Path(__file__).parent / "openings.json"
_OPENINGS_DATA: dict[str, str] | None = None


def _load_openings() -> dict[str, str]:
    global _OPENINGS_DATA
    if _OPENINGS_DATA is None:
        _OPENINGS_DATA = json.loads(_OPENINGS_PATH.read_text())
    return _OPENINGS_DATA


# Regex fallback for the ~0.6% of opening names not in the classifier
# (deeper sub-variations that appear in our data but not in the canonical TSV).
_FALLBACK_BLACK_PATTERNS = (
    "defense", "defence", "indian", "marshall attack", "schliemann",
    "albin counter", "benoni", "benko", "grünfeld", "grunfeld",
)
_FALLBACK_REGEX = "(?i)" + "|".join(_FALLBACK_BLACK_PATTERNS)


def opening_rankings(
    games: pl.DataFrame,
    clustered_players: pl.DataFrame,
    cluster: int,
    color: str,
    min_games_in_opening: int = 30,
    top_n: int = 10,
) -> pl.DataFrame:
    """Top openings for a (cluster, color), ranked by skill-adjusted score
    (mean of actual - Elo-expected) across all games of cluster members in
    that opening. Filtered to openings with >= min_games_in_opening samples.

    Color-appropriate filter:
        For color="white", we exclude openings characterised as Black's choice
        (defenses, Indian setups, Marshall, etc.) — White can't choose what
        Black plays against them.

        For color="black", we keep only openings characterised as Black's
        choice — these are responses Black can actually steer toward.

    Why skill-adjusted instead of raw win rate:
        Two openings can have the same raw win rate while one is played
        against weaker opposition. Score residual normalises for that.
    """
    members = clustered_players.filter(pl.col("cluster") == cluster).select("username")

    expected = 1.0 / (
        1.0 + (10.0 ** ((pl.col("opponent_rating") - pl.col("user_rating")) / 400.0))
    )
    actual = (
        pl.when(pl.col("result") == "win")
        .then(1.0)
        .when(pl.col("result") == "draw")
        .then(0.5)
        .otherwise(0.0)
    )

    # Color classification: lookup in opening DB first, regex as fallback.
    openings_db = _load_openings()
    # Build a polars expression: opening_name → "white" | "black" | None
    # We do this via a Python-side mapping converted to a polars literal lookup.
    classified_color = (
        pl.col("opening_name")
        .replace_strict(openings_db, default=None, return_dtype=pl.String)
    )
    # Fallback regex for names not in the DB
    fallback_is_black = pl.col("opening_name").str.contains(_FALLBACK_REGEX)
    fallback_color = (
        pl.when(fallback_is_black).then(pl.lit("black")).otherwise(pl.lit("white"))
    )
    effective_color = pl.when(classified_color.is_not_null()).then(classified_color).otherwise(fallback_color)
    color_filter = effective_color == color

    return (
        games.join(members, on="username", how="inner")
        .filter(
            (pl.col("color") == color)
            & pl.col("opening_eco").is_not_null()
            & pl.col("opening_name").is_not_null()
            & color_filter
        )
        .with_columns(score_diff=(actual - expected))
        .group_by(["opening_eco", "opening_name"])
        .agg(
            n=pl.len(),
            score_residual=pl.col("score_diff").mean(),
        )
        .filter(pl.col("n") >= min_games_in_opening)
        .sort("score_residual", descending=True)
        .head(top_n)
    )


def render_recommendations(
    console: Console,
    username: str,
    cluster: int,
    white: pl.DataFrame,
    black: pl.DataFrame,
    verbose: bool = False,
) -> None:
    console.print(
        f"\n[bold green]{username}[/bold green] -> cluster [bold]{cluster}[/bold]"
    )
    console.print(
        "[dim]Openings ranked by how well players with your style profile "
        "have historically done with them.[/dim]\n"
    )

    for color, df in (("White", white), ("Black", black)):
        if df.is_empty():
            console.print(f"[yellow]No openings as {color} cleared the sample-size threshold.[/yellow]")
            continue

        table = Table(title=f"Suggested openings as {color}", title_justify="left")
        table.add_column("#", style="bold")
        table.add_column("ECO")
        table.add_column("Opening")
        table.add_column("Support", justify="right", style="dim")
        if verbose:
            table.add_column("Score res.", justify="right", style="dim")

        for rank, row in enumerate(df.iter_rows(named=True), start=1):
            cells = [
                str(rank),
                row["opening_eco"],
                row["opening_name"],
                f"{row['n']} games",
            ]
            if verbose:
                cells.append(f"{row['score_residual']:+.3f}")
            table.add_row(*cells)
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend openings for a Lichess user")
    parser.add_argument("username", help="Lichess username")
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-opening-games", type=int, default=30)
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show internal scoring (score residual). Off by default to keep "
             "the UX honest — see README for why."
    )
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
    render_recommendations(
        console, args.username, cluster, white, black, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
