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
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import polars as pl
from rich.console import Console

# Below this many unique games, parse single-threaded — spawning a worker
# pool costs ~1-2s of startup, not worth it for the live recommender (~100
# games). Above it (training runs of millions), parallelise across cores.
_PARALLEL_PARSE_THRESHOLD = 20_000


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FEATURE_COLUMNS = [
    # ── Performance (skill-adjusted) ──────────────────────────────────
    "score_residual",
    "white_score_residual",
    "black_score_residual",
    "long_game_residual",
    # ── Result mix ────────────────────────────────────────────────────
    "draw_rate",
    "mate_rate",
    "timeout_rate",
    "resign_rate",
    # ── Game shape ────────────────────────────────────────────────────
    "avg_moves",
    "short_game_rate",
    # ── Repertoire / opening choice ───────────────────────────────────
    "opening_diversity",
    "pct_e4_as_white",
    "pct_d4_as_white",
    # ── Black's defence choice vs 1.e4 (shares among Black-vs-e4 games) ─
    "pct_sicilian_as_black",
    "pct_french_as_black",
    "pct_carokann_as_black",
    "pct_e5_as_black",
    # ── Tier 1 move-level (parsed from PGN) ──────────────────────────
    "avg_castle_move",
    "pct_queenside_castle",
    "avg_queens_off_move",
    "early_pawn_pushes",
]


def _parse_move_features(moves_str: str) -> dict:
    """Parse a SAN move string and extract per-side move-level features.

    Returns a dict with eight keys covering castle timing, castle side,
    queens-off timing, and early pawn pushes — for both colors.
    Missing/invalid moves yield None values (skipped in aggregation).

    Detection rules:
        - Castle: token starts with "O-O". Queenside if "O-O-O".
        - Queens off: chess.popcount(board.queens) == 0 after a move.
        - Early pawn push: token in first 20 plies whose first char is
          lowercase (pawn moves like "e4", "exd5"); excludes castles.
    """
    out = {
        "white_castle_ply": None,
        "black_castle_ply": None,
        "white_castle_qside": False,
        "black_castle_qside": False,
        "queens_off_ply": None,
        "white_early_pawn": 0,
        "black_early_pawn": 0,
    }
    if not moves_str:
        return out

    tokens = moves_str.split()
    board = chess.Board()

    for i, san in enumerate(tokens, start=1):
        is_white_ply = (i % 2 == 1)
        first_char = san[0]

        # ── Castle ────────────────────────────────────────────────
        if san.startswith("O-O"):
            qside = san.startswith("O-O-O")
            if is_white_ply and out["white_castle_ply"] is None:
                out["white_castle_ply"] = i
                out["white_castle_qside"] = qside
            elif not is_white_ply and out["black_castle_ply"] is None:
                out["black_castle_ply"] = i
                out["black_castle_qside"] = qside

        # ── Early pawn push (in first 20 plies = 10 per side) ────
        # Pawn SAN starts with a lowercase file letter (a-h).
        # Castles ("O-O...") are handled above and excluded.
        elif i <= 20 and first_char.islower():
            if is_white_ply:
                out["white_early_pawn"] += 1
            else:
                out["black_early_pawn"] += 1

        # ── Push the move so we can check board state ────────────
        try:
            move = board.parse_san(san)
            board.push(move)
        except (ValueError, AssertionError):
            # Malformed PGN — bail out, keep what we have so far.
            break

        # ── Queens-off detection (popcount=0) ────────────────────
        if out["queens_off_ply"] is None and not board.queens:
            out["queens_off_ply"] = i

    return out


def _parse_move_features_with_id(args: tuple[str, str]) -> dict:
    """Pool worker: parse one game's moves, tag with its game_id."""
    gid, moves = args
    feats = _parse_move_features(moves)
    feats["game_id"] = gid
    return feats


def _add_move_features(games: pl.DataFrame, n_workers: int | None = None) -> pl.DataFrame:
    """Enrich games with per-row move-level features.

    Each game (game_id) is parsed once; the resulting per-side features are
    joined back onto both rows of the game. Parsing is the bottleneck of the
    whole feature build, so for large inputs it's spread across a process
    pool (same trick as dump_extract).
    """
    # Dedupe by game to parse each game exactly once.
    unique_games = (
        games.unique(subset=["game_id"]).select(["game_id", "moves"])
    )
    n = unique_games.height
    rows = unique_games.iter_rows()  # (game_id, moves) tuples
    start = time.monotonic()

    if n < _PARALLEL_PARSE_THRESHOLD:
        # Single-threaded — small input (e.g. live recommender).
        print(f"Parsing moves for {n:,} unique games (single-thread)...", flush=True)
        parsed = [_parse_move_features_with_id(r) for r in rows]
    else:
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 4) - 2)
        print(
            f"Parsing moves for {n:,} unique games ({n_workers} workers)...",
            flush=True,
        )
        ctx = mp.get_context("spawn")
        parsed = []
        with ctx.Pool(processes=n_workers) as pool:
            for idx, feats in enumerate(
                pool.imap(_parse_move_features_with_id, rows, chunksize=500), start=1
            ):
                parsed.append(feats)
                if idx % 100_000 == 0:
                    elapsed = time.monotonic() - start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (n - idx) / rate if rate > 0 else 0
                    print(
                        f"  {idx:>10,}/{n:,} games  rate {rate:.0f}/s  ETA {eta:.0f}s",
                        flush=True,
                    )

    parsed_df = pl.DataFrame(parsed)

    enriched = games.join(parsed_df, on="game_id", how="left")
    # Pick the player's side of each parsed field.
    enriched = enriched.with_columns(
        castle_ply=pl.when(pl.col("color") == "white")
        .then(pl.col("white_castle_ply"))
        .otherwise(pl.col("black_castle_ply")),
        castle_qside=pl.when(pl.col("color") == "white")
        .then(pl.col("white_castle_qside"))
        .otherwise(pl.col("black_castle_qside")),
        early_pawn=pl.when(pl.col("color") == "white")
        .then(pl.col("white_early_pawn"))
        .otherwise(pl.col("black_early_pawn")),
    )
    return enriched


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
    g = _add_move_features(games)
    g = _annotate_games(g)

    is_white = pl.col("color") == "white"
    is_black = pl.col("color") == "black"
    white_n = is_white.sum()
    black_n = is_black.sum()

    # Opening family flags
    is_e4_opening = pl.col("eco_letter").is_in(["B", "C"])  # 1.e4 systems
    is_d4_opening = pl.col("eco_letter").is_in(["D", "E"])  # 1.d4 systems
    # Black's reply to 1.e4, by ECO range:
    #   Sicilian   B20-B99   (1...c5)
    #   Caro-Kann  B10-B19   (1...c6)
    #   French     C00-C19   (1...e6)
    #   Open 1...e5 C20-C99  (king-pawn games)
    is_sicilian = (pl.col("eco_letter") == "B") & (pl.col("eco_num") >= 20)
    is_carokann = (pl.col("eco_letter") == "B") & (pl.col("eco_num").is_between(10, 19))
    is_french = (pl.col("eco_letter") == "C") & (pl.col("eco_num") <= 19)
    is_e5 = (pl.col("eco_letter") == "C") & (pl.col("eco_num") >= 20)

    # Black-against-e4 games: opponent played e4, we're Black (ECO B or C)
    is_black_vs_e4 = is_black & is_e4_opening
    black_vs_e4_n = is_black_vs_e4.sum()

    # Long-game stratification — games >= 80 plies (~40 moves per side).
    is_long_game = pl.col("n_moves") >= 80
    long_n = is_long_game.sum()

    # Castled games: castle_ply is not null. Aggregations restrict to these.
    has_castled = pl.col("castle_ply").is_not_null()
    castled_n = has_castled.sum()

    # Queens-off games: queens_off_ply not null
    has_queens_off = pl.col("queens_off_ply").is_not_null()
    qoff_n = has_queens_off.sum()

    # Analysed games: have [%eval] → acpl/blunders populated. Only ~12% of
    # games. Accuracy is computed as weighted aggregates over a player's
    # analysed games and kept as METADATA (not a clustering feature), since
    # most players have no analysed games and live users never do.
    is_analyzed = pl.col("has_eval").fill_null(False)
    analyzed_n = is_analyzed.sum()
    eval_moves_total = pl.col("n_eval_moves").filter(is_analyzed).sum()

    features = (
        g.group_by("username")
        .agg(
            n_games=pl.len(),
            avg_rating=pl.col("user_rating").mean(),
            # ── Accuracy metadata (from [%eval], analysed games only) ──
            n_analyzed_games=analyzed_n,
            acpl=pl.when(analyzed_n > 0)
            .then(
                (pl.col("acpl") * pl.col("n_eval_moves")).filter(is_analyzed).sum()
                / eval_moves_total
            )
            .otherwise(None),
            blunder_rate=pl.when(analyzed_n > 0)
            .then(pl.col("blunders").filter(is_analyzed).sum() / eval_moves_total)
            .otherwise(None),
            # ── Performance ───────────────────────────────────────────
            score_residual=pl.col("score_diff").mean(),
            white_score_residual=pl.when(white_n > 0)
            .then(pl.col("score_diff").filter(is_white).mean())
            .otherwise(None),
            black_score_residual=pl.when(black_n > 0)
            .then(pl.col("score_diff").filter(is_black).mean())
            .otherwise(None),
            long_game_residual=pl.when(long_n > 0)
            .then(pl.col("score_diff").filter(is_long_game).mean())
            .otherwise(None),
            # ── Result mix ────────────────────────────────────────────
            draw_rate=(pl.col("result") == "draw").mean(),
            mate_rate=(pl.col("status") == "mate").mean(),
            timeout_rate=(pl.col("status") == "outoftime").mean(),
            resign_rate=(pl.col("status") == "resign").mean(),
            # ── Game shape ────────────────────────────────────────────
            # n_moves is the SAN ply count (half-moves). Divide by 2 to give
            # the chess-standard "full moves per side" interpretation in the
            # feature value and the cluster summary.
            avg_moves=pl.col("n_moves").mean() / 2.0,
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
            pct_carokann_as_black=pl.when(black_vs_e4_n > 0)
            .then((is_black_vs_e4 & is_carokann).sum() / black_vs_e4_n)
            .otherwise(None),
            pct_french_as_black=pl.when(black_vs_e4_n > 0)
            .then((is_black_vs_e4 & is_french).sum() / black_vs_e4_n)
            .otherwise(None),
            pct_e5_as_black=pl.when(black_vs_e4_n > 0)
            .then((is_black_vs_e4 & is_e5).sum() / black_vs_e4_n)
            .otherwise(None),
            # ── Tier 1 move-level ─────────────────────────────────────
            avg_castle_move=pl.when(castled_n > 0)
            .then(pl.col("castle_ply").filter(has_castled).mean())
            .otherwise(None),
            pct_queenside_castle=pl.when(castled_n > 0)
            .then(pl.col("castle_qside").filter(has_castled).cast(pl.Float64).mean())
            .otherwise(None),
            avg_queens_off_move=pl.when(qoff_n > 0)
            .then(pl.col("queens_off_ply").filter(has_queens_off).mean())
            .otherwise(None),
            early_pawn_pushes=pl.col("early_pawn").mean(),
        )
        .filter(pl.col("n_games") >= min_games)
        .with_columns(
            pl.col("white_score_residual").fill_null(pl.col("score_residual")),
            pl.col("black_score_residual").fill_null(pl.col("score_residual")),
            pl.col("long_game_residual").fill_null(pl.col("score_residual")),
            pl.col("pct_e4_as_white").fill_null(
                pl.col("pct_e4_as_white").mean()
            ),
            pl.col("pct_d4_as_white").fill_null(
                pl.col("pct_d4_as_white").mean()
            ),
            pl.col("pct_sicilian_as_black").fill_null(
                pl.col("pct_sicilian_as_black").mean()
            ),
            pl.col("pct_carokann_as_black").fill_null(
                pl.col("pct_carokann_as_black").mean()
            ),
            pl.col("pct_french_as_black").fill_null(
                pl.col("pct_french_as_black").mean()
            ),
            pl.col("pct_e5_as_black").fill_null(
                pl.col("pct_e5_as_black").mean()
            ),
            # Players who never castled / never had queens come off:
            # fill with dataset mean so clustering doesn't break.
            pl.col("avg_castle_move").fill_null(
                pl.col("avg_castle_move").mean()
            ),
            pl.col("pct_queenside_castle").fill_null(
                pl.col("pct_queenside_castle").mean()
            ),
            pl.col("avg_queens_off_move").fill_null(
                pl.col("avg_queens_off_move").mean()
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
