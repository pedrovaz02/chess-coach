"""Stream-parse a Lichess .pgn.zst dump into games.parquet (per-player rows).

Schema matches collector.py's output exactly so the rest of the pipeline
(features → cluster → recommender) is unchanged.

For each PGN game that passes the filters, we emit TWO rows — one for the
White player's perspective, one for the Black player's. This keeps the
downstream group_by("username") logic intact.

The .pgn.zst file is decompressed in streaming mode (never fully on disk
or in memory) — peak RAM is bounded by output-buffer size (~few hundred MB
even for the full 28 GB input).

Filters applied:
    - rated games only
    - standard variant only
    - time controls in --time-controls (default: blitz, rapid, classical)
    - both players' ratings present
    - both players' usernames present (not ?, not BOT, not stockfish)
    - ratings within [min-elo, max-elo]
    - opening info present (ECO + Opening headers)

Output is per-player (so each game produces 2 rows, one row per side).

Run:
    uv run python -m chess_coach.dump_extract \
        --input data/dumps/lichess_db_standard_rated_2026-04.pgn.zst \
        --output data/games.parquet \
        --max-games 5000000
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import chess.pgn
import polars as pl
import zstandard as zstd

from chess_coach.collector import GAME_COLUMNS


DEFAULT_TIME_CONTROLS = {"blitz", "rapid", "classical"}

# Module-level state for the worker pool, set via `_init_worker`.
# We don't pickle filter config with every task — set once per process.
_WORKER_FILTERS: dict = {}

# Lichess time-control bucketing: "<initial>+<increment>" in seconds.
# Bucket boundaries match the Lichess UI conventions.
def speed_from_tc(tc: str) -> str | None:
    """'600+0' -> 'rapid' etc. Returns None for variants we can't classify."""
    if "+" not in tc:
        return None
    try:
        initial, increment = tc.split("+", 1)
        initial = int(initial)
        increment = int(increment)
    except ValueError:
        return None
    # Lichess formula: estimated game duration = initial + 40 * increment
    est = initial + 40 * increment
    if est < 30:
        return "ultraBullet"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


STATUS_MAP = {
    "Normal": None,  # filled by result-based logic below
    "Time forfeit": "outoftime",
    "Abandoned": "noStart",
    "Rules infraction": "cheat",
}


def termination_to_status(termination: str, result: str, moves: str) -> str:
    """Map PGN Termination header + result to our `status` vocabulary."""
    if termination == "Time forfeit":
        return "outoftime"
    if termination == "Abandoned":
        return "noStart"

    # Normal — distinguish mate / resign / draw / stalemate
    if result == "1/2-1/2":
        # Could be agreement, stalemate, repetition, fifty-move, insufficient material
        if moves.rstrip().endswith("1/2-1/2"):
            return "draw"
        return "draw"
    # Decisive game ending normally — usually resignation, occasionally mate.
    # Mate is detected by '#' in last move.
    last_token = moves.strip().split()[-1] if moves.strip() else ""
    # Strip the result tail if present
    last_token = re.sub(r"(1-0|0-1|1/2-1/2)$", "", last_token).strip()
    if last_token.endswith("#"):
        return "mate"
    return "resign"


GAME_ID_RE = re.compile(r"https?://lichess\.org/(\w+)")

BOT_NAMES = {"?", "BOT"}


def game_id_from_site(site_header: str | None) -> str | None:
    if not site_header:
        return None
    m = GAME_ID_RE.search(site_header)
    return m.group(1) if m else None


def date_to_epoch_ms(date: str, time: str) -> int | None:
    """'2026.04.15' + '13:42:01' -> ms epoch UTC."""
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(f"{date} {time}", "%Y.%m.%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def is_valid_player(name: str | None) -> bool:
    if not name or name == "?":
        return False
    if name.upper() in BOT_NAMES:
        return False
    return True


# ── Accuracy / ACPL from [%eval] annotations ────────────────────────────
# Per-move centipawn loss is capped so a single blunder doesn't dominate the
# average; moves losing >= BLUNDER_THRESHOLD count as blunders.
CP_CAP = 1000
BLUNDER_THRESHOLD = 200
INITIAL_EVAL = 20  # assumed White edge at the start position (centipawns)


def _walk_game(game: chess.pgn.Game):
    """Single pass over the mainline. Returns (san_moves, white_pov_cps).

    white_pov_cps[i] is the engine eval in centipawns (White's POV) after
    ply i+1, or None if that ply has no [%eval] annotation.
    """
    board = game.board()
    san_moves: list[str] = []
    cps: list[int | None] = []
    for node in game.mainline():
        san_moves.append(board.san(node.move))
        board.push(node.move)
        score = node.eval()  # python-chess parses [%eval ...] from the comment
        cps.append(score.white().score(mate_score=10000) if score is not None else None)
    return san_moves, cps


def _acpl_from_cps(cps: list[int | None]) -> dict | None:
    """Per-colour ACPL + blunder counts from White-POV centipawn evals.

    Returns None if the game has no evals at all. Centipawn loss for a move
    is how much the position worsened from the mover's perspective relative
    to the eval before the move (clamped to [0, CP_CAP]).
    """
    if not any(cp is not None for cp in cps):
        return None

    white_losses: list[int] = []
    black_losses: list[int] = []
    white_blun = black_blun = 0

    for i, after in enumerate(cps):
        if after is None:
            continue
        before = cps[i - 1] if i >= 1 and cps[i - 1] is not None else INITIAL_EVAL
        is_white_move = (i % 2 == 0)  # ply index 0 == White's first move
        # Loss = how much worse it got for the mover (White POV evals).
        loss = (before - after) if is_white_move else (after - before)
        loss = max(0, min(loss, CP_CAP))
        if is_white_move:
            white_losses.append(loss)
            if loss >= BLUNDER_THRESHOLD:
                white_blun += 1
        else:
            black_losses.append(loss)
            if loss >= BLUNDER_THRESHOLD:
                black_blun += 1

    def mean(xs: list[int]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "white_acpl": mean(white_losses),
        "black_acpl": mean(black_losses),
        "white_blunders": white_blun,
        "black_blunders": black_blun,
        "white_n_eval": len(white_losses),
        "black_n_eval": len(black_losses),
    }


def game_to_player_rows(game: chess.pgn.Game) -> list[dict] | None:
    """Convert a parsed PGN game to two rows (one per side). None if invalid."""
    h = game.headers

    if h.get("Variant", "Standard") != "Standard":
        return None
    if h.get("Rated", "").lower() == "false":
        return None

    white = h.get("White")
    black = h.get("Black")
    if not (is_valid_player(white) and is_valid_player(black)):
        return None

    try:
        white_rating = int(h.get("WhiteElo", ""))
        black_rating = int(h.get("BlackElo", ""))
    except (ValueError, TypeError):
        return None

    result = h.get("Result")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None

    eco = h.get("ECO")
    opening_name = h.get("Opening")
    if not (eco and opening_name):
        return None

    tc = h.get("TimeControl", "")
    speed = speed_from_tc(tc)
    if speed is None:
        return None

    # Single pass over the mainline: reconstruct SAN + pull [%eval] evals.
    san_moves, cps = _walk_game(game)
    moves_str = " ".join(san_moves)
    n_moves = len(san_moves)

    if n_moves < 4:
        return None

    # Accuracy stats (None if the game wasn't computer-analysed).
    acpl = _acpl_from_cps(cps)
    has_eval = acpl is not None

    termination = h.get("Termination", "Normal")
    status = termination_to_status(termination, result, moves_str)

    game_id = game_id_from_site(h.get("Site"))
    created_at = date_to_epoch_ms(h.get("UTCDate", ""), h.get("UTCTime", ""))

    rows = []
    for side, user, opp_user, user_rating, opp_rating in (
        ("white", white, black, white_rating, black_rating),
        ("black", black, white, black_rating, white_rating),
    ):
        if result == "1/2-1/2":
            per_side_result = "draw"
        elif (result == "1-0" and side == "white") or (result == "0-1" and side == "black"):
            per_side_result = "win"
        else:
            per_side_result = "loss"

        if acpl is None:
            side_acpl = side_blun = side_neval = None
        else:
            side_acpl = acpl[f"{side}_acpl"]
            side_blun = acpl[f"{side}_blunders"]
            side_neval = acpl[f"{side}_n_eval"]

        rows.append({
            "username": user,
            "game_id": game_id,
            "created_at": created_at,
            "color": side,
            "result": per_side_result,
            "status": status,
            "n_moves": n_moves,
            "moves": moves_str,
            "opening_eco": eco,
            "opening_name": opening_name,
            "speed": speed,
            "rated": True,
            "variant": "standard",
            "user_rating": user_rating,
            "opponent_username": opp_user,
            "opponent_rating": opp_rating,
            "has_eval": has_eval,
            "acpl": side_acpl,
            "blunders": side_blun,
            "n_eval_moves": side_neval,
        })
    return rows


def stream_pgn(path: Path) -> "io.TextIOBase":
    """Open a .pgn.zst as a streaming text file."""
    raw = open(path, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    return io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8")


# Match a PGN game's result token on the final move line. This lets us split
# the stream into per-game text blocks cheaply, without parsing PGN ourselves.
_RESULT_RE = re.compile(r"\s(?:1-0|0-1|1/2-1/2|\*)\s*$")


def _iter_game_texts(file: "io.TextIOBase"):
    """Yield each PGN game as a self-contained string.

    A PGN game ends at a non-header line whose last whitespace-separated token
    is one of {1-0, 0-1, 1/2-1/2, *}. We accumulate lines until we hit one
    and emit the buffer.
    """
    buf: list[str] = []
    for line in file:
        buf.append(line)
        if line.startswith("["):
            continue
        if _RESULT_RE.search(line):
            yield "".join(buf)
            buf = []
    if buf:
        yield "".join(buf)


def _init_worker(time_controls: set[str], min_elo: int, max_elo: int) -> None:
    """Initialize per-worker filter config. Called once when each pool worker spawns."""
    global _WORKER_FILTERS
    _WORKER_FILTERS = {
        "time_controls": time_controls,
        "min_elo": min_elo,
        "max_elo": max_elo,
    }


def _parse_one_game(game_text: str) -> tuple[list[dict] | None, str | None]:
    """Worker function: parse one game's PGN text and return rows or skip reason.

    Returns:
        (rows, None)        if game was parsed AND passes all filters
        (None, "reason")    if it was filtered or failed parsing
    """
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
    except Exception as exc:  # noqa: BLE001
        return None, f"parse:{type(exc).__name__}"
    if game is None:
        return None, "null"
    try:
        rows = game_to_player_rows(game)
    except Exception as exc:  # noqa: BLE001
        return None, f"convert:{type(exc).__name__}"
    if rows is None:
        return None, "filtered"

    # Post-filter on speed and rating band.
    f = _WORKER_FILTERS
    speed = rows[0]["speed"]
    if speed not in f["time_controls"]:
        return None, "filtered_speed"
    ur = rows[0]["user_rating"]
    opr = rows[0]["opponent_rating"]
    if not (f["min_elo"] <= ur <= f["max_elo"]):
        return None, "filtered_rating"
    if not (f["min_elo"] <= opr <= f["max_elo"]):
        return None, "filtered_rating"

    return rows, None


def extract(
    input_path: Path,
    output_path: Path,
    max_games: int | None,
    min_elo: int,
    max_elo: int,
    time_controls: set[str],
    flush_every: int = 100_000,
    n_workers: int | None = None,
    chunksize: int = 64,
    skip: int = 0,
) -> None:
    """Parallel stream-extract a .pgn.zst into a per-player parquet.

    Architecture:
        Main thread:     splits the zstd-decompressed PGN stream into per-game
                         text strings (cheap, single-pass).
        Worker pool:     N processes each parse one game's PGN text with
                         python-chess and apply all filters. They emit either
                         rows or a skip reason.
        Main thread:     collects results in arrival order (imap_unordered),
                         buffers rows, flushes chunk parquets periodically.

    With 14 workers on an M-series Mac we observe ~8x speedup vs single-threaded.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)

    print(f"Opening {input_path}")
    print(f"Using {n_workers} worker processes", flush=True)
    pgn = stream_pgn(input_path)

    rows_buffer: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seen_games = 0
    kept_games = 0
    start = time.monotonic()
    last_report = start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent / f".{output_path.stem}_chunks"
    tmp_dir.mkdir(exist_ok=True)
    chunk_idx = 0

    def flush_chunk():
        nonlocal rows_buffer, chunk_idx
        if not rows_buffer:
            return
        df = pl.DataFrame(rows_buffer, schema={c: None for c in GAME_COLUMNS}, strict=False)
        df.write_parquet(tmp_dir / f"chunk_{chunk_idx:06d}.parquet")
        chunk_idx += 1
        rows_buffer = []

    # Use 'spawn' explicitly — default on macOS, but be deterministic.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(time_controls, min_elo, max_elo),
    )

    try:
        text_iter = _iter_game_texts(pgn)
        # Skip the first `skip` games (cheaply, without parsing) so an
        # independent disjoint sample can be drawn from later in the stream.
        if skip > 0:
            print(f"Skipping first {skip:,} games before extracting...", flush=True)
            for n_skipped, _ in enumerate(text_iter, start=1):
                if n_skipped >= skip:
                    break
            print(f"Skipped {skip:,}. Now extracting.", flush=True)
        results = pool.imap_unordered(_parse_one_game, text_iter, chunksize=chunksize)

        for rows, err in results:
            seen_games += 1
            if rows is None:
                counts[err or "filtered"] += 1
            else:
                rows_buffer.extend(rows)
                kept_games += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = now - start
                rate = seen_games / elapsed if elapsed > 0 else 0
                print(
                    f"  seen {seen_games:>10,}  kept {kept_games:>10,} games "
                    f"({2*kept_games:,} rows)  "
                    f"buf {len(rows_buffer):>7,}  rate {rate:>7,.0f}/s",
                    flush=True,
                )
                last_report = now

            if len(rows_buffer) >= flush_every:
                flush_chunk()

            if max_games is not None and kept_games >= max_games:
                print(f"Reached max-games={max_games}, stopping.", flush=True)
                break
    finally:
        # terminate() not close() — discard pending work and shut down fast.
        pool.terminate()
        pool.join()
        flush_chunk()
        pgn.close()

    # Concat all chunks into final parquet
    chunk_files = sorted(tmp_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        print("No rows extracted.")
        return
    print(f"\nConcatenating {len(chunk_files)} chunk(s) into {output_path}")
    df = pl.concat([pl.read_parquet(f) for f in chunk_files])
    df.write_parquet(output_path)
    for f in chunk_files:
        f.unlink()
    tmp_dir.rmdir()

    elapsed = time.monotonic() - start
    print(
        f"\nDone in {elapsed:.0f}s.\n"
        f"  PGN games seen:    {seen_games:,}\n"
        f"  Games kept:        {kept_games:,}\n"
        f"  Rows in parquet:   {df.height:,} (2 per kept game)\n"
        f"  Unique players:    {df['username'].n_unique():,}\n"
        f"  Output size:       {output_path.stat().st_size / 1024**2:.1f} MB"
    )
    print(f"  Counts:            {dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Lichess PGN dump to games.parquet")
    parser.add_argument("--input", type=Path, required=True, help=".pgn.zst path")
    parser.add_argument("--output", type=Path, required=True, help="output .parquet path")
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many kept games. Default: all."
    )
    parser.add_argument("--min-elo", type=int, default=800)
    parser.add_argument("--max-elo", type=int, default=3500)
    parser.add_argument(
        "--time-controls",
        nargs="+",
        default=list(DEFAULT_TIME_CONTROLS),
        choices=["ultraBullet", "bullet", "blitz", "rapid", "classical"],
    )
    parser.add_argument(
        "--flush-every", type=int, default=100_000,
        help="Write a chunk parquet every N rows accumulated."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parser worker processes. Default: cpu_count - 2."
    )
    parser.add_argument(
        "--chunksize", type=int, default=64,
        help="Tasks per worker dispatch. Higher = less IPC overhead, more memory."
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip the first N games in the stream before extracting. Use to "
             "draw a disjoint independent sample from later in the dump."
    )
    args = parser.parse_args()

    extract(
        input_path=args.input,
        output_path=args.output,
        max_games=args.max_games,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        time_controls=set(args.time_controls),
        flush_every=args.flush_every,
        n_workers=args.workers,
        chunksize=args.chunksize,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
