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

    # Reconstruct move SAN string by walking the mainline.
    # python-chess gives us the parsed mainline; the original move text was
    # consumed during parsing. Rebuild it cheaply.
    board = game.board()
    san_moves: list[str] = []
    for move in game.mainline_moves():
        san_moves.append(board.san(move))
        board.push(move)
    moves_str = " ".join(san_moves)
    n_moves = len(san_moves)

    if n_moves < 4:
        return None

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
        })
    return rows


def stream_pgn(path: Path) -> "io.TextIOBase":
    """Open a .pgn.zst as a streaming text file."""
    raw = open(path, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    return io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8")


def extract(
    input_path: Path,
    output_path: Path,
    max_games: int | None,
    min_elo: int,
    max_elo: int,
    time_controls: set[str],
    flush_every: int = 100_000,
) -> None:
    print(f"Opening {input_path}")
    pgn = stream_pgn(input_path)

    rows_buffer: list[dict] = []
    counts = defaultdict(int)
    seen_games = 0
    kept_games = 0
    start = time.monotonic()
    last_report = start

    # We write to a per-chunk parquet then concat at the end. This bounds RAM.
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

    try:
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception as exc:
                counts["parse_error"] += 1
                continue
            if game is None:
                break
            seen_games += 1

            try:
                rows = game_to_player_rows(game)
            except Exception as exc:
                counts["convert_error"] += 1
                continue

            if rows is None:
                counts["filtered"] += 1
            else:
                speed = rows[0]["speed"]
                ur = rows[0]["user_rating"]
                opr = rows[0]["opponent_rating"]
                if speed not in time_controls:
                    counts["filtered_speed"] += 1
                elif not (min_elo <= ur <= max_elo) or not (min_elo <= opr <= max_elo):
                    counts["filtered_rating"] += 1
                else:
                    rows_buffer.extend(rows)
                    kept_games += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                rate = seen_games / (now - start)
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
                print(f"Reached max-games={max_games}, stopping.")
                break
    finally:
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
    args = parser.parse_args()

    extract(
        input_path=args.input,
        output_path=args.output,
        max_games=args.max_games,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        time_controls=set(args.time_controls),
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
