"""Collect rated games from a diverse set of Lichess players.

Strategy (two-stage to get rating + style diversity):
    Stage 1 — strong players, varied styles:
        Top N per perf type (bullet/blitz/rapid/classical). Strong players from
        different time controls have meaningfully different styles.

    Stage 2 — lower-rated players:
        Extract opponents (with rating < threshold) of stage-1 players, sample
        from them, fetch their games too. Without this the dataset is all
        super-GMs and clustering finds nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.console import Console


LICHESS_API = "https://lichess.org/api"


def _make_session() -> requests.Session:
    """Session with automatic retry+backoff on 429 / 5xx and connection errors.

    Lichess will silently slow-roll requests if you hammer them; the retry
    layer turns those into observable failures we can back off from.
    """
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2.0,  # 0s, 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


_SESSION = _make_session()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

GAME_COLUMNS = [
    "username",
    "game_id",
    "created_at",
    "color",
    "result",
    "status",
    "n_moves",
    "moves",
    "opening_eco",
    "opening_name",
    "speed",
    "rated",
    "variant",
    "user_rating",
    "opponent_username",
    "opponent_rating",
]


def fetch_top_players(perf_type: str, count: int) -> list[str]:
    url = f"{LICHESS_API}/player/top/{count}/{perf_type}"
    response = _SESSION.get(url, timeout=10)
    response.raise_for_status()
    return [p["username"] for p in response.json()["users"]]


def fetch_games(username: str, max_games: int) -> list[dict]:
    """Fetch a user's recent rated games as NDJSON, parse to list of dicts.

    Non-streaming on purpose — even 200 games is < 1MB. Streaming makes
    iter_lines() vulnerable to half-open connections that hang indefinitely
    (we hit this in production: process at 0% CPU for an hour waiting on a
    dead socket). A bounded read with a hard read timeout is more robust.
    """
    url = f"{LICHESS_API}/games/user/{username}"
    params = {
        "max": max_games,
        "opening": "true",
        "moves": "true",
        "rated": "true",
        "perfType": "blitz,rapid,classical",
    }
    headers = {"Accept": "application/x-ndjson"}
    # (connect, read) timeouts. Read timeout is the max gap between bytes —
    # set tight so we fail fast on stalled connections.
    response = _SESSION.get(url, params=params, headers=headers, timeout=(10, 30))
    response.raise_for_status()

    games = []
    for line in response.text.splitlines():
        line = line.strip()
        if line:
            games.append(json.loads(line))
    return games


def game_to_row(game: dict, target_user: str) -> dict | None:
    """Flatten one game into a row from the perspective of `target_user`.

    Returns None if the target user isn't actually one of the players (defensive).
    """
    players = game.get("players", {})
    white_user = players.get("white", {}).get("user", {}).get("name", "")
    black_user = players.get("black", {}).get("user", {}).get("name", "")

    target_lower = target_user.lower()
    played_white = white_user.lower() == target_lower
    played_black = black_user.lower() == target_lower

    if not (played_white or played_black):
        return None

    winner = game.get("winner")
    if winner is None:
        result = "draw"
    elif (winner == "white" and played_white) or (winner == "black" and played_black):
        result = "win"
    else:
        result = "loss"

    moves = game.get("moves", "")
    n_moves = len(moves.split()) if moves else 0

    opening = game.get("opening") or {}
    user_side = "white" if played_white else "black"
    opp_side = "black" if played_white else "white"

    opp_user = players.get(opp_side, {}).get("user", {}).get("name")

    return {
        "username": target_user,
        "game_id": game.get("id"),
        "created_at": game.get("createdAt"),
        "color": user_side,
        "result": result,
        "status": game.get("status"),
        "n_moves": n_moves,
        # Full SAN move string (space-separated). Adds ~400 bytes per game on
        # average — fine for the dataset size. Required for Tier-1 move-level
        # features (castling timing, queen exchanges, early pawn pushes).
        "moves": moves,
        "opening_eco": opening.get("eco"),
        "opening_name": opening.get("name"),
        "speed": game.get("speed"),
        "rated": game.get("rated", False),
        "variant": game.get("variant"),
        "user_rating": players.get(user_side, {}).get("rating"),
        "opponent_username": opp_user,
        "opponent_rating": players.get(opp_side, {}).get("rating"),
    }


def _fetch_for_usernames(
    usernames: list[str],
    games_per_player: int,
    sleep_between: float,
    label: str,
    console: Console,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch games for a list of usernames.

    Emits a plain-text progress line every 10 players so progress is visible
    when stdout is redirected to a file (Rich's progress bar uses ANSI escapes
    that don't survive file redirection).
    """
    import sys

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    total = len(usernames)
    start = time.monotonic()

    print(f"[{label}] starting: {total} users", flush=True)
    sys.stdout.flush()

    for idx, username in enumerate(usernames, start=1):
        try:
            games = fetch_games(username, games_per_player)
            ok_rows = 0
            for game in games:
                row = game_to_row(game, username)
                if row is not None:
                    rows.append(row)
                    ok_rows += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((username, str(exc)))
            print(
                f"[{label}] {idx}/{total} {username} ERROR: {exc}",
                flush=True,
            )

        # Heartbeat every 10 users (or always for last user)
        if idx % 10 == 0 or idx == total:
            elapsed = time.monotonic() - start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate if rate > 0 else 0
            print(
                f"[{label}] {idx}/{total} ({rate:.1f} users/s, "
                f"ETA {eta:.0f}s) — {len(rows)} rows so far, "
                f"{len(errors)} errors",
                flush=True,
            )

        time.sleep(sleep_between)

    return rows, errors


def collect_diverse(
    perf_types: list[str],
    top_n_per_perf: int,
    games_per_player: int,
    n_low_rated: int,
    low_rating_max: int,
    output: Path,
    sleep_between: float = 1.0,
    seed: int = 42,
) -> None:
    """Two-stage collection:
        Stage 1: top_n per perf type → strong players, varied styles
        Stage 2: sample n_low_rated opponents (rating < low_rating_max) from
                 stage-1 games, fetch their games
    """
    console = Console()
    rng = random.Random(seed)

    # ── Stage 1 ─────────────────────────────────────────────────────────
    console.print("[bold cyan]Stage 1[/bold cyan]: top players across perf types")
    strong: set[str] = set()
    for perf_type in perf_types:
        try:
            top = fetch_top_players(perf_type, top_n_per_perf)
            strong.update(top)
            console.print(f"  {perf_type}: +{len(top)} (total unique: {len(strong)})")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]{perf_type} failed: {exc}[/red]")
        time.sleep(sleep_between)

    strong_list = sorted(strong)
    console.print(f"→ {len(strong_list)} unique strong players\n")

    stage1_rows, stage1_errs = _fetch_for_usernames(
        strong_list, games_per_player, sleep_between, "Stage 1 games", console
    )
    console.print(
        f"Stage 1: [green]{len(stage1_rows):,}[/green] games, "
        f"{len(stage1_errs)} errors\n"
    )

    # ── Stage 2: opponent sampling ──────────────────────────────────────
    console.print("[bold cyan]Stage 2[/bold cyan]: sampling lower-rated opponents")

    # Best rating per opponent username
    opp_pool: dict[str, int] = {}
    for row in stage1_rows:
        name = row.get("opponent_username")
        rating = row.get("opponent_rating")
        if name and rating and rating < low_rating_max:
            key = name.lower()
            if key not in opp_pool or rating > opp_pool[key]:
                opp_pool[key] = rating

    # Don't refetch users we already have
    already = {u.lower() for u in strong_list}
    candidates = [(n, r) for n, r in opp_pool.items() if n not in already]
    console.print(
        f"  Found {len(candidates)} unique opponents below rating {low_rating_max}"
    )

    sample_size = min(n_low_rated, len(candidates))
    sample = rng.sample(candidates, sample_size)
    sample_names = [name for name, _ in sample]
    console.print(f"  Sampling {sample_size} of them\n")

    stage2_rows, stage2_errs = _fetch_for_usernames(
        sample_names, games_per_player, sleep_between, "Stage 2 games", console
    )
    console.print(
        f"Stage 2: [green]{len(stage2_rows):,}[/green] games, "
        f"{len(stage2_errs)} errors\n"
    )

    # ── Save ────────────────────────────────────────────────────────────
    all_rows = stage1_rows + stage2_rows
    df = pl.DataFrame(all_rows, schema={c: None for c in GAME_COLUMNS}, strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    console.print(
        f"[bold green]Total:[/bold green] {df.height:,} games "
        f"across {df['username'].n_unique()} players\n"
        f"Saved to [bold]{output}[/bold]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chess games from Lichess")
    parser.add_argument(
        "--perf-types",
        nargs="+",
        default=["bullet", "blitz", "rapid", "classical"],
        help="Leaderboards to draw top players from (Stage 1)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50, help="Top players per perf type"
    )
    parser.add_argument(
        "--games", type=int, default=80, help="Max games per player"
    )
    parser.add_argument(
        "--low-rated", type=int, default=100,
        help="Number of lower-rated opponents to sample (Stage 2)",
    )
    parser.add_argument(
        "--low-rating-max", type=int, default=2200,
        help="Maximum rating to count as 'lower-rated' (Stage 2)",
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "games.parquet"
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_diverse(
        perf_types=args.perf_types,
        top_n_per_perf=args.top_n,
        games_per_player=args.games,
        n_low_rated=args.low_rated,
        low_rating_max=args.low_rating_max,
        output=args.output,
        sleep_between=args.sleep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
