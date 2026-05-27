"""Build a name → color classification of every Lichess opening.

Source: github.com/lichess-org/chess-openings (a.tsv ... e.tsv).

For each (eco, name, pgn) entry we replay the PGN and look at *whose move
was last*. That move is what makes the opening's identity:

    Sicilian Defense          1.e4 c5         → last move c5 by Black → "black"
    Italian Game              1.e4 e5 ... Bc4 → last move Bc4 by White → "white"
    Marshall Attack           ... 7...d5      → Black                  → "black"
    Sicilian Najdorf English  ... 6.Be3       → White (response setup) → "white"

The result is written to src/chess_coach/openings.json and used by the
recommender to filter recommendations by the color that actually chooses
the opening.

Run from the project root:
    uv run python scripts/build_openings_db.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import requests


REPO_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
TSV_FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "src" / "chess_coach" / "openings.json"


def parse_pgn_moves(pgn: str) -> chess.Board:
    """Replay PGN move text and return the resulting board state."""
    board = chess.Board()
    for token in pgn.split():
        if "." in token:  # skip "1." / "1..." / "1.5." etc.
            continue
        try:
            board.push_san(token)
        except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
            raise
    return board


def classify_opening(pgn: str) -> str | None:
    """Return 'white' or 'black' based on who made the last move. None on parse fail."""
    try:
        board = parse_pgn_moves(pgn)
    except Exception:
        return None
    # After White's move, it's Black's turn. So if board.turn == BLACK, last move was White's.
    return "white" if board.turn == chess.BLACK else "black"


def fetch_tsv(filename: str) -> list[tuple[str, str, str]]:
    """Download one TSV. Returns list of (eco, name, pgn) tuples."""
    url = f"{REPO_BASE}/{filename}"
    print(f"  fetching {filename}...", flush=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows: list[tuple[str, str, str]] = []
    lines = response.text.strip().split("\n")
    # First line is header: eco \t name \t pgn
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def main() -> None:
    print(f"Building opening classifier from {REPO_BASE}")
    all_rows: list[tuple[str, str, str]] = []
    for fname in TSV_FILES:
        all_rows.extend(fetch_tsv(fname))
    print(f"Total openings: {len(all_rows):,}\n")

    classified: dict[str, str] = {}
    name_conflicts: dict[str, set[str]] = {}
    failed = 0

    for eco, name, pgn in all_rows:
        color = classify_opening(pgn)
        if color is None:
            failed += 1
            continue
        if name in classified and classified[name] != color:
            name_conflicts.setdefault(name, set()).add(classified[name])
            name_conflicts[name].add(color)
        classified[name] = color

    n_white = sum(1 for v in classified.values() if v == "white")
    n_black = sum(1 for v in classified.values() if v == "black")

    print(f"Classified: {len(classified):,} unique openings")
    print(f"  White-led: {n_white:,}")
    print(f"  Black-led: {n_black:,}")
    print(f"  Parse failures: {failed}")
    if name_conflicts:
        print(f"  Name conflicts (same name → different color): {len(name_conflicts)}")
        for n, colors in list(name_conflicts.items())[:5]:
            print(f"    {n}: {colors}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(classified, indent=2, ensure_ascii=False, sort_keys=True)
    )
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
