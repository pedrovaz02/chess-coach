"""Download a Lichess monthly database dump (.pgn.zst).

The Lichess open database publishes one .pgn.zst per month at
    https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst

These are ~28 GB each. Streaming download with Range-based resume support, so
killing the script mid-way isn't fatal.

Run:
    uv run python -m chess_coach.dump_download --month 2026-04
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "dumps"
BASE_URL = "https://database.lichess.org/standard"
CHUNK = 1024 * 1024  # 1 MiB


def url_for(month: str) -> str:
    return f"{BASE_URL}/lichess_db_standard_rated_{month}.pgn.zst"


def remote_size(url: str) -> int | None:
    """Total size of the remote file, via HEAD. None if header missing."""
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    raw = r.headers.get("Content-Length")
    return int(raw) if raw else None


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(month: str, output_dir: Path, force: bool = False) -> Path:
    url = url_for(month)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"lichess_db_standard_rated_{month}.pgn.zst"

    if target.exists() and not force:
        local = target.stat().st_size
        remote = remote_size(url)
        if remote and local == remote:
            print(f"Already complete: {target} ({fmt_bytes(local)})")
            return target
        if remote is None:
            print(f"Already exists, remote size unknown: {target} ({fmt_bytes(local)})")
            return target
        # Partial — try to resume
        print(
            f"Resuming download: local {fmt_bytes(local)} / remote {fmt_bytes(remote)} "
            f"({local/remote:.1%})"
        )
        headers = {"Range": f"bytes={local}-"}
        mode = "ab"
        existing = local
        total = remote
    else:
        if target.exists():
            target.unlink()
        headers = {}
        mode = "wb"
        existing = 0
        total = remote_size(url) or 0
        print(f"Downloading {url}")
        print(f"Expected size: {fmt_bytes(total)}")

    with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as r:
        r.raise_for_status()
        downloaded = existing
        last_print = time.monotonic()
        start = last_print
        with open(target, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_print >= 2.0:
                    elapsed = now - start
                    rate = (downloaded - existing) / elapsed if elapsed > 0 else 0
                    pct = (downloaded / total * 100) if total else 0
                    eta = (total - downloaded) / rate if rate > 0 and total else 0
                    print(
                        f"  {fmt_bytes(downloaded)} / {fmt_bytes(total)} "
                        f"({pct:.1f}%) — {fmt_bytes(rate)}/s — ETA {eta:.0f}s",
                        flush=True,
                    )
                    last_print = now

    print(f"Done: {target} ({fmt_bytes(target.stat().st_size)})")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Lichess monthly dump")
    parser.add_argument("--month", required=True, help="e.g. 2026-04")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download from scratch")
    args = parser.parse_args()

    try:
        download(args.month, args.output_dir, force=args.force)
    except KeyboardInterrupt:
        print("\nInterrupted — partial file kept for resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
