"""Compare two Lichess players and suggest openings that suit *both*.

Two outputs:

1. compare_players — each player's style cluster plus a side-by-side feature
   table, sorted by where they differ most.

2. mutual_openings — for a given side assignment (one plays White, the other
   Black), the opening families that are a good fit for BOTH at once: the
   White player's cluster over-performs as White in that family AND the Black
   player's cluster over-performs as Black in it. A game neither would dread —
   each is on their preferred side of an opening that suits their style.

Run:
    uv run python -m chess_coach.versus pedrovaz02 pinhasz
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
from chess_coach.features import build_player_features


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = DATA_DIR / "models"


def _profile(username: str, scaler, model, recs: dict) -> dict:
    """Fetch a user's games, build features, predict cluster."""
    raw = fetch_games(username, 100)
    rows = [r for r in (game_to_row(g, username) for g in raw) if r is not None]
    if not rows:
        raise ValueError(f"No rated games found for '{username}'")
    feats = build_player_features(pl.DataFrame(rows), min_games=1)
    row = feats.row(0, named=True)
    # Use the production feature order the scaler/model were trained on, read
    # from recommendations.json — NOT the working-tree FEATURE_COLUMNS, which
    # may carry experimental extra features (see DECISIONS § 4.2).
    cols = recs["feature_columns"]
    X = feats.select(cols).to_numpy()
    cluster_id = int(model.predict(scaler.transform(X))[0])
    meta = next(c for c in recs["clusters"] if c["id"] == cluster_id)
    return {"username": username, "features": row, "cluster": meta,
            "rating": float(row["avg_rating"]), "n_games": feats["n_games"][0]}


def _centered(residuals: dict[str, float]) -> dict[str, float]:
    """Center a cluster-colour's family residuals on their own mean.

    Clusters have different baselines (an over-performing cluster is positive
    everywhere, an under-performing one negative everywhere). Centering asks
    the fair question: is this family a *relative strength* for this cluster —
    does it do better here than across its own repertoire?
    """
    if not residuals:
        return {}
    mean = sum(residuals.values()) / len(residuals)
    return {fam: r - mean for fam, r in residuals.items()}


def mutual_openings(white_cluster: dict, black_cluster: dict, top_n: int = 6) -> list[dict]:
    """Families that suit the White player AND the Black player at once.

    Each side's family residuals are centered on that cluster's own mean, so
    the comparison is "relative strength", not raw residual (which is
    confounded by whether the cluster over- or under-performs overall). A
    family is good for both when it's an above-baseline family for the White
    player as White AND for the Black player as Black. Ranked by the combined
    relative strength.
    """
    w = _centered(white_cluster["family_residuals"]["white"])
    b = _centered(black_cluster["family_residuals"]["black"])
    shared = set(w) & set(b)
    rows = [
        {
            "family": fam,
            "white_rel": w[fam],
            "black_rel": b[fam],
            "combined": w[fam] + b[fam],
            "both_positive": w[fam] > 0 and b[fam] > 0,
        }
        for fam in shared
    ]
    rows.sort(key=lambda r: r["combined"], reverse=True)
    return rows[:top_n]


def build_versus_payload(a: dict, b: dict) -> dict:
    """JSON-serialisable comparison of two profiled players (for the API)."""
    fa, fb = a["features"], b["features"]
    # Production feature order = the keys of the cluster's feature_means
    # (written from the 18-feature training set), not FEATURE_COLUMNS.
    cols = list(a["cluster"]["feature_means"].keys())
    feature_comparison = sorted(
        (
            {
                "feature": c,
                "a": float(fa[c]),
                "b": float(fb[c]),
                "delta": float(fa[c]) - float(fb[c]),
            }
            for c in cols
        ),
        key=lambda r: abs(r["delta"]),
        reverse=True,
    )

    def cluster_brief(p: dict) -> dict:
        return {
            "id": p["cluster"]["id"],
            "name": p["cluster"]["name"],
            "blurb": p["cluster"]["blurb"],
        }

    return {
        "player_a": {
            "username": a["username"], "rating": a["rating"],
            "n_games": int(a["n_games"]), "cluster": cluster_brief(a),
        },
        "player_b": {
            "username": b["username"], "rating": b["rating"],
            "n_games": int(b["n_games"]), "cluster": cluster_brief(b),
        },
        "feature_comparison": feature_comparison,
        "matchups": {
            "a_white_b_black": mutual_openings(a["cluster"], b["cluster"]),
            "b_white_a_black": mutual_openings(b["cluster"], a["cluster"]),
        },
    }


def render(console: Console, a: dict, b: dict, recs: dict) -> None:
    # ── Cluster headline ────────────────────────────────────────────────
    console.print(
        f"\n[bold green]{a['username']}[/bold green] "
        f"([dim]{a['rating']:.0f}[/dim]) — {a['cluster']['name']}    "
        f"[bold]vs[/bold]    "
        f"[bold cyan]{b['username']}[/bold cyan] "
        f"([dim]{b['rating']:.0f}[/dim]) — {b['cluster']['name']}\n"
    )

    # ── Feature comparison, biggest differences first ───────────────────
    fa, fb = a["features"], b["features"]
    diffs = sorted(
        FEATURE_COLUMNS,
        key=lambda c: abs(float(fa[c]) - float(fb[c])),
        reverse=True,
    )
    table = Table(title="Playstyle comparison (largest differences first)")
    table.add_column("Feature")
    table.add_column(a["username"], justify="right")
    table.add_column(b["username"], justify="right")
    table.add_column("Δ", justify="right", style="dim")
    for c in diffs:
        va, vb = float(fa[c]), float(fb[c])
        table.add_row(c, f"{va:+.3f}" if abs(va) < 10 else f"{va:.1f}",
                      f"{vb:+.3f}" if abs(vb) < 10 else f"{vb:.1f}",
                      f"{va - vb:+.3f}" if abs(va - vb) < 10 else f"{va - vb:+.1f}")
    console.print(table)

    # ── Mutual openings, both side assignments ──────────────────────────
    for white, black in ((a, b), (b, a)):
        muts = mutual_openings(white["cluster"], black["cluster"])
        t = Table(
            title=f"Good games — {white['username']} (White) vs {black['username']} (Black)",
            title_justify="left",
        )
        t.add_column("Opening family")
        t.add_column(f"suits {white['username']} (W)", justify="right", style="dim")
        t.add_column(f"suits {black['username']} (B)", justify="right", style="dim")
        if not muts:
            t.add_row("[dim]no shared family with data on both sides[/dim]", "", "")
        for m in muts:
            mark = " ✓" if m["both_positive"] else ""
            t.add_row(m["family"] + mark,
                      f"{m['white_rel']:+.3f}", f"{m['black_rel']:+.3f}")
        console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Lichess players")
    parser.add_argument("player_a")
    parser.add_argument("player_b")
    args = parser.parse_args()

    console = Console()
    model = joblib.load(MODELS_DIR / "kmeans.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    recs = json.loads((DATA_DIR / "recommendations.json").read_text())

    console.print(f"Fetching games for [bold]{args.player_a}[/bold] and [bold]{args.player_b}[/bold]...")
    a = _profile(args.player_a, scaler, model, recs)
    b = _profile(args.player_b, scaler, model, recs)
    render(console, a, b, recs)


if __name__ == "__main__":
    main()
