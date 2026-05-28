"""Fit a different K and judge whether the extra clusters are meaningful.

Trains K-Means at --k on the same features, characterises each cluster, and
cross-tabulates against the production K=5 labels (from
players_clustered.parquet) to see whether the new clusters are distinct
identities or just splits of existing ones.

Does NOT touch the production model — analysis only.

Run:
    uv run python scripts/inspect_k.py --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_coach.features import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
RANDOM_STATE = 42

# K=5 identity names (production model) for the cross-tab.
K5_NAMES = {
    0: "Quick 1.e4 amateur", 1: "Underrated overperf", 2: "1.e4 grinder",
    3: "Queenside king-hunter", 4: "1.d4 specialist",
}

# Most discriminating features to show in the characterisation.
SHOW = ["score_residual", "pct_e4_as_white", "pct_d4_as_white",
        "pct_sicilian_as_black", "pct_queenside_castle", "avg_castle_move",
        "mate_rate", "timeout_rate", "avg_moves", "opening_diversity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    K = args.k
    console = Console(width=160)

    feats = pl.read_parquet(DATA_DIR / "features.parquet")
    clustered = pl.read_parquet(DATA_DIR / "players_clustered.parquet")
    console.print(f"Loaded {feats.height:,} players")

    X = StandardScaler().fit_transform(feats.select(FEATURE_COLUMNS).to_numpy())
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=10_000, random_state=RANDOM_STATE)

    sil5 = silhouette_score(
        X,
        KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10).fit_predict(X),
        sample_size=10_000, random_state=RANDOM_STATE,
    )
    console.print(f"\nSilhouette: K=5 → {sil5:.3f}   K={K} → {sil:.3f}\n")

    f = feats.with_columns(pl.Series("k_label", labels))

    # ── Characterisation table ──────────────────────────────────────────
    summary = (
        f.group_by("k_label")
        .agg(
            size=pl.len(),
            rating=pl.col("avg_rating").mean(),
            acpl=pl.col("acpl").mean(),
            **{c: pl.col(c).mean() for c in SHOW},
        )
        .sort("k_label")
    )
    t = Table(title=f"K={K} cluster characterisation")
    t.add_column("C", style="bold"); t.add_column("size", justify="right")
    t.add_column("rating", justify="right"); t.add_column("ACPL", justify="right")
    for c in SHOW:
        t.add_column(c.replace("pct_", "").replace("_as_white", "")[:8], justify="right")
    for r in summary.iter_rows(named=True):
        cells = [str(r["k_label"]), str(r["size"]), f"{r['rating']:.0f}", f"{r['acpl']:.0f}"]
        for c in SHOW:
            v = r[c]
            cells.append(f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
        t.add_row(*cells)
    console.print(t)

    # ── Cross-tab vs production K=5 ─────────────────────────────────────
    joined = f.select(["username", "k_label"]).join(
        clustered.select(["username", "cluster"]), on="username", how="inner"
    )
    ct = joined.group_by(["k_label", "cluster"]).len().sort(["k_label", "cluster"])
    # Build matrix K x 5
    mat = np.zeros((K, 5), dtype=int)
    for row in ct.iter_rows(named=True):
        mat[row["k_label"], row["cluster"]] += row["len"]

    console.print(f"\n[bold]Where each K={K} cluster comes from (% of its members in each K=5 cluster):[/bold]")
    ctab = Table()
    ctab.add_column(f"K{K}\\K5", style="bold")
    for j in range(5):
        ctab.add_column(f"C{j}\n{K5_NAMES[j][:10]}", justify="right")
    ctab.add_column("dominant", style="bold")
    for i in range(K):
        rowtot = mat[i].sum()
        pcts = mat[i] / rowtot * 100 if rowtot else mat[i]
        dom = int(np.argmax(mat[i]))
        cells = [str(i)] + [f"{p:.0f}%" if p >= 1 else "·" for p in pcts]
        cells.append(f"C{dom} {K5_NAMES[dom][:14]} ({pcts[dom]:.0f}%)")
        ctab.add_row(*cells)
    console.print(ctab)

    # how many K5 clusters does each K-cluster draw >20% from? (purity)
    purities = [(mat[i] / mat[i].sum()).max() for i in range(K) if mat[i].sum()]
    console.print(
        f"\nMean dominant-K5 share per K={K} cluster: {np.mean(purities):.0%} "
        f"(high = clean splits of K=5; low = genuinely new groupings)"
    )

    # ── Heatmap figure ──────────────────────────────────────────────────
    cm = summary.select(SHOW).to_pandas()
    cm.index = [f"C{i}" for i in summary["k_label"].to_list()]
    cmz = (cm - cm.mean()) / cm.std()
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, K * 0.5 + 1))
    sns.heatmap(cmz, annot=cm.values, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "z-score"}, linewidths=0.4)
    ax.set_title(f"K={K} cluster characterisation (color = z-score, annot = raw mean)")
    fig.tight_layout()
    out = FIG_DIR / f"10_k{K}_characterisation.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    console.print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
