"""Compare two independent 5M-game samples to test finding reproducibility.

Sample v1 = first 5M kept games of the April-2026 dump.
Sample v2 = next 5M kept games (extracted with --skip 8.5M), disjoint from v1.

If the clusters, silhouette, GMM/HDBSCAN continuum signals and the
ACPL-by-cluster finding all replicate across two disjoint samples, the
findings are robust and not artifacts of one particular slice of games.

Clusters are matched between runs by nearest feature-mean profile (K-Means
labels are arbitrary, so cluster 0 in v1 need not be cluster 0 in v2).

Run after both features.parquet and features_v2.parquet exist:
    uv run python scripts/compare_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_coach.features import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
RANDOM_STATE = 42
K = 5
K_RANGE = range(2, 9)


def load(path: Path):
    feats = pl.read_parquet(path)
    X = feats.select(FEATURE_COLUMNS).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    return feats, Xs


def silhouette_curve(Xs):
    out = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
        out.append(silhouette_score(Xs, labels, sample_size=10_000, random_state=RANDOM_STATE))
    return out


def gmm_bic_curve(Xs):
    pca = PCA(n_components=6, random_state=RANDOM_STATE).fit_transform(Xs)
    out = []
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=RANDOM_STATE, n_init=2).fit(pca)
        out.append(gmm.bic(pca))
    return out


def cluster_profiles(feats, Xs):
    """Fit K=5, return (labels, standardized centroids, raw feature-mean table)."""
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(Xs)
    labels = km.labels_
    f = feats.with_columns(pl.Series("cluster", labels))
    means = (f.group_by("cluster")
             .agg(
                 size=pl.len(),
                 rating=pl.col("avg_rating").mean(),
                 acpl=pl.col("acpl").mean(),
                 **{c: pl.col(c).mean() for c in FEATURE_COLUMNS},
             )
             .sort("cluster"))
    return km.cluster_centers_, means


def main() -> None:
    print("Loading samples...")
    f1, X1 = load(DATA_DIR / "features.parquet")
    f2, X2 = load(DATA_DIR / "features_v2.parquet")
    print(f"  v1: {f1.height:,} players")
    print(f"  v2: {f2.height:,} players")
    overlap = set(f1["username"]).intersection(set(f2["username"]))
    print(f"  username overlap: {len(overlap):,} ({len(overlap)/f1.height*100:.1f}% of v1)")

    print("\nSilhouette curves...")
    sil1, sil2 = silhouette_curve(X1), silhouette_curve(X2)
    for k, a, b in zip(K_RANGE, sil1, sil2):
        print(f"  K={k}: v1={a:.3f}  v2={b:.3f}")

    print("\nGMM BIC curves (PCA-6)...")
    bic1, bic2 = gmm_bic_curve(X1), gmm_bic_curve(X2)

    print("\nCluster profiles + matching...")
    cent1, means1 = cluster_profiles(f1, X1)
    cent2, means2 = cluster_profiles(f2, X2)
    # Match v2 clusters to v1 by nearest standardized centroid (Hungarian).
    dist = np.linalg.norm(cent1[:, None, :] - cent2[None, :, :], axis=2)
    row, col = linear_sum_assignment(dist)
    matching = {int(r): int(c) for r, c in zip(row, col)}  # v1 -> v2

    key = ["rating", "acpl", "pct_e4_as_white", "pct_d4_as_white",
           "pct_queenside_castle", "score_residual"]
    m1 = {r["cluster"]: r for r in means1.iter_rows(named=True)}
    m2 = {r["cluster"]: r for r in means2.iter_rows(named=True)}

    print("\nMatched cluster comparison (v1 vs v2):")
    print(f"  {'feature':<22}" + "".join(f"{f'C{v1}~C{matching[v1]}':>16}" for v1 in sorted(matching)))
    for feat in key:
        line = f"  {feat:<22}"
        for v1 in sorted(matching):
            v2 = matching[v1]
            a, b = m1[v1][feat], m2[v2][feat]
            line += f"{f'{a:.2f}/{b:.2f}':>16}"
        print(line)

    # ── Figure: silhouette + BIC, v1 vs v2 ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ks = list(K_RANGE)

    ax = axes[0]
    ax.plot(ks, sil1, "o-", label="sample v1", color="#6a7d4f")
    ax.plot(ks, sil2, "s--", label="sample v2", color="#b3793f")
    ax.set(xlabel="K", ylabel="Silhouette", title="K-Means silhouette — two disjoint 5M samples")
    ax.set_ylim(0, 0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(ks, bic1, "o-", label="sample v1", color="#6a7d4f")
    ax.plot(ks, bic2, "s--", label="sample v2", color="#b3793f")
    ax.set(xlabel="K (GMM components)", ylabel="BIC", title="GMM BIC (PCA-6) — two disjoint samples")
    ax.legend()

    fig.tight_layout()
    out = FIG_DIR / "09_sample_reproducibility.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
