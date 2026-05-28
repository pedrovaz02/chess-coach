"""Compare K-Means against GMM and HDBSCAN to test the continuum hypothesis.

The headline finding across the project is that K-Means silhouette stays
~0.08 regardless of data scale or feature count, suggesting chess playstyle
is a continuum rather than a discrete partition. This script stress-tests
that claim with two alternative algorithms:

    Gaussian Mixture Model — soft, probabilistic assignments. If the data
        had well-separated groups, each player's max component responsibility
        would be near 1.0. Low/spread responsibilities ⇒ continuum.

    HDBSCAN — density-based. It finds clusters as dense regions and labels
        the rest "noise". If playstyle were discrete we'd expect several
        persistent dense clusters; a continuum collapses to one mega-cluster
        or labels most points as noise.

Run:
    uv run python scripts/clustering_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_coach.features import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
RANDOM_STATE = 42
SUBSAMPLE = 30_000  # HDBSCAN/GMM are slow on 154k; 30k is statistically plenty
K_RANGE = range(2, 9)


def main() -> None:
    feats = pl.read_parquet(DATA_DIR / "features.parquet")
    print(f"Loaded {feats.height:,} players × {len(FEATURE_COLUMNS)} features")

    rng = np.random.default_rng(RANDOM_STATE)
    if feats.height > SUBSAMPLE:
        idx = rng.choice(feats.height, SUBSAMPLE, replace=False)
        feats = feats[idx]
        print(f"Subsampled to {feats.height:,} for the comparison")

    X = feats.select(FEATURE_COLUMNS).to_numpy()
    X = StandardScaler().fit_transform(X)

    # GMM responsibilities and HDBSCAN density both degrade in 18-dim
    # (distances concentrate; full-covariance Gaussians become overconfident).
    # Run those two on a PCA projection where density/probability are
    # meaningful, so an "all-noise" or "hard-assignment" result reflects the
    # data's geometry, not the curse of dimensionality.
    pca = PCA(n_components=6, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(
        f"PCA(6) explains {pca.explained_variance_ratio_.sum():.0%} of variance "
        f"— GMM/HDBSCAN run on this projection"
    )

    # ── K-Means silhouette (the baseline claim) ──────────────────────────
    print("\n=== K-Means ===")
    km_sil = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=10_000, random_state=RANDOM_STATE)
        km_sil.append(sil)
        print(f"  K={k}: silhouette={sil:.3f}")

    # ── GMM: BIC + soft-assignment confidence (on PCA-6) ─────────────────
    print("\n=== Gaussian Mixture (PCA-6) ===")
    gmm_bic, gmm_maxresp, gmm_entropy_norm = [], [], []
    for k in K_RANGE:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=3, max_iter=200,
        ).fit(X_pca)
        probs = gmm.predict_proba(X_pca)
        max_resp = probs.max(axis=1).mean()
        # Normalised entropy of the responsibility vector: 0 = certain,
        # 1 = uniform (max confusion). Averaged over players.
        ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        ent_norm = (ent / np.log(k)).mean()
        gmm_bic.append(gmm.bic(X_pca))
        gmm_maxresp.append(max_resp)
        gmm_entropy_norm.append(ent_norm)
        print(
            f"  K={k}: BIC={gmm.bic(X_pca):,.0f}  "
            f"avg max-responsibility={max_resp:.2f}  "
            f"norm-entropy={ent_norm:.2f}"
        )

    # ── HDBSCAN: does density find discrete groups? (on PCA-6) ───────────
    print("\n=== HDBSCAN (PCA-6) ===")
    hdb_rows = []
    for mcs in (50, 100, 250, 500):
        hdb = HDBSCAN(min_cluster_size=mcs)
        labels = hdb.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()
        # Size of the largest non-noise cluster, as a fraction
        if n_clusters > 0:
            sizes = [(labels == c).sum() for c in set(labels) if c != -1]
            biggest = max(sizes) / len(labels)
        else:
            biggest = 0.0
        hdb_rows.append((mcs, n_clusters, noise, biggest))
        print(
            f"  min_cluster_size={mcs:>4}: clusters={n_clusters}  "
            f"noise={noise:.1%}  biggest-cluster={biggest:.1%}"
        )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ks = list(K_RANGE)

    ax = axes[0]
    ax.plot(ks, km_sil, "o-", color="#6a7d4f")
    ax.axhline(0.25, ls="--", color="#b34033", lw=1)
    ax.text(ks[-1], 0.26, "0.25 = weak-structure floor", ha="right", fontsize=8, color="#b34033")
    ax.set(xlabel="K", ylabel="Silhouette", title="K-Means: silhouette stays low")
    ax.set_ylim(0, 0.4)

    ax = axes[1]
    ax.plot(ks, gmm_maxresp, "o-", color="#6a7d4f", label="avg max responsibility")
    ax.plot(ks, gmm_entropy_norm, "s--", color="#b3793f", label="norm. entropy")
    ax.set(xlabel="K (GMM components)", title="GMM: assignments are soft")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[2]
    mcs_vals = [r[0] for r in hdb_rows]
    noise_vals = [r[2] * 100 for r in hdb_rows]
    big_vals = [r[3] * 100 for r in hdb_rows]
    x = np.arange(len(mcs_vals))
    ax.bar(x - 0.2, noise_vals, 0.4, label="% noise", color="#b34033")
    ax.bar(x + 0.2, big_vals, 0.4, label="% in biggest cluster", color="#6a7d4f")
    ax.set(xticks=x, xticklabels=mcs_vals, xlabel="HDBSCAN min_cluster_size",
           ylabel="% of players", title="HDBSCAN: no crisp density groups")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "08_clustering_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
