"""Build notebooks/01_data_exploration.ipynb programmatically.

Run with:  uv run python notebooks/_build_eda.py
Then execute:  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb

Keeping the build script in the repo (instead of editing the .ipynb JSON by
hand) means the notebook is reproducible and reviewable as Python.
"""

from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).parent / "01_data_exploration.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# Chess Coach — Exploratory Data Analysis\n"
        "\n"
        "Walks through the player dataset, the 18-dimensional playstyle feature space, and the K-Means clusters built on it.\n"
        "\n"
        "All figures are saved to `docs/figures/` so they can be referenced from the README."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import polars as pl\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "# Make src/ importable when running the notebook from notebooks/\n"
        "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
        "from chess_coach.features import FEATURE_COLUMNS\n"
        "\n"
        "sns.set_theme(style='whitegrid', palette='Set2')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "\n"
        "DATA_DIR = Path.cwd().parent / 'data'\n"
        "FIG_DIR = Path.cwd().parent / 'docs' / 'figures'\n"
        "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    ),
    md("## 1. Dataset overview"),
    code(
        "games = pl.read_parquet(DATA_DIR / 'games.parquet')\n"
        "features = pl.read_parquet(DATA_DIR / 'features.parquet')\n"
        "clustered = pl.read_parquet(DATA_DIR / 'players_clustered.parquet')\n"
        "\n"
        "print(f'Games:                       {games.height:>7,}')\n"
        "print(f'Players (raw):               {games[\"username\"].n_unique():>7}')\n"
        "print(f'Players (after min_games):   {features.height:>7}')\n"
        "print(f'Rating range:                {features[\"avg_rating\"].min():.0f} – {features[\"avg_rating\"].max():.0f}')\n"
        "print(f'Avg games per player:        {features[\"n_games\"].mean():.1f}')\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "\n"
        "ax = axes[0]\n"
        "ax.hist(features['avg_rating'], bins=30, edgecolor='white')\n"
        "ax.set(xlabel='Average rating', ylabel='Players', title='Player rating distribution')\n"
        "\n"
        "ax = axes[1]\n"
        "ax.hist(features['n_games'], bins=20, edgecolor='white', color='#4ECDC4')\n"
        "ax.set(xlabel='Games per player', ylabel='Players', title='Sample size per player')\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '01_dataset_overview.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    md(
        "## 2. Feature distributions\n"
        "\n"
        "Each player is represented by 13 features. Two design choices:\n"
        "\n"
        "- **Skill adjustment** via Elo expected score — `score_residual` is actual score minus what Elo predicts given opponent ratings. A `+0.05` means the player over-performs their rating by 5 score points per game on average. Orthogonal to raw strength.\n"
        "- **Opening family proportions** built from ECO code prefixes (`B`/`C` = 1.e4 systems, `D`/`E` = 1.d4). No PGN parsing needed."
    ),
    code(
        "n_cols = 4\n"
        "n_rows = int(np.ceil(len(FEATURE_COLUMNS) / n_cols))\n"
        "fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.7*n_rows))\n"
        "\n"
        "for ax, col in zip(axes.flat, FEATURE_COLUMNS):\n"
        "    ax.hist(features[col], bins=25, edgecolor='white')\n"
        "    ax.set_title(col, fontsize=10)\n"
        "for ax in axes.flat[len(FEATURE_COLUMNS):]:\n"
        "    ax.set_visible(False)\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '02_feature_distributions.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    md(
        "## 3. Feature correlations\n"
        "\n"
        "Heavily correlated features add redundancy without information. Anything > 0.8 in absolute value is worth flagging."
    ),
    code(
        "features_pd = features.select(FEATURE_COLUMNS).to_pandas()\n"
        "corr = features_pd.corr()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 8))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,\n"
        "            ax=ax, annot_kws={'fontsize': 8}, square=True, cbar_kws={'shrink': 0.7})\n"
        "ax.set_title('Feature correlation matrix')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '03_correlations.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    md(
        "**Reading the matrix:**\n"
        "\n"
        "- `score_residual` is highly correlated with `white_score_residual` and `black_score_residual` by construction — they're sub-aggregates of the same quantity. Carrying all three lets the model give different weight to White vs Black performance.\n"
        "- `pct_e4_as_white` and `pct_d4_as_white` are strongly negative (≈ -1) — a player who plays 1.e4 80% of the time can only play 1.d4 at most 20%. This is the kind of perfect anti-correlation that suggests we could drop one feature, but keeping both makes downstream interpretation cleaner.\n"
        "- `resign_rate` and `mate_rate` trade off — losing on resignation excludes losing by checkmate."
    ),
    md(
        "## 4. PCA projection\n"
        "\n"
        "18-dimensional feature space projected to 2D for visualisation. Each point is a player, coloured first by K-Means cluster (left), then by rating (right) — to see whether the clusters are tracking style or just rating."
    ),
    code(
        "X = clustered.select(FEATURE_COLUMNS).to_numpy()\n"
        "X_scaled = StandardScaler().fit_transform(X)\n"
        "\n"
        "pca = PCA(n_components=2)\n"
        "pcs = pca.fit_transform(X_scaled)\n"
        "\n"
        "clusters_arr = clustered['cluster'].to_numpy()\n"
        "ratings_arr = clustered['avg_rating'].to_numpy()\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "\n"
        "ax = axes[0]\n"
        "for c in sorted(set(clusters_arr.tolist())):\n"
        "    mask = clusters_arr == c\n"
        "    ax.scatter(pcs[mask, 0], pcs[mask, 1], s=45, label=f'C{c}', edgecolor='white', linewidth=0.4)\n"
        "ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',\n"
        "       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',\n"
        "       title='K-Means clusters in PCA space')\n"
        "ax.legend(title='Cluster', loc='best', frameon=True)\n"
        "\n"
        "ax = axes[1]\n"
        "sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=ratings_arr, cmap='viridis', s=45, edgecolor='white', linewidth=0.4)\n"
        "ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',\n"
        "       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',\n"
        "       title='Same projection, coloured by rating')\n"
        "fig.colorbar(sc, ax=ax, label='Avg rating')\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '04_pca.png', bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print(f'Total variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.1%}')\n"
    ),
    md(
        "**Reading the PCA plots:**\n"
        "\n"
        "If clusters tracked *only* rating, the left and right plots would look identical. The fact that they differ means clusters capture **style on top of rating** — exactly what the skill-adjustment design was meant to surface."
    ),
    md(
        "## 5. Cluster characterisation\n"
        "\n"
        "Heatmap of cluster-mean values for each feature, z-scored *across clusters* so each row highlights what makes that feature distinctive between clusters."
    ),
    code(
        "cluster_means = clustered.group_by('cluster').agg([pl.col(c).mean() for c in FEATURE_COLUMNS]).sort('cluster')\n"
        "cm_df = cluster_means.select(FEATURE_COLUMNS).to_pandas()\n"
        "cm_df.index = [f'C{i}' for i in cluster_means['cluster'].to_list()]\n"
        "\n"
        "# Z-score across clusters for each feature (per row, after transpose)\n"
        "cm_z = (cm_df - cm_df.mean()) / cm_df.std()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n"
        "sns.heatmap(cm_z.T, annot=cm_df.T, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,\n"
        "            cbar_kws={'label': 'z-score (within feature)'}, linewidths=0.4)\n"
        "ax.set_title('Cluster characterisation — color = z-score, annotation = raw mean')\n"
        "ax.set_xlabel('Cluster')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '05_cluster_means.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    md(
        "## 6. Opening-family preference by cluster\n"
        "\n"
        "The clearest behavioural signal: how each cluster splits between 1.e4 and 1.d4 as White."
    ),
    code(
        "cluster_sizes = clustered.group_by('cluster').len().sort('cluster')\n"
        "labels = [f'C{i}\\n(n={n})' for i, n in zip(cluster_sizes['cluster'], cluster_sizes['len'])]\n"
        "\n"
        "x = np.arange(len(labels))\n"
        "width = 0.35\n"
        "e4 = cm_df['pct_e4_as_white'].values * 100\n"
        "d4 = cm_df['pct_d4_as_white'].values * 100\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.bar(x - width/2, e4, width, label='1.e4 (ECO B/C)', color='#FF6B6B', edgecolor='white')\n"
        "ax.bar(x + width/2, d4, width, label='1.d4 (ECO D/E)', color='#4ECDC4', edgecolor='white')\n"
        "ax.set(xticks=x, xticklabels=labels, ylabel='% of White games',\n"
        "       title='Opening-family preference by cluster')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '06_openings_by_cluster.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    md(
        "## Cluster identities\n"
        "\n"
        "Reading the heatmap above and the opening bar chart (K=5 on 154,510 players):\n"
        "\n"
        "| Cluster | Avg rating | Identity |\n"
        "| --- | --- | --- |\n"
        "| **0** | ~1431 | Quick 1.e4 amateur — 84% e4, decides games fast (~29 moves) |\n"
        "| **1** | ~1540 | Underrated 1.e4 overperformer — score residual +0.11, +0.15 in long games |\n"
        "| **2** | ~1789 | 1.e4 grinder — higher-rated, longest games (~38 moves) |\n"
        "| **3** | ~1415 | Queenside king-hunter — 41% O-O-O, castles late, 38% mate rate |\n"
        "| **4** | ~1740 | 1.d4 specialist — only 6% e4 vs 48% d4 |\n"
        "\n"
        "Each identity emerges from a *combination* of features. The Queenside king-hunter (C3) only became visible once castle-side and castle-timing features entered the vector — outcome features alone couldn't see it."
    ),
    md(
        "## Phase 3 — does accuracy carry style signal beyond rating?\n"
        "\n"
        "~12% of Lichess games carry `[%eval]` computer-analysis annotations. We parse them into per-player **ACPL** (average centipawn loss) and **blunder rate** — kept as metadata, not clustering features. The question: do the clusters differ in accuracy *independently of rating*?"
    ),
    code(
        "analyzed = clustered.filter(pl.col('n_analyzed_games') >= 3)\n"
        "\n"
        "overall = (analyzed.group_by('cluster')\n"
        "           .agg(acpl=pl.col('acpl').mean(), rating=pl.col('avg_rating').mean(), n=pl.len())\n"
        "           .sort('cluster').to_pandas())\n"
        "\n"
        "band = analyzed.filter((pl.col('avg_rating') >= 1350) & (pl.col('avg_rating') <= 1500))\n"
        "banded = (band.group_by('cluster')\n"
        "          .agg(acpl=pl.col('acpl').mean(), n=pl.len())\n"
        "          .sort('cluster').to_pandas())\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "\n"
        "ax = axes[0]\n"
        "ax.bar([f'C{c}' for c in overall['cluster']], overall['acpl'], color='#6a7d4f', edgecolor='white')\n"
        "ax.set(ylabel='Mean ACPL (centipawns)', title='Accuracy by cluster (all ratings)')\n"
        "for i, (a, r) in enumerate(zip(overall['acpl'], overall['rating'])):\n"
        "    ax.text(i, a + 0.5, f'{a:.0f}\\n({r:.0f})', ha='center', fontsize=8)\n"
        "\n"
        "ax = axes[1]\n"
        "ax.bar([f'C{c}' for c in banded['cluster']], banded['acpl'], color='#b3793f', edgecolor='white')\n"
        "ax.set(ylabel='Mean ACPL (centipawns)', title='Accuracy by cluster - rating band 1350-1500')\n"
        "for i, a in enumerate(banded['acpl']):\n"
        "    ax.text(i, a + 0.5, f'{a:.0f}', ha='center', fontsize=9)\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / '07_acpl_by_cluster.png', bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "corr = analyzed.select(pl.corr('acpl', 'avg_rating')).item()\n"
        "print(f'ACPL-rating correlation: {corr:.3f}')\n"
    ),
    md(
        "**Reading the two panels.** Left: ACPL drops as rating rises (global "
        "ACPL-rating correlation ~ -0.55 — accuracy *is* a large part of strength). "
        "Right: holding rating fixed (~1428), the clusters **still** differ by "
        "~16 cp/move. The Queenside king-hunter is the least accurate at any given "
        "rating — it trades precision for attacking chances — while the Underrated "
        "overperformer is the most accurate, consistent with its +0.11 score "
        "residual. Accuracy was never an input to the clustering, yet it separates "
        "the clusters exactly as the identities predict: the strongest evidence that "
        "the model captures **style**, not just rating."
    ),
    md(
        "## Limitations & next steps\n"
        "\n"
        "- **Silhouette ~0.08** across 281 → 5k → 154k players and 8 → 18 features. The metric won't move with scale or features — chess playstyle is most likely a continuum, not a discrete partition. K-Means imposes hard borders on a smooth cloud. A Gaussian-mixture model or continuous embedding would likely fit better.\n"
        "- **`pct_e4_as_white` ↔ `pct_d4_as_white` correlation near -1** — could drop one feature without information loss; kept for interpretability.\n"
        "- **Accuracy can't be a live feature** — the recommender fetches a user's games from the REST API, which has no evals. ACPL stays a characterisation overlay.\n"
        "- **Single-month training window** — Phase 4 (12 months of dumps) would smooth seasonal/meta shifts."
    ),
]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH}")
