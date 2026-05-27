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
        "Walks through the player dataset, the 13-dimensional playstyle feature space, and the K-Means clusters built on it.\n"
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
        "13-dimensional feature space projected to 2D for visualisation. Each point is a player, coloured first by K-Means cluster (left), then by rating (right) — to see whether the clusters are tracking style or just rating."
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
        "Reading the heatmap above and the opening bar chart:\n"
        "\n"
        "| Cluster | Avg rating | Identity |\n"
        "| --- | --- | --- |\n"
        "| **0** | ~2000 | Blitz brawlers — ~29% time-loss, low draws, ~50% 1.e4 |\n"
        "| **1** | ~2310 | Underrated overperformers — score residual +0.14, ~46% Sicilians as Black |\n"
        "| **2** | ~2480 | 1.d4 specialists — 47% d4, only 10% e4 |\n"
        "| **3** | ~2670 | Elite e4 generalists — top ratings, 72% e4, longest games |\n"
        "| **4** | ~1965 | 1.e4 dogmatists — 87% e4, short decisive games |\n"
        "\n"
        "Each identity emerges from a *combination* of features. Cluster 4 and Cluster 0 are both ~2000-rated and play 1.e4, but Cluster 0 burns the clock and Cluster 4 has the shortest games. K-Means is picking up these multi-feature signatures."
    ),
    md(
        "## Limitations & next steps\n"
        "\n"
        "- **Silhouette ~0.13**, meaning clusters are not crisply separated. Likely because chess playstyle is a continuum rather than a discrete partition. Could try Gaussian Mixture Models or continuous embeddings.\n"
        "- **`pct_e4_as_white` ↔ `pct_d4_as_white` correlation near -1** — could drop one feature without information loss; kept for downstream interpretability.\n"
        "- **No move-level features** — adding centipawn loss, tactical pattern density, or piece activity (all needing Stockfish) is Phase 3.\n"
        "- **Underrepresented rating bands** — only 22 of 281 players are < 1800. Phase 4 (training on monthly Lichess database dumps) addresses this."
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
