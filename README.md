# Chess Coach

Opening recommender that matches a Lichess player to a stylistic cluster of
other players and surfaces openings that work well within that cluster.

The model doesn't recommend openings that "are good" — it recommends openings
that **players who play like you tend to succeed with**. The clustering is
built on skill-adjusted features, so style is separated from raw rating.

---

## Example

```
$ uv run python -m chess_coach.recommender pedrovaz02

pedrovaz02 -> cluster 1
Openings ranked by how well players with your style profile have historically
done with them.

Suggested openings as White                   Suggested openings as Black
1. B30  Rossolimo Attack      (15 games)      1. B10  Caro-Kann Defense           (15)
2. A05  Nimzo-Larsen          (15 games)      2. A45  Indian: Acc. London         (22)
3. D35  QGD: Exchange         (21 games)      3. B30  Sicilian: Old Sicilian      (16)
4. D02  London System         (23 games)      4. B00  Owen Defense                (15)
5. D01  Rapport-Jobava System (26 games)      5. C78  Ruy Lopez: Morphy Defense   (19)
```

The ranking is by **Elo-adjusted score residual** (actual − expected score
given opponent rating) across all games of cluster members in that opening.
Raw percentages are deliberately not shown — they reflect win rates of 2300+
players against 2300+ opposition, and showing them as if they applied to any
user would be misleading. The number in parentheses is sample size: more
games supporting the recommendation = more confidence in the ordering.

---

## How it works

```
Lichess API
   |
   v
collector.py   ---->  data/games.parquet   (~28k games, 284 players)
                       one row per game
   |
   v
features.py    ---->  data/features.parquet  (one row per player)
                       13 skill-adjusted playstyle features
   |
   v
cluster.py     ---->  data/models/kmeans.joblib
                      data/players_clustered.parquet
                       K-Means with K=5
   |
   v
recommender.py ---->  username -> cluster -> top openings (per color)
```

### Data collection

Two-stage scrape of the Lichess REST API to get both stylistic and rating
diversity:

1. **Stage 1**: top 50 players in each of bullet / blitz / rapid / classical
   (~186 unique strong players with naturally varied styles).
2. **Stage 2**: extract their opponents with rating < 2200, sample 100 of them,
   fetch their games too.

Result: 27,757 rated games across 284 players, ratings 784–3014.

![Dataset overview](docs/figures/01_dataset_overview.png)

### Playstyle features

Computed per player from their games. The key design decision is **skill
adjustment**: instead of raw `win_rate`, we use the *score residual* — actual
score minus the score Elo predicts given the opponent's rating. A residual of
`+0.05` means "you over-perform your rating by 5 score points per game on
average", and it's orthogonal to raw strength.

| Feature                 | Meaning                                              |
| ---                     | ---                                                  |
| `score_residual`        | Skill-adjusted overall performance                   |
| `white_score_residual`  | Same, restricted to games as White                   |
| `black_score_residual`  | Same, restricted to games as Black                   |
| `draw_rate`             | Fraction of games drawn                              |
| `avg_moves`             | Mean ply count (long games = positional grinder)     |
| `opening_diversity`     | Unique ECO codes / total games (narrow vs broad rep) |
| `timeout_rate`          | Fraction lost on time (time management)              |
| `resign_rate`           | Fraction ending in resignation (fighting spirit)     |
| `mate_rate`             | Fraction ending in checkmate (real sharpness)        |
| `short_game_rate`       | Fraction of games < 40 plies (tactical decisions)    |
| `pct_e4_as_white`       | As White: fraction starting 1.e4 (ECO B/C)           |
| `pct_d4_as_white`       | As White: fraction starting 1.d4 (ECO D/E)           |
| `pct_sicilian_as_black` | As Black vs 1.e4: fraction playing Sicilian          |

### Clustering

`StandardScaler` then `KMeans`. K=5 chosen from a sweep over K=2..10 by
inspecting the inertia elbow and silhouette score (`uv run python -m
chess_coach.cluster --evaluate`).

Projected into 2D via PCA, with side-by-side colourings for cluster
membership and rating — if the clusters were just sorting by rating, the two
plots would be identical. They aren't, which is the point.

![PCA projection](docs/figures/04_pca.png)

The cluster identities, read from the feature-mean heatmap:

![Cluster characterisation](docs/figures/05_cluster_means.png)

| # |  n  | Avg rating | Identity                                                          |
| - | --- | ---        | ---                                                               |
| 0 |  39 | 2008       | **Blitz brawlers**: 29% timeout rate, high mate rate, ~50% 1.e4   |
| 1 |  26 | 2312       | **Underrated overperformers**: score residual +0.14, 46% Sicilians |
| 2 |  84 | 2478       | **1.d4 specialists**: 47% d4, only 10% e4                          |
| 3 |  90 | 2669       | **Elite e4 generalists**: top ratings, 72% e4, longest games       |
| 4 |  42 | 1966       | **1.e4 dogmatists**: 87% e4, shortest games, low diversity         |

The opening-family preference is the strongest behavioural signal between
clusters:

![Opening preference by cluster](docs/figures/06_openings_by_cluster.png)

See [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)
for the full analysis: feature distributions, correlation matrix, and
discussion of the cluster identities.

---

## Limitations

Be honest about what this does and doesn't do.

- **Silhouette score is ~0.13**, meaning the clusters are not crisply
  separated. This is most likely because chess playstyle is a continuum, not
  a discrete partition — K-Means imposes hard boundaries on a smooth cloud.
  The clusters are still interpretable, but a different algorithm (Gaussian
  mixture, hierarchical, or even a continuous embedding) might fit the data
  better.
- **Recommendations are not personal win-rate predictions.** The ranking
  reflects how well *cluster members* have done with each opening, against
  cluster-member-level opposition. A 1300-rated player projected into a
  cluster of 2300s won't suddenly score 85% with the London System. The
  signal is "this opening tends to suit this playstyle", not "expect this
  win rate".
- **Top-heavy training distribution.** Even with Stage-2 sampling, only ~22
  of 281 players are below 1800. Recommendations for low-rated users rely on
  style match, not direct examples.
- **No engine analysis yet.** Features are surface-level (results, move
  counts, statuses, opening-family proportions). Things like average
  centipawn loss, sacrifice frequency, or tactical pattern usage would
  require Stockfish and are not in scope for Phase 1.

---

## Setup

```bash
# Install uv if you don't have it
brew install uv               # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Build the dataset (~25 minutes, polite 1s sleep between API calls)
uv run python -m chess_coach.collector --top-n 50 --games 80 --low-rated 100

# Extract features
uv run python -m chess_coach.features

# Pick K by inspecting the sweep
uv run python -m chess_coach.cluster --evaluate

# Train final model
uv run python -m chess_coach.cluster --k 5

# Recommend for any Lichess user
uv run python -m chess_coach.recommender <lichess_username>
```

---

## Project layout

```
chess-coach/
|-- src/chess_coach/
|   |-- collector.py        Two-stage Lichess scrape
|   |-- features.py         Per-player playstyle vectors (skill-adjusted)
|   |-- cluster.py          K-Means with elbow + silhouette tooling
|   |-- recommender.py      End-to-end pipeline for a single user
|   `-- hello_lichess.py    Sanity check that the API is reachable
|-- notebooks/
|   `-- 01_data_exploration.ipynb   EDA + cluster visualisation
|-- docs/figures/           PNGs referenced from this README
|-- data/                   (gitignored) parquet files + trained model
|-- tests/                  (planned)
`-- pyproject.toml
```

---

## Roadmap

- **Phase 1** (this) — end-to-end recommender, local CLI, EDA notebook. **Done.**
- **Phase 2** — FastAPI + minimal frontend, deploy to a free tier so anyone
  can paste their Lichess username and get recommendations.
- **Phase 3** — Stockfish-derived features (centipawn loss, tactical density)
  and a serious comparison with rule-based baselines.
- **Phase 4** — train on a monthly Lichess database dump (~90M games) so
  recommendations are based on millions of similar players, not 281.

---

## Stack

`uv` · `polars` · `pandas` · `pyarrow` · `requests` · `scikit-learn` ·
`joblib` · `matplotlib` · `seaborn` · `rich` · `jupyter` · `python-chess`
(planned for Phase 3)
