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
                       8 skill-adjusted playstyle features
   |
   v
cluster.py     ---->  data/models/kmeans.joblib
                      data/players_clustered.parquet
                       K-Means with K=4
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

### Playstyle features

Computed per player from their games. The key design decision is **skill
adjustment**: instead of raw `win_rate`, we use the *score residual* — actual
score minus the score Elo predicts given the opponent's rating. A residual of
`+0.05` means "you over-perform your rating by 5 score points per game on
average", and it's orthogonal to raw strength.

| Feature                | Meaning                                              |
| ---                    | ---                                                  |
| `score_residual`       | Skill-adjusted overall performance                   |
| `white_score_residual` | Same, restricted to games as White                   |
| `black_score_residual` | Same, restricted to games as Black                   |
| `draw_rate`            | Fraction of games drawn                              |
| `avg_moves`            | Mean ply count (long games = positional grinder)     |
| `opening_diversity`    | Unique ECO codes / total games (narrow vs broad rep) |
| `timeout_rate`         | Fraction lost on time (time management)              |
| `resign_rate`          | Fraction ending in resignation (fighting spirit)     |

### Clustering

`StandardScaler` then `KMeans`. K=4 chosen from a sweep over K=2..10 by
inspecting the inertia elbow and silhouette score (`uv run python -m
chess_coach.cluster --evaluate`).

The four clusters that emerged:

| # | n   | Avg rating | Identity                                                       |
| - | --- | ---        | ---                                                            |
| 0 |  47 | 1988       | **Time-pressure players**: 29% timeout rate, short games, narrow rep |
| 1 |  38 | 2326       | **Under-rated overperformers**: +0.12 score residual            |
| 2 |  88 | 2241       | **Narrow grinders**: low opening diversity, high resign rate    |
| 3 | 108 | 2688       | **Elite varied positional**: longest games, broadest repertoire |

---

## Limitations

Be honest about what this does and doesn't do.

- **Silhouette score is ~0.16**, meaning the clusters are not crisply
  separated. This is most likely because chess playstyle is a continuum, not
  a discrete partition — K-Means imposes hard boundaries on a smooth cloud.
  The clusters are still interpretable, but a different algorithm (Gaussian
  mixture, hierarchical, or even a continuous embedding) might fit the data
  better.
- **Win rates in the output are not predictions for you.** They're the
  aggregate win rate of *all players in the target cluster* in that opening.
  A 1300-rated player projected into a cluster of 2300s won't suddenly score
  85% with the London System. The signal is "this opening tends to suit this
  playstyle", not "expect this win rate".
- **Top-heavy training distribution.** Even with Stage-2 sampling, only ~22
  of 284 players are below 1800. Recommendations for low-rated users rely on
  style match, not direct examples.
- **No engine analysis yet.** Features are surface-level (results, move
  counts, statuses). Things like average centipawn loss, sacrifice frequency,
  or tactical pattern usage would require Stockfish and are not in scope for
  Phase 1.

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
uv run python -m chess_coach.cluster --k 4

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
|-- data/                   (gitignored) parquet files + trained model
|-- notebooks/              (planned) EDA + cluster visualisation
|-- tests/                  (planned)
`-- pyproject.toml
```

---

## Roadmap

- **Phase 1** (this) — end-to-end recommender, local CLI. **Done.**
- **Phase 2** — FastAPI + minimal frontend, deploy to a free tier.
- **Phase 3** — Stockfish-derived features (centipawn loss, tactical density)
  and a serious comparison with rule-based baselines.
- **Phase 4** — train on a monthly Lichess database dump (~90M games) so
  recommendations are based on millions of similar players, not 284.

---

## Stack

`uv` · `polars` · `pandas` · `requests` · `scikit-learn` · `joblib` ·
`matplotlib` · `seaborn` · `rich` · `python-chess` (planned for Phase 3)
