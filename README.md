---
title: Chess Coach
emoji: ♟️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Lichess opening recommender via playstyle clustering
---

# Chess Coach

Opening recommender that matches a Lichess player to a stylistic cluster of
other players and surfaces the openings those players historically succeed
with. Trained on **5 M rated games from the Lichess April-2026 monthly
database dump**, covering **154 510 players**.

The model doesn't recommend openings that "are good" — it recommends openings
that **players who play like you tend to succeed with**. The clustering is
built on skill-adjusted features, so style is separated from raw rating.

> Full engineering rationale for every decision the project made:
> [`DECISIONS.md`](DECISIONS.md).

---

## Example

```
$ uv run python -m chess_coach.recommender pedrovaz02

pedrovaz02 -> cluster 1 (Underrated 1.e4 overperformer)
Openings ranked by how well players with your style profile have historically
done with them.

Suggested openings as White         Suggested openings as Black
1. Italian Game                     1. Sicilian Defense
2. Scotch Game                      2. French Defense
3. King's Pawn Game                 3. Caro-Kann Defense
4. Queen's Pawn Game                4. Scandinavian Defense
5. Ruy Lopez                        5. Philidor Defense
```

The ranking uses the **Elo-adjusted score residual** (actual − expected
score given opponent rating) with **Bayesian shrinkage** applied so obscure
openings — high residual but tiny sample, mostly self-selection bias — get
pushed down in favour of mainstream ones with proven track records.

Opening names are aggregated to the **family** level ("Sicilian Defense",
not "Sicilian Defense: Najdorf, Poisoned Pawn, Main Line"), because
sub-variations are statistically noisy and unreadable.

Raw win-rate percentages are deliberately not shown — they'd reflect win
rates of cluster members against cluster-level opposition, not what *you*
would score. See [`DECISIONS.md` § 5](DECISIONS.md#51-drop-raw-win-rate-percentages).

---

## How it works

```
Lichess monthly dump (.pgn.zst, ~28 GB)
   |
   v
dump_download.py + dump_extract.py
   |   stream zstd → split per-game PGN → multiprocessing pool of 14
   |   parser workers (~10 000 games/s on an M-series Mac)
   v
data/games.parquet                10 M rows (2 per kept game)
   |
   v
features.py                       per-player aggregation +
   |                              per-game move parsing via python-chess
   v
data/features.parquet             154 510 players × 18 features
   |
   v
cluster.py                        StandardScaler → KMeans (K=5)
   |                              random_state=42, n_init=10
   v
data/models/{kmeans,scaler}.joblib
data/players_clustered.parquet
   |
   v
precompute.py                     top openings per (cluster, color),
   |                              with Bayesian shrinkage + family
   v                              aggregation + classifier-based filter
data/recommendations.json
   |
   v
FastAPI (api.py)  +  static frontend  →  /recommend/{username}
```

The pipeline has two entry points: the live recommender (which fetches a
user's most recent games from the Lichess REST API on demand) and the
training pipeline (which works off the monthly dump).

### Data collection

For training data we use **Lichess's open monthly database dump** — one
`.pgn.zst` file per month, ~28 GB each, covering ~90 M games. Streaming
decompression (`zstandard.stream_reader` + per-game PGN slicing) keeps peak
memory under 1 GB regardless of input size.

Extraction is parallelised across CPU cores: the main thread slices the
stream into per-game text, a `multiprocessing.Pool` of 14 workers parses
each game with python-chess in parallel (~10 000 games/s observed). A 5 M-
game extraction takes about 16 minutes on an M-series Mac.

Filters applied at extract time: rated standard games only, in
blitz/rapid/classical, both players' Elo present, both within a configurable
rating band, with ECO + opening info present.

For the live recommender we still hit the **Lichess REST API** to fetch a
user's recent games — that path has been hardened with `requests.Session` +
`urllib3.util.Retry` for resilience against rate limits and stalled
connections.

### Playstyle features

Eighteen per-player features, organised in four groups. The most important
design choice is **skill adjustment via Elo expected score**: instead of raw
`win_rate`, we use the *score residual* — actual score minus what Elo
predicts for the matchup. A residual of `+0.05` means "you over-perform
your Elo by 5 score points per game on average". This is orthogonal to raw
strength.

| Group       | Feature                  | Meaning                                                  |
| ---         | ---                      | ---                                                      |
| Performance | `score_residual`         | Skill-adjusted overall                                   |
|             | `white_score_residual`   | Same, restricted to White                                |
|             | `black_score_residual`   | Same, restricted to Black                                |
|             | `long_game_residual`     | Skill-adjusted in games ≥ 80 plies (endgame ability)     |
| Result mix  | `draw_rate`              | Fraction drawn                                           |
|             | `mate_rate`              | Fraction ending in checkmate                             |
|             | `timeout_rate`           | Fraction lost on time                                    |
|             | `resign_rate`            | Fraction ending in resignation                           |
| Game shape  | `avg_moves`              | Mean full moves per side                                 |
|             | `short_game_rate`        | Fraction < 40 plies                                      |
| Repertoire  | `opening_diversity`      | Unique ECO codes / total games                           |
|             | `pct_e4_as_white`        | As White: % starting 1.e4 (ECO B/C)                      |
|             | `pct_d4_as_white`        | As White: % starting 1.d4 (ECO D/E)                      |
|             | `pct_sicilian_as_black`  | As Black vs 1.e4: % playing Sicilian                     |
| Tier-1 move | `avg_castle_move`        | Mean ply when this player castled                        |
|             | `pct_queenside_castle`   | Fraction of castles that were O-O-O                      |
|             | `avg_queens_off_move`    | Mean ply when both queens left the board (simplification)|
|             | `early_pawn_pushes`      | Pawn moves in this player's first 10 moves               |

The Tier-1 features are parsed from the SAN move string via python-chess
(~2 800 games/s for a single-threaded sweep). They're what let the
`Queenside king-hunter` cluster emerge — invisible without castle-side and
castle-timing data.

Engine-derived features (centipawn loss, sacrifice detection) are not in
the model; see [Phase 3 in the roadmap](#roadmap).

### Clustering

`StandardScaler` then `KMeans(n_clusters=5, random_state=42, n_init=10)`.
K=5 was chosen from a sweep over K=2..10 (see `uv run python -m
chess_coach.cluster --evaluate`) — both the inertia elbow and the
silhouette peak align at 5.

The five clusters in the trained model:

| # | Size   | Avg rating | Identity                                                                    |
| - | ---    | ---        | ---                                                                         |
| 0 | 28 956 | 1431       | **Quick 1.e4 amateur** — 84% e4 White, decides games fast (~29 moves)       |
| 1 | 28 558 | 1540       | **Underrated 1.e4 overperformer** — +0.11 residual, +0.15 in long games     |
| 2 | 40 932 | 1789       | **1.e4 grinder** — higher-rated, longest games (~38 moves)                  |
| 3 | 20 803 | 1415       | **Queenside king-hunter** — 41% O-O-O, castles late, 38% mate rate          |
| 4 | 35 261 | 1740       | **1.d4 specialist** — 6% e4 vs 48% d4 — exclusive closed-position player    |

Cluster IDs are saved with the model; the human-readable identity blurbs
live in `precompute.py:CLUSTER_PROFILES`. If you retrain with a different
K, those need re-labelling (`cluster.py --k <N>` then inspect the cluster
summary heatmap).

### Recommendation

Once a user is projected into a cluster, the recommender pulls the top
openings for that `(cluster, color)` from `recommendations.json` — a lookup
table precomputed during training.

Each `(cluster, color)` ranking goes through three filters:

1. **Color-appropriate filter.** Openings classified as Black's strategic
   choice (Sicilian Defense, French Defense, all Indian setups, etc.) only
   appear under "as Black"; White-led openings (Italian Game, Ruy Lopez,
   Polish Opening, Queen's Gambit) only under "as White". The classifier is
   built from the official Lichess [chess-openings TSVs](https://github.com/lichess-org/chess-openings)
   by replaying each opening's canonical PGN and checking whose move was
   last; ambiguous edge cases fall back to family-name lookup. 99.4% of
   names in the dataset are covered.

2. **Family aggregation.** Sub-variations ("Sicilian Defense: Najdorf
   Variation, Poisoned Pawn Variation, Main Line") collapse into the
   family ("Sicilian Defense"). Readable for non-masters, and yields larger
   per-row sample sizes.

3. **Bayesian shrinkage on the residual.** `shrunk = n/(n+30000) × raw`.
   Without this, obscure openings (Borg Defense, Elephant Gambit) dominate
   the top ranks because the few players who specialise in them
   self-select as over-performers. Shrinkage pulls small-sample residuals
   toward 0; mainstream openings with 50 k+ samples are barely affected.

---

## Honest findings

- **Silhouette ~0.08** across every experiment — 281 players, 5 044 players,
  154 510 players; 8 features, 13 features, 18 features. This pattern is
  stable enough to be the headline finding: chess playstyle is most likely
  a **continuum**, not a discrete partition. K-Means imposes hard borders
  on a cloud that doesn't naturally have them. The clusters are still
  interpretable, but a continuous embedding or Gaussian-mixture model would
  probably fit the data better. See [`DECISIONS.md` § 4.3](DECISIONS.md#43-chess-style-is-a-continuum-not-clusters).

- **Skill adjustment is the single most valuable feature engineering
  choice.** Without it, raw `win_rate` re-discovers the rating column —
  clusters would just sort by Elo. Score residual is what makes the
  "Underrated overperformer" cluster exist consistently across data scales.

- **Move-level features unlocked identities outcome-only features can't
  see.** The Queenside king-hunter cluster — 41% queenside castles, mean
  castle ply 14+, 38% mate rate — is invisible without castle-timing data.

- **Obscure openings have inflated score residuals from self-selection.**
  Only the rare players who study Borg Defense play it, and they study it
  deeply, so they over-perform. Bayesian shrinkage corrects for this so the
  recommendations end up grounded in mainstream openings.

---

## Limitations

- The recommendations are **not personal win-rate predictions**. They
  reflect how well *cluster members* have done with each opening, against
  cluster-level opposition. A 1300 player projected into a cluster of 2300s
  won't suddenly score 85% with the London System.

- The training data is from **one calendar month** of Lichess play. Style
  populations and opening fashions shift over time — a longer training
  window (Phase 4) would smooth this.

- The Lichess Elo distribution is **slightly under-representing very low
  ratings** (< 1200) because few players in that band have ≥ 20 games per
  month in standard time controls.

---

## Setup

```bash
# Install uv if you don't have it
brew install uv               # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# ── Training pipeline (data → model → recommendations) ────────────────

# 1. Download the Lichess monthly dump (~28 GB, ~25 min)
uv run python -m chess_coach.dump_download --month 2026-04

# 2. Extract games matching our filters (~16 min with multiprocessing)
uv run python -m chess_coach.dump_extract \
    --input data/dumps/lichess_db_standard_rated_2026-04.pgn.zst \
    --output data/games.parquet \
    --max-games 5000000

# 3. Build per-player features (~30 min — bottleneck is PGN parsing)
uv run python -m chess_coach.features

# 4. Inspect the K sweep, then train the chosen K
uv run python -m chess_coach.cluster --evaluate
uv run python -m chess_coach.cluster --k 5

# 5. Precompute the recommendations.json table
uv run python -m chess_coach.precompute

# ── Inference ────────────────────────────────────────────────────────

# CLI — recommend for any Lichess user
uv run python -m chess_coach.recommender <lichess_username>

# Or run the local web app
uv run uvicorn chess_coach.api:app --port 8000
# then open http://localhost:8000
```

The trained-model artifacts (`data/models/`, `data/recommendations.json`,
`data/players_clustered.parquet`) are needed at inference time but are
gitignored — you can either run the training pipeline once or download a
snapshot release.

---

## Project layout

```
chess-coach/
|-- src/chess_coach/
|   |-- dump_download.py        Streaming download of monthly Lichess dump
|   |-- dump_extract.py         Parallel PGN parser → games.parquet
|   |-- collector.py            API-based collection (fallback for live recommender)
|   |-- features.py             Per-player playstyle vectors (18 features)
|   |-- cluster.py              K-Means with elbow + silhouette tooling
|   |-- precompute.py           Build recommendations.json lookup
|   |-- recommender.py          End-to-end pipeline for a single user
|   |-- api.py                  FastAPI backend
|   |-- openings.json           Opening → color classifier (built from Lichess TSVs)
|   `-- hello_lichess.py        Sanity check
|-- scripts/
|   `-- build_openings_db.py    (Re)build openings.json from the chess-openings repo
|-- static/                     Frontend (HTML/CSS/JS, no build step)
|-- notebooks/
|   `-- 01_data_exploration.ipynb   EDA + cluster visualisation
|-- docs/figures/               PNGs referenced from this README and notebook
|-- data/                       (gitignored) parquet files + trained model
|-- DECISIONS.md                Full engineering rationale log
`-- pyproject.toml
```

---

## Roadmap

- **Phase 1** — End-to-end recommender, local CLI, EDA notebook. **Done.**
- **Phase 2** — FastAPI + minimal frontend, locally runnable. **Done.**
- **Phase 2.5** — Migrate training to Lichess monthly dumps; parallelise PGN
  extraction; Tier-1 move-level features; family-level recommendations with
  Bayesian shrinkage. **Done.**
- **Phase 3** — Parse the `[%eval ...]` annotations already present in
  ~10–15% of dump games to extract accuracy / blunder rate / ACPL as
  extra features. Pre-computed by Lichess, so no local Stockfish needed.
- **Phase 4** — Train on 12 months of dumps for temporal-trend coverage
  and a more robust style atlas.
- **Phase 5** (speculative) — Replace K-Means with a continuous embedding
  to address the silhouette finding, or compare against Gaussian Mixture
  Models + HDBSCAN for soft cluster assignments.

---

## Stack

`uv` · `polars` · `pandas` · `pyarrow` · `zstandard` · `requests` ·
`scikit-learn` · `joblib` · `python-chess` · `matplotlib` · `seaborn` ·
`rich` · `fastapi` · `uvicorn` · `jupyter`
