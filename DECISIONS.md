# Decisions log

*🇬🇧 English · [🇵🇹 Português](DECISIONS.pt.md)*

This document tracks the design and engineering decisions that shaped
chess-coach. Each entry has the context (what was the situation), what we
chose, what alternatives we considered, and the outcome with hindsight.

The intent is to make the project legible to anyone reading the code, and to
hold the rationale accountable to the data. Several decisions here were
reversed after evidence — those are kept on purpose.

---

## Contents

1. [Stack and tooling](#1-stack-and-tooling)
2. [Data collection](#2-data-collection)
3. [Feature engineering](#3-feature-engineering)
4. [Clustering](#4-clustering)
5. [Recommendation UX](#5-recommendation-ux)
6. [Scaling and performance](#6-scaling-and-performance)
7. [Honest findings and deferred work](#7-honest-findings-and-deferred-work)

---

## 1. Stack and tooling

### 1.1 `uv` instead of `pip` / `poetry`

**Context.** Need a Python package manager that handles virtualenvs, lockfiles
and dependency resolution.

**Decision.** Use `uv`.

**Alternatives.** `pip + venv` (manual), `poetry` (slower, more legacy),
`pipenv` (declining adoption), `conda` (overkill, ML-only).

**Why.** Modern, 10-100x faster than pip, builtin lockfile, single tool for
venv + deps. Standard in the 2025 Python community. Signals current tooling
literacy.

**Outcome.** No regrets. `uv add`, `uv sync`, `uv run` covered every workflow.

---

### 1.2 `polars` *and* `pandas`

**Context.** Need to manipulate tabular data — initially small (10k games),
later large (10M+ rows).

**Decision.** `polars` as the primary, `pandas` only inside notebooks for
seaborn/matplotlib compatibility.

**Alternatives.** Pure pandas (familiar but slow on 10M rows), DuckDB
(SQL-first, less ergonomic for feature engineering), Dask (over-engineered
for our scale).

**Why.** Polars is 5-20x faster than pandas for groupby / join / aggregation,
handles parquet natively, and lazy execution lets us write declarative
pipelines. Pandas only enters via `.to_pandas()` for plotting libraries that
need it.

**Outcome.** Polars carried 10M rows without sweat. `group_by("username")
.agg(...)` over the full dataset finishes in seconds.

---

### 1.3 FastAPI for the backend

**Context.** Need a small HTTP service that loads the trained model in memory
and serves a `/recommend/{username}` endpoint.

**Decision.** FastAPI.

**Alternatives.** Flask (more legacy, sync-only), Starlette (FastAPI's base —
unnecessary boilerplate), Django (way too heavy for one endpoint), Streamlit
(would have been faster but signals "demo project").

**Why.** Async-native, type-hinted routes, free OpenAPI docs, ergonomic.
Lifespan hook for loading the K-Means + scaler + recommendations.json once
at startup (not per-request).

**Outcome.** ~80 lines for the whole API including static file mount.

---

### 1.4 Vanilla HTML/CSS/JS frontend

**Context.** Need a UI for the recommender.

**Decision.** Single `index.html` + `style.css` + `app.js`.

**Alternatives.** Streamlit (the "I gave up on UI" choice), React/Next.js
(over-engineered for one form + two tables), HTMX (interesting but adds a
dependency for no clear win at this size).

**Why.** Recruiters opening the deployed site should see something polished,
not a demo template. Three static files, no build step, no runtime
dependencies. Loads in 50 ms.

**Outcome.** ~150 lines of HTML/CSS/JS total. Lighthouse-friendly.

---

## 2. Data collection

### 2.1 Lichess REST API first, database dumps later

**Context.** Need ~10k+ rated games to build a clustering model. Lichess
offers a public REST API and monthly PGN dumps (~30 GB each, ~90 M games per
month).

**Decision.** Start with the REST API; migrate to dumps when the pipeline
worked end-to-end.

**Alternatives.**
- Start with dumps (~30 GB download, complex parsing).
- Use chess.com data (closed API, less standard format).

**Why.** API gives instant feedback for one player and works at small scale
without infrastructure. Dumps unlock millions of games but require zstd
streaming + PGN parsing + multiprocessing. Sequence: working pipeline first,
scale later.

**Outcome.** API got us a 27k-game dataset and a deployed recommender within
hours. The eventual migration to dumps was clean because the schema was the
same.

**Reversal.** The API became a tax after a few hours of work: rate-limits,
half-open connections, multi-minute Retry-After waits. By session 2 we
switched to dumps as the primary training source. The API path remains for
the live recommender endpoint (it has to fetch the *user's* games on demand).

---

### 2.2 Two-stage scrape (top players + opponents)

**Context.** Lichess's `/api/player/top/{n}/{perfType}` returns the top
players of one time control. Top-only is rating-skewed: everyone is 2400+.

**Decision.** Stage 1: top 100 per perf type (bullet/blitz/rapid/classical/
correspondence). Stage 2: sample lower-rated opponents from Stage 1 games to
broaden the rating distribution.

**Alternatives.** Single-stage top-only (rating range too narrow), team-based
scraping (Lichess teams API), random user pull (no such endpoint).

**Why.** Without Stage 2, the dataset is all super-GMs. The clusters would
just sort by rating, not by style. Stage 2 captures opponents from various
rating bands by piggybacking on top-player game lists.

**Outcome.** Got to 284 players spanning ratings 784-3014, which was enough
to validate the pipeline. Later replaced by the database-dump path which has
natural rating diversity.

---

### 2.3 Skill adjustment via Elo expected score

**Context.** Raw `win_rate` correlates almost perfectly with rating. Strong
players win more, period. A clustering model on raw win rate would just
re-discover the rating column.

**Decision.** Use **score residual** instead — `actual_score - elo_expected_score`,
where `elo_expected_score = 1 / (1 + 10^((opp_rating - user_rating) / 400))`.

**Alternatives.** Drop rating-correlated features (loses information),
include rating as a covariate (still doesn't isolate style).

**Why.** Residual measures how well a player performs *relative to their
Elo*. A 1400 player who over-performs by +0.10 per game is interesting for
the same reason as a 2400 player who does — both are over-performing for
their level. Orthogonal to raw strength by construction.

**Outcome.** This is the cleverest feature in the project. Score residual is
what makes the "Underrated overperformer" cluster appear consistently across
data scales.

---

### 2.4 Parquet over CSV / JSON / pickle

**Context.** Need a serialisation format for ~10 M-row datasets.

**Decision.** Parquet.

**Alternatives.** CSV (huge, text-heavy, slow to read), JSON (worse), pickle
(insecure, format unstable across Python versions), feather (less ecosystem
support).

**Why.** Columnar, compressed (snappy by default), typed schema preserved,
read by polars / pandas / DuckDB / Spark with no glue. 10 M rows → 738 MB
parquet → reads in 2 s on this Mac.

**Outcome.** No regrets. The dataset can be inspected with `polars` from any
machine.

---

## 3. Feature engineering

### 3.1 Eight features → thirteen → eighteen

**Context.** Initial feature set was minimal: skill-adjusted residuals,
draw/mate/timeout rates, average move count, opening diversity. Clusters
sorted by rating + outcome patterns and were hard to interpret.

**Decision.** Added features in two waves:

- **Wave 2 (13 features).** Opening-family proportions: `pct_e4_as_white`,
  `pct_d4_as_white`, `pct_sicilian_as_black`. Computed from ECO letter
  prefixes — no PGN parsing needed.
- **Wave 3 (18 features).** Move-level signals parsed from the SAN move
  string via python-chess: `avg_castle_move`, `pct_queenside_castle`,
  `avg_queens_off_move`, `early_pawn_pushes`, `long_game_residual`.

**Alternatives.** Add Stockfish-derived features (centipawn loss, sacrifice
detection) — see [3.3](#33-deferred-tier-3-stockfish-features).

**Why.** The user articulated their playstyle ontology (aggression, sacrifice
propensity, solidity, tactical vs positional). The cheap proxies for those
are move-level patterns. Move parsing runs at ~2,800 games/s — even 5 M
games are processable in ~30 minutes.

**Outcome.** Wave 2 unlocked clusters separated by opening choice
(`1.d4 specialist`, `1.e4 dogmatist`). Wave 3 unlocked the
`Queenside king-hunter` cluster — invisible without castle-side and
castle-timing features.

---

### 3.2 Don't add features blindly — wait for a mental model

**Context.** Tempting to throw every imaginable feature at the clustering
problem and hope something separates.

**Decision.** Defer feature additions until the user could articulate which
style dimensions they wanted to capture.

**Alternatives.** Brute-force feature engineering (every chess.com /
Lichess-exposed signal).

**Why.** Each feature adds a dimension to a space where silhouette is
already low. Random features make clusters worse, not better. The user wrote
out their mental model of chess playstyle (aggression / sacrifice / solidity
/ tactical vs positional) — features then map back to those concepts.

**Outcome.** The 18-feature set is opinionated and defensible. Every feature
corresponds to a chess concept; none are there because "more data is
better".

---

### 3.3 Deferred: Tier-3 Stockfish features

**Context.** The strongest possible playstyle signals come from engine
analysis: centipawn loss (accuracy), sacrifice detection (eval-stable
material drops), tactical hit rate.

**Decision.** Not in scope for Phase 2/3. Phase 3 *will* extract accuracy
from the `[%eval ...]` annotations already present in ~10-15% of dump games
— Lichess analysed them once, we just have to parse.

**Alternatives.** Run Stockfish locally over all 5 M games (~2-3 days at
depth 15-20 on this Mac), commit to Phase 4 territory.

**Why.** The marginal value of locally-analysed Stockfish data over
Lichess's pre-computed `[%eval]` is small. The 10-15% sub-sample of
analysed games gives ~500 k-700 k games — enough for an accuracy-aware
cluster overlay.

**Outcome.** Not built yet, but the plan is concrete.

---

## 4. Clustering

### 4.1 K-Means with StandardScaler

**Context.** Need to partition players into stylistic groups.

**Decision.** K-Means after `StandardScaler`. K chosen by sweep over 2..10
inspecting inertia and silhouette.

**Alternatives.** Gaussian Mixture Models (soft clusters, probabilistic
assignments), HDBSCAN (auto-finds cluster count, handles outliers),
hierarchical / agglomerative (visualisable dendrogram), no clustering at
all (continuous similarity-based recommendations).

**Why.** K-Means is the simplest defensible choice, well-known, easy to
explain in an interview, and produces hard cluster labels for the
recommender lookup. StandardScaler is mandatory because features have very
different scales: `avg_castle_move` spans 9-56, `score_residual` spans
-0.4 to +0.6.

**Outcome.** Works. K=5 produces interpretable identities. But silhouette
stays low across all data scales — see [4.3](#43-chess-style-is-a-continuum-not-clusters).

---

### 4.2 K=4 → K=5 as features and data scaled

**Context.** Choice of K depends on the data.

**Decision.** First fit was K=4 with 8 features on 281 players. After adding
opening-family features (wave 2), K=5 became the elbow. K=5 held when the
data scaled to 5 M games and 154 k players.

**Alternatives.** Stick with K=4 indefinitely (loses one identity), explore
much higher K (clusters become unintelligible past 7).

**Why.** Each time we re-fit we re-evaluated. The "1.d4 specialist" cluster
emerged at K=5 specifically — a meaningful identity that K=4 couldn't
isolate.

**Outcome.** K=5 is stable across data scales. Cluster identities adjust
naturally:

| # | n      | Rating | Identity                       |
| - | ------ | ------ | ------------------------------ |
| 0 | 28,956 | 1,431  | Quick 1.e4 amateur             |
| 1 | 28,558 | 1,540  | Underrated 1.e4 overperformer  |
| 2 | 40,932 | 1,789  | 1.e4 grinder                   |
| 3 | 20,803 | 1,415  | Queenside king-hunter          |
| 4 | 35,261 | 1,740  | 1.d4 specialist                |

**Is 5 enough? We tested K=10** (`scripts/inspect_k.py --k 10`). Findings:

- Silhouette *drops* 0.083 → 0.070 — no clustering-quality gain, as the
  continuum (§ 4.3) predicts.
- Mean dominant-K5 share per K=10 cluster is 73% — most of the extra
  clusters are **rating splits** of the existing five (the same style at a
  higher/lower band: e.g. the 1.d4 specialist splits into a ~1566 and a
  ~1811 version; Quick amateur into ~1348 and ~1545).
- But **one genuinely new identity surfaces**: a *Sicilian specialist*
  (92% Sicilian as Black, ~1847) that K=5 had spread across the grinder
  and d4-specialist clusters. Two weaker behavioural sub-types also appear
  — a high-timeout e4 player (39%) and a high-mate aggressor (49%).

Read: "Black defence choice" is an axis K=5 under-resolves. K=10 doesn't
unlock hidden structure (it's still slicing one cloud), but the Sicilian
specialist is a real, nameable group worth knowing about. For the
production model K=5 stays — interpretable, replicable, stable. The
Sicilian finding is a candidate for a future dedicated feature rather than
a reason to raise K.

![K=10 characterisation](docs/figures/10_k10_characterisation.png)

---

### 4.3 Chess style is a continuum, not clusters

**Context.** Silhouette score stayed around 0.08-0.13 across every
experiment: 281 players, 5 k players, 154 k players, 8 features, 13
features, 18 features.

**Decision.** Document this as a finding rather than fight it.

**Alternatives.** Try harder to push silhouette up (add more features,
re-engineer the loss).

**Why.** A silhouette of ~0.1 is mathematically poor *and* doesn't move
with more data or features. The most likely explanation is that
chess playstyle is a smooth manifold, not a partition. K-Means imposes hard
borders on a cloud that doesn't naturally have them.

The interesting consequence: more data didn't help. The bottleneck wasn't
sample size, it was that the model's assumption was wrong about the geometry
of the problem.

**Outcome.** This is the headline finding of the project. Honest in the
README, defensible in an interview. The next iteration would experiment with
GMM (soft clusters) or continuous similarity instead of clustering.

---

## 5. Recommendation UX

### 5.1 Drop raw win-rate percentages

**Context.** First version of the recommender showed "85.7%" win rate next
to each opening.

**Decision.** Remove the percentage. Show rank + sample size only.

**Alternatives.** Keep the percentage with a disclaimer, show "expected win
rate for your rating" (computed via Elo from the user's actual rating).

**Why.** The percentage is the win rate of *cluster members* against
*cluster-level opposition*. A 1300 player projected into a cluster of 2300s
won't suddenly score 85% with the London System. Surfacing the number as if
it applied to the user would be dishonest.

**Outcome.** Cleaner UX, less false promise. The number in parentheses is
now the *sample size* — a confidence indicator, not a prediction.

---

### 5.2 Switch ranking metric to score residual

**Context.** Ranking by raw win-rate favoured openings that cluster members
happened to play against weaker opponents.

**Decision.** Rank by mean **score residual** (actual − Elo-expected),
consistent with the feature-engineering choice in [2.3](#23-skill-adjustment-via-elo-expected-score).

**Alternatives.** Raw win rate (simpler, biased), Wilson lower bound (good
for small samples but harder to explain).

**Why.** Same opening played against same-strength opposition produces a
"how well did this player do relative to expectation" signal. Surfaces the
opening's *true* effect on the cluster's performance.

**Outcome.** Quality of the ranking went up, especially for clusters with
high rating spread.

---

### 5.3 Aggregate to opening family, not sub-variation

**Context.** Recommendations were emitting things like "Sicilian Defense:
Najdorf Variation, Poisoned Pawn Variation, Main Line, ABBA Variation" —
the chess world has thousands of named sub-lines.

**Decision.** Group recommendations by **family** (the text before the first
`:` in the opening name). "Sicilian Defense: X, Y, Z" all collapse into
"Sicilian Defense".

**Alternatives.** Show full names (opaque), show ECO codes only
(uninformative outside chess geeks).

**Why.** Family-level names are what any club player recognises. Aggregating
also increases sample size per recommendation by 1-2 orders of magnitude,
making the residual estimate stable.

**Outcome.** Recommendations now look like:

```
White                  Black
Italian Game           Sicilian Defense
Scotch Game            French Defense
King's Pawn Game       Caro-Kann Defense
Queen's Pawn Game      Scandinavian Defense
Ruy Lopez              Philidor Defense
```

Names anyone can act on.

---

### 5.4 Opening → color classifier (with two debugging stops)

**Context.** "Suggested openings as White" was returning defenses like
Tarrasch and Lion. Defenses are Black's strategic choices; White can't
*choose* what Black plays.

**Decision.** Three iterations:

1. **Regex on the name** — exclude "Defense" / "Defence" / "Indian" from
   White recommendations. Missed cases (Sicilian sub-variations).
2. **Replay each opening's canonical PGN, classify by whose move was last**
   — built from `github.com/lichess-org/chess-openings`. Worked for
   sub-variations but was visually confusing ("Italian Game: Two Knights
   Defense, Max Lange Attack" classified White, but the name reads
   "Defense").
3. **Classify by the family** (text before the first colon). "Sicilian
   Defense" → Black for all sub-variations. With full-name fallback for
   edge cases where the family isn't its own TSV entry (e.g., "King's
   Gambit Declined" only exists as ":<X>" sub-variants).

**Alternatives at step 2.** Could have just hand-curated a list of
~50 opening families.

**Why.** Iteration 1 was wrong on sub-variations. Iteration 2 was correct
but UX-confusing. Iteration 3 aligns with how chess players actually think
about repertoire ("I play the Sicilian", not "I play the
Sicilian-Najdorf-English-Attack").

**A subtle bug found along the way.** The build script was overwriting
classifications when the same name appeared in the TSV with multiple PGNs
(e.g. "Scandinavian Defense" with `1.e4 d5` AND `1.e4 d5 2.b3`). The second
entry won, so Scandinavian, Dutch, Nimzo-Indian etc. were all stored as
"white". Fix: keep the entry with the shortest PGN — that's the canonical
root.

**Outcome.** 99.4% of opening names in the dataset are now classified by
direct DB lookup; the rest fall back to the regex.

---

### 5.5 Bayesian shrinkage on score residual

**Context.** Even after family aggregation, the top-ranked recommendations
were obscure: Borg Defense, Elephant Gambit, Anderssen's Opening. These
have *huge* score residuals, but they're rare openings played by a
self-selecting tiny minority who study them deeply.

**Decision.** Apply Bayesian shrinkage to the score residual:

```
shrunk_residual = (n / (n + k)) * raw_residual,  k = 30_000
```

**Alternatives.** Hard min-sample threshold (which we also use, at 5,000),
popularity weighting (`residual * sqrt(n)`), Wilson interval.

**Why.** Without shrinkage, an opening with 2 k games and residual +0.30
beats an opening with 100 k games and residual +0.04 — even though the
former is mostly noise / self-selection. Shrinkage pulls small-sample
residuals toward 0 (the population mean) and keeps large-sample estimates
intact.

The choice of `k = 30_000` is calibrated: openings with ~30 k samples are
weighted at 50%, mainstream openings (50 k+ samples) at 60%+, obscure
gambits (2-3 k samples) at <10%.

**Outcome.** Top recommendations are now consistently mainstream openings.
Borg Defense and friends still appear occasionally in cluster Black
recommendations (their residual after shrinkage is still positive), which
is defensible — they *are* over-performing for the players who play them.

---

## 6. Scaling and performance

### 6.1 Streaming PGN, not full-buffer

**Context.** A monthly Lichess dump is ~30 GB compressed, ~150 GB
decompressed.

**Decision.** Stream-decompress via `zstandard.ZstdDecompressor.stream_reader`
+ wrap in `io.TextIOWrapper`. Walk the stream game-by-game.

**Alternatives.** Decompress to disk first (~150 GB temp file), load into
memory (impossible).

**Why.** Peak memory of the extractor stays under 1 GB regardless of input
size. Same code handles a 28 GB monthly file or a hypothetical 360 GB
annual file.

**Outcome.** Trivially scales to the full Phase-4 dataset whenever we want
it.

---

### 6.2 Multiprocessing for PGN parsing — 10x speedup

**Context.** Single-threaded extraction ran at ~990 games/s. A 5 M-game
target was ~2.5 hours. The Python process was at 99% CPU on *one* core out
of 14 available on an M-series Mac.

**Decision.** Producer-consumer with `multiprocessing.Pool`:
- Main thread reads the zstd stream and slices it into per-game text strings
  (cheap, single-threaded).
- 14 worker processes parse PGN with python-chess in parallel.
- Results stream back via `imap_unordered`.

**Alternatives.** Threading (blocked by Python GIL during parsing),
async/await (no benefit — workload is CPU-bound), Rust port (huge effort).

**Why.** The PGN parser itself is the bottleneck. Spreading it across
workers is embarrassingly parallel. Worker init via `Pool(initializer=)`
ships the filter config once per worker (not per task).

**Outcome.** Extraction rate went from ~990 games/s to ~9,800 games/s
(10x). 5 M-game runs dropped from 2.5 hours to ~16 minutes.

Real parallel efficiency: ~61% of theoretical max (14 workers × 990/s).
The remaining ~40% is split between zstd decompression being
single-threaded, IPC pickle overhead, and main-thread aggregation.

---

### 6.3 Hardening the API collector after multi-hour hangs

**Context.** Multiple early `collector.py` runs sat at 0% CPU for an hour
without crashing. Cause: `requests.get(..., stream=True)` + `iter_lines()`
holding a half-open socket whose other end had been closed.

**Decision.** Several layers:
- Drop streaming for the games endpoint — the response is < 1 MB anyway.
- Use a tuple timeout `(connect=10, read=30)` instead of a single value.
- `requests.Session` with `urllib3.util.Retry` configured for 429/5xx
  responses with exponential backoff and `respect_retry_after_header=True`.
- Explicit `print(..., flush=True)` heartbeats every 10 users so progress
  is visible when stdout is redirected via `nohup`.
- `python -u` flag for unbuffered output across the board.

**Alternatives.** Build a SIGALRM-based hard timeout (complex, signal
handlers are global), wrap each request in `concurrent.futures` with a
deadline.

**Why.** The original hang wasn't catchable by `requests`'s timeout because
the read timeout resets between bytes — a server sending 1 byte every 29 s
would never trigger it. Non-streaming + retry-on-failure side-steps the
class of bug entirely.

**Outcome.** No more silent hangs. Errors surface as HTTP errors that the
retry handler either resolves or escalates.

---

### 6.4 Switch from API to Lichess monthly dumps

**Context.** After multiple API hangs, debugging cycles, and rate-limit
backoffs, the API was costing us hours per data refresh. A monthly Lichess
dump is one 28 GB download.

**Decision.** Make the dump the primary training data source. Keep the API
only for the live recommender (it has to fetch the user's recent games on
demand).

**Alternatives.** Stick with API + better caching.

**Why.** Dumps eliminate rate limits entirely, give orders of magnitude
more data (90 M games per month vs 28 k from API), and are infinitely
re-processable locally. The migration cost was a single new module
(`dump_extract.py`) producing the same schema as the API collector — the
rest of the pipeline didn't change.

**Outcome.** Single download + single 16-min extraction replaces hours of
API scraping. 5 M kept games / 154 k players with ≥20 games each — 30x more
training data than the API path got us in twice the wall time.

---

### 6.5 Cap at 5 M kept games, not full dump

**Context.** A month dump has ~90 M games. With our filters (rated +
standard + blitz/rapid/classical + rating band), ~60% pass — ~54 M
"keepable" games.

**Decision.** Cap extraction at 5 M games for now.

**Alternatives.** No cap (130 GB output parquet, ~50 hours of feature
parsing).

**Why.** Diminishing returns. 5 M games yields 154 k players with ≥20 games
each. That's already two orders of magnitude more than we need to train
stable K-Means clusters — silhouette doesn't budge between 5 k and 154 k
players. Spending compute on the full 54 M would be vanity scaling.

**Outcome.** Storage stays modest (738 MB parquet). The cap is a CLI flag,
easy to bump for Phase 4.

---

## 7. Honest findings and deferred work

### Findings worth surfacing

- **Chess style is a continuum.** Silhouette pinned at ~0.08-0.13 across
  every data scale and feature set. K-Means is producing useful labels but
  not natural clusters.
- **Score residual is the single most valuable feature.** Without it, every
  cluster sorted by rating. With it, clusters separate on style.
- **Move-level features unlock identities outcome features can't.** The
  "Queenside king-hunter" cluster — defined by 41% queenside castles at
  move 14+, 38% mate rate — was invisible until castle-side data entered
  the feature set.
- **Obscure openings have inflated score residuals.** Self-selection bias:
  only the rare players who study Borg Defense play it, and they study it
  deeply, so they over-perform. Bayesian shrinkage corrects for this.
- **Accuracy (ACPL) carries style signal beyond rating** — Phase 3 result.
  See § 8.

### Things deliberately deferred

| Deferred work               | Why now isn't the right time                         |
| --------------------------- | ---------------------------------------------------- |
| Public deployment           | Local works end-to-end. Deploy is 1-2 h of work.     |
| Notebook re-render          | Existing figures still tell the right story; will refresh before showcase. |
| Phase 3 — `[%eval]` parsing | Phase 2 work first, then add accuracy as overlay.    |
| Phase 4 — 12-month dataset  | 5 M games is enough for clustering; multi-month adds temporal-trend analysis, separate project. |
| Alt-algorithm comparison    | GMM / HDBSCAN — only worth doing if we commit to the continuum-modelling pivot. |

### Things we explicitly chose NOT to do

- **Stockfish-derived features for the main pipeline.** ~50 hours of
  compute for marginal gain over the cheaper `[%eval]` approach.
- **A Rust/C++ PGN parser.** Python multiprocessing got us to ~10 k
  games/s, which is fast enough. Diminishing returns from a port.
- **More than 5 clusters.** K=5 captures the meaningful divisions; K=6+
  splits one cluster in two without adding interpretability.

---

## 8. Phase 3 — accuracy from `[%eval]` annotations

### 8.1 Use Lichess's pre-computed evals, not local Stockfish

**Context.** The strongest playstyle signals come from engine analysis —
accuracy, blunder rate, centipawn loss. Running Stockfish locally over 5 M
games would take ~2-3 days even parallelised.

**Decision.** Parse the `[%eval ...]` comments that Lichess already embeds
in analysed games. ~12% of dump games carry them (verified: 13.5% of our
extracted games).

**Why.** Lichess analysed those games once, for free. python-chess reads the
evals via `node.eval()` in the same mainline pass that reconstructs SAN —
zero extra engine compute. The 12% sub-sample still yields 1.35 M analysed
games and accuracy data for 87% of clustered players (a player needs only
one analysed game among their 20+).

**Outcome.** ACPL (average centipawn loss, capped [0, 1000] per move,
weighted across a player's analysed games) and blunder rate, computed during
extraction at no measurable speed cost.

### 8.2 Accuracy is metadata, not a clustering feature

**Context.** Could add ACPL to the 18-feature clustering vector.

**Decision.** Keep it as metadata (`acpl`, `blunder_rate`,
`n_analyzed_games` columns alongside `n_games` and `avg_rating`), not in
`FEATURE_COLUMNS`.

**Why.** The live recommender fetches a user's games from the REST API,
which doesn't return evals — so a live user has *no* accuracy data. If ACPL
were a clustering feature it would be null at inference and filled with the
population mean, contributing nothing. Better to keep clustering on signals
available for everyone, and use accuracy as an enrichment overlay.

### 8.3 Finding: accuracy carries style signal beyond rating

The payoff. ACPL correlates with rating at −0.55 (accuracy *is* a big part
of strength), but the cluster differences persist **within a fixed rating
band**, proving style — not strength — drives them.

All players at ~1428 rating (band 1350-1500):

| Cluster                       | ACPL | Blunder rate |
| ---                           | ---  | ---          |
| C1 Underrated 1.e4 overperf.  | 70.4 | 8.5%         |
| C4 1.d4 specialist            | 73.4 | 8.9%         |
| C0 Quick 1.e4 amateur         | 74.8 | 9.3%         |
| C2 1.e4 grinder               | 77.8 | 9.4%         |
| C3 Queenside king-hunter      | 86.2 | 10.5%        |

Two independent validations of the cluster identities:

- The **Queenside king-hunter** is ~16 cp/move less accurate than the
  Underrated cluster *at the same rating* — it genuinely trades precision
  for attacking chances, exactly what an aggressive sacrificial style should
  look like.
- The **Underrated overperformer** is the most accurate in the band,
  consistent with its +0.11 score residual: it really does play above its
  rating level.

This is the strongest evidence in the project that the clusters capture real
playstyle and not just rating buckets — accuracy was never an input to the
clustering, yet it separates the clusters in a way the identities predict.

---

## 9. Validating the continuum hypothesis (alt clustering)

### 9.1 The claim under test

K-Means silhouette sat at ~0.08 across every data scale and feature set.
That's consistent with "playstyle is a continuum, not discrete clusters" —
but low silhouette alone is weak evidence (K-Means could just be a poor fit
while real clusters exist). So we stress-tested with two independent
algorithms (`scripts/clustering_comparison.py`).

### 9.2 Controlling for the curse of dimensionality

First attempt ran GMM and HDBSCAN on the raw 18-dim standardised features.
The results were misleading: GMM gave max-responsibility ≈ 1.0 (looks like
crisp clusters) and HDBSCAN labelled ~100% of points as noise. Both are
**high-dimensional artifacts** — in 18-dim, full-covariance Gaussian
densities become extreme (overconfident assignments) and Euclidean
distances concentrate (density-based methods break).

Fix: run GMM and HDBSCAN on a **PCA-6 projection** (65% of variance), where
probabilities and density are meaningful. Now an "all-noise" or
"hard-assignment" result reflects the data's geometry, not the metric
breaking down.

### 9.3 Result: three algorithms converge on continuum

On the PCA-6 projection (30k-player subsample):

| Method   | Signal                                              | Reading                          |
| ---      | ---                                                 | ---                              |
| K-Means  | silhouette 0.08–0.10, flat across K=2..8            | no crisp boundaries              |
| GMM      | BIC nearly flat (591k → 583k, −1.4% from K=2 to 8); max responsibility decays 0.97 → 0.71; entropy rises 0.14 → 0.37 | no natural K; components overlap |
| HDBSCAN  | 82% noise at min_cluster_size 50–100; 100% noise at ≥250 | no density-separated groups      |

The flat GMM BIC is the cleanest piece: if there were K natural clusters,
BIC would show a clear knee at K. It doesn't — adding components barely
improves the fit, exactly what you'd expect when tiling a smooth cloud with
more Gaussians.

![Clustering comparison](docs/figures/08_clustering_comparison.png)

### 9.4 What this means for the project

The clusters are **useful labels imposed on a continuum**, not discovered
natural groups. That's a legitimate and common situation — the K-Means
clusters are still interpretable and drive sensible recommendations (the
Phase 3 accuracy finding shows they track real style). But the honest
framing is "we partition a continuous style space into 5 reference regions",
not "chess players fall into 5 types".

A genuinely continuum-native model would skip hard clustering entirely:
recommend openings from the *k nearest neighbours* in style space, or learn
a continuous style embedding. That's the natural next architecture if the
project is taken further.

---

## 10. Reproducibility on a disjoint sample

### 10.1 The test

Everything so far was trained on the first 5M kept games of the April-2026
dump. To check the findings aren't an artifact of that particular slice,
`dump_extract --skip 8_500_000` drew a second, **disjoint** 5M-game sample
(games 8.5M–16.5M of the stream). `scripts/compare_samples.py` then ran the
full analysis on both and matched clusters across runs by nearest centroid
(Hungarian assignment — K-Means labels are arbitrary).

Note the two samples are disjoint in *games* but overlap ~59% in *players*
(a player active across the month appears in both, represented by different
games). That's the right setup for testing whether a player's *style* is
stable: do different games of the same player, plus a different population
mix, reproduce the same cluster structure?

### 10.2 Result: the structure replicates almost exactly

| Cluster (matched)        | Rating v1/v2 | score_res v1/v2 | key feature v1/v2 | ACPL v1/v2 |
| ---                      | ---          | ---             | ---               | ---        |
| Quick 1.e4 amateur       | 1431 / 1441  | −0.05 / −0.06   | 84% / 83% e4      | 78.6 / 78.4|
| Underrated overperformer | 1540 / 1528  | +0.11 / +0.11   | 83% / 83% e4      | 70.6 / 71.3|
| 1.e4 grinder             | 1789 / 1793  | −0.02 / −0.02   | 86% / 86% e4      | 68.2 / 68.2|
| Queenside king-hunter    | 1415 / 1414  | −0.00 / −0.00   | 41% / 42% O-O-O   | 91.2 / 91.9|
| 1.d4 specialist          | 1740 / 1753  | +0.01 / +0.01   | 48% / 48% d4      | 66.6 / 66.0|

All five clusters reappear with near-identical profiles — ratings within
~12 points, opening percentages within 1–2 points, score residual identical
to two decimals, ACPL within ~1 cp. Silhouette curves overlap; the GMM BIC
curves have the same flat shape (different absolute scale, same "no natural
K" pattern).

The Phase 3 accuracy finding replicates too: the Queenside king-hunter is
the least accurate cluster (~91 cp) and the 1.d4 specialist the most
accurate (~66 cp) in *both* samples, ordering preserved.

### 10.3 What this buys

Two disjoint 5M samples producing the same five clusters, the same
continuum signal, and the same accuracy ordering is strong evidence the
structure is a real property of the Lichess population, not noise from one
slice. Combined with § 9 (it's a continuum) the honest summary is: **the
style space is smooth and stable — K-Means carves it into five reproducible
reference regions.**

---

## Reading list / references

- Lichess open database: <https://database.lichess.org/>
- Lichess chess-openings TSVs: <https://github.com/lichess-org/chess-openings>
- python-chess docs: <https://python-chess.readthedocs.io/>
- polars user guide: <https://docs.pola.rs/>
- "How to choose K in K-Means" — see `chess_coach/cluster.py --evaluate`
  output and the rationale in [4.2](#42-k4--k5-as-features-and-data-scaled).
