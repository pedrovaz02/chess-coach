# Chess Coach

*[🇬🇧 English](README.md) · 🇵🇹 Português*

Recomendador de aberturas que associa um jogador do Lichess a um cluster
estilístico de outros jogadores e sugere as aberturas com que esses jogadores
historicamente têm sucesso. Treinado com **5 M de jogos rated do dump mensal
do Lichess de Abril-2026**, cobrindo **154 510 jogadores**.

O modelo não recomenda aberturas que "são boas" — recomenda aberturas com que
**jogadores que jogam como tu tendem a ter sucesso**. O clustering é construído
sobre features ajustadas à força (skill-adjusted), por isso o estilo é separado
do rating em bruto.

> Racional de engenharia completo de cada decisão do projeto:
> [`DECISIONS.pt.md`](DECISIONS.pt.md).

---

## Exemplo

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

O ranking usa o **score residual ajustado por Elo** (score real − score
esperado dado o rating do adversário) com **shrinkage Bayesiano** aplicado,
para que aberturas obscuras — residual alto mas amostra minúscula, na maioria
viés de auto-selecção — sejam empurradas para baixo a favor das mainstream com
historial comprovado.

Os nomes das aberturas são agregados ao nível da **família** ("Sicilian
Defense", não "Sicilian Defense: Najdorf, Poisoned Pawn, Main Line"), porque as
sub-variações são estatisticamente ruidosas e ilegíveis.

As percentagens de win-rate em bruto são deliberadamente omitidas — reflectiriam
win rates de membros do cluster contra oposição de nível-cluster, não o que *tu*
marcarias. Ver [`DECISIONS.pt.md` § 5](DECISIONS.pt.md#5-ux-das-recomendações).

---

## Como funciona

```
Dump mensal do Lichess (.pgn.zst, ~28 GB)
   |
   v
dump_download.py + dump_extract.py
   |   stream zstd → divide PGN por jogo → pool de 14 workers
   |   de parsing (~10 000 jogos/s num Mac M-series)
   v
data/games.parquet                10 M linhas (2 por jogo mantido)
   |
   v
features.py                       agregação por jogador +
   |                              parsing de movimentos via python-chess
   v
data/features.parquet             154 510 jogadores × 18 features
   |
   v
cluster.py                        StandardScaler → KMeans (K=5)
   |                              random_state=42, n_init=10
   v
data/models/{kmeans,scaler}.joblib
data/players_clustered.parquet
   |
   v
precompute.py                     top aberturas por (cluster, cor),
   |                              com shrinkage Bayesiano + agregação
   v                              por família + filtro do classifier
data/recommendations.json
   |
   v
FastAPI (api.py)  +  frontend estático  →  /recommend/{username}
```

A pipeline tem dois pontos de entrada: o recomendador live (que vai buscar os
jogos mais recentes do utilizador à API REST do Lichess on demand) e a pipeline
de treino (que trabalha a partir do dump mensal).

### Recolha de dados

Para dados de treino usamos o **dump mensal aberto do Lichess** — um ficheiro
`.pgn.zst` por mês, ~28 GB cada, cobrindo ~90 M de jogos. A descompressão em
streaming (`zstandard.stream_reader` + divisão de PGN por jogo) mantém o pico
de memória abaixo de 1 GB, independentemente do tamanho do input.

A extracção é paralelizada pelos cores do CPU: a thread principal divide o
stream em texto por jogo, um `multiprocessing.Pool` de 14 workers parseia cada
jogo com python-chess em paralelo (~10 000 jogos/s observados). Uma extracção de
5 M jogos demora ~16 minutos num Mac M-series.

Filtros aplicados na extracção: apenas jogos rated standard, em
blitz/rapid/classical, com Elo de ambos os jogadores presente, ambos dentro de
uma banda de rating configurável, e com info de ECO + abertura presente.

Para o recomendador live continuamos a usar a **API REST do Lichess** para ir
buscar os jogos recentes do utilizador — esse caminho foi endurecido com
`requests.Session` + `urllib3.util.Retry` para resiliência contra rate limits e
ligações paradas.

### Features de playstyle

Dezoito features por jogador, organizadas em quatro grupos. A escolha de design
mais importante é o **ajuste à força via score esperado de Elo**: em vez do
`win_rate` em bruto, usamos o *score residual* — score real menos o que o Elo
prevê para o confronto. Um residual de `+0.05` significa "over-performas o teu
Elo em 5 pontos de score por jogo, em média". Isto é ortogonal à força em bruto.

| Grupo       | Feature                  | Significado                                              |
| ---         | ---                      | ---                                                      |
| Performance | `score_residual`         | Ajustado à força, global                                 |
|             | `white_score_residual`   | Idem, só como Brancas                                    |
|             | `black_score_residual`   | Idem, só como Pretas                                     |
|             | `long_game_residual`     | Ajustado à força em jogos ≥ 80 plies (capacidade de final)|
| Resultado   | `draw_rate`              | Fracção de empates                                       |
|             | `mate_rate`              | Fracção que termina em xeque-mate                        |
|             | `timeout_rate`           | Fracção perdida por tempo                                |
|             | `resign_rate`            | Fracção que termina por desistência                      |
| Forma do jogo| `avg_moves`             | Média de lances completos por lado                       |
|             | `short_game_rate`        | Fracção < 40 plies                                       |
| Reportório  | `opening_diversity`      | Códigos ECO únicos / total de jogos                      |
|             | `pct_e4_as_white`        | Como Brancas: % começa 1.e4 (ECO B/C)                    |
|             | `pct_d4_as_white`        | Como Brancas: % começa 1.d4 (ECO D/E)                    |
|             | `pct_sicilian_as_black`  | Como Pretas vs 1.e4: % joga Siciliana                    |
| Tier-1 move | `avg_castle_move`        | Ply médio em que o jogador roca                          |
|             | `pct_queenside_castle`   | Fracção de roques que foram O-O-O                        |
|             | `avg_queens_off_move`    | Ply médio em que ambas as damas saem (simplificação)     |
|             | `early_pawn_pushes`      | Lances de peão nos primeiros 10 lances do jogador        |

As features Tier-1 são parseadas da string de movimentos SAN via python-chess
(~2 800 jogos/s num varrimento single-threaded). São elas que fazem emergir o
cluster `Queenside king-hunter` — invisível sem dados de lado e timing do roque.

Features derivadas de engine (centipawn loss, detecção de sacrifícios) não estão
no modelo; ver [Phase 3 no roadmap](#roadmap).

### Clustering

`StandardScaler` e depois `KMeans(n_clusters=5, random_state=42, n_init=10)`.
K=5 foi escolhido a partir de um sweep sobre K=2..10 (ver `uv run python -m
chess_coach.cluster --evaluate`) — tanto o cotovelo da inertia como o pico do
silhouette alinham em 5.

Os cinco clusters do modelo treinado:

| # | Tamanho| Rating médio | Identidade                                                                  |
| - | ---    | ---        | ---                                                                         |
| 0 | 28 956 | 1431       | **Quick 1.e4 amateur** — 84% e4 com Brancas, decide jogos rápido (~29 lances)|
| 1 | 28 558 | 1540       | **Underrated 1.e4 overperformer** — residual +0.11, +0.15 em jogos longos   |
| 2 | 40 932 | 1789       | **1.e4 grinder** — rating mais alto, jogos mais longos (~38 lances)         |
| 3 | 20 803 | 1415       | **Queenside king-hunter** — 41% O-O-O, roca tarde, 38% mate rate            |
| 4 | 35 261 | 1740       | **1.d4 specialist** — 6% e4 vs 48% d4 — jogador exclusivo de posições fechadas|

Os IDs de cluster são guardados com o modelo; as descrições legíveis das
identidades vivem em `precompute.py:CLUSTER_PROFILES`. Se retreinares com um K
diferente, é preciso re-rotulá-las (`cluster.py --k <N>` e depois inspeccionar o
heatmap-resumo dos clusters).

O heatmap das médias de feature de onde cada identidade é lida (cor = z-score
entre clusters, anotação = média em bruto):

![Caracterização dos clusters](docs/figures/05_cluster_means.png)

A preferência de família de abertura é o sinal comportamental mais forte —
ex. o cluster 4 é quase puro 1.d4, o cluster 0 quase puro 1.e4:

![Preferência de abertura por cluster](docs/figures/06_openings_by_cluster.png)

### Recomendação

Quando um utilizador é projetado num cluster, o recomendador puxa as top
aberturas para esse `(cluster, cor)` do `recommendations.json` — uma tabela de
lookup pré-computada durante o treino.

Cada ranking `(cluster, cor)` passa por três filtros:

1. **Filtro apropriado à cor.** Aberturas classificadas como escolha estratégica
   das Pretas (Sicilian Defense, French Defense, todos os setups Indianos, etc.)
   só aparecem em "as Black"; aberturas lideradas pelas Brancas (Italian Game,
   Ruy Lopez, Polish Opening, Queen's Gambit) só em "as White". O classifier é
   construído a partir dos [TSVs oficiais chess-openings do Lichess](https://github.com/lichess-org/chess-openings)
   reproduzindo o PGN canónico de cada abertura e verificando de quem foi o
   último lance; casos ambíguos caem para lookup pelo nome de família. 99.4% dos
   nomes no dataset são cobertos.

2. **Agregação por família.** Sub-variações ("Sicilian Defense: Najdorf
   Variation, Poisoned Pawn Variation, Main Line") colapsam na família
   ("Sicilian Defense"). Legível para não-mestres, e gera amostras maiores por
   linha.

3. **Shrinkage Bayesiano no residual.** `shrunk = n/(n+30000) × raw`. Sem isto,
   aberturas obscuras (Borg Defense, Elephant Gambit) dominariam o topo porque
   os poucos jogadores que se especializam nelas se auto-seleccionam como
   over-performers. O shrinkage puxa residuais de amostra pequena para 0;
   aberturas mainstream com 50 k+ amostras quase não são afectadas.

---

## Achados honestos

- **O estilo de xadrez é um continuum, não clusters discretos** — confirmado por
  três algoritmos independentes. O silhouette do K-Means fica ~0.08 ao longo de
  281 → 154 510 jogadores e 8 → 18 features. Para descartar "o K-Means é só um
  mau encaixe", corremos GMM e HDBSCAN numa projeção PCA-6 (para evitar
  artefactos de alta dimensão): o BIC do GMM é quase plano (sem K natural) com
  responsabilidades que amolecem à medida que se acrescentam componentes, e o
  HDBSCAN rotula 82–100% dos jogadores como ruído (sem grupos separados por
  densidade). Os três convergem: os clusters são labels úteis impostos num
  espaço de estilo suave, não tipos naturais.
  Ver [`DECISIONS.pt.md` § 9](DECISIONS.pt.md#9-validação-da-hipótese-do-continuum-clustering-alternativo).

![Comparação de clustering](docs/figures/08_clustering_comparison.png)

- **O ajuste à força é a escolha de feature engineering mais valiosa.** Sem ele,
  o `win_rate` em bruto re-descobre a coluna de rating — os clusters apenas
  ordenariam por Elo. O score residual é o que faz o cluster "Underrated
  overperformer" existir de forma consistente em todas as escalas de dados.

- **Features ao nível do lance desbloquearam identidades que features de
  outcome não conseguem ver.** O cluster Queenside king-hunter — 41% de roques
  longos, ply médio de roque 14+, 38% mate rate — é invisível sem dados de
  timing de roque.

- **Aberturas obscuras têm score residuals inflacionados por auto-selecção.**
  Só os raros jogadores que estudam a Borg Defense é que a jogam, e estudam-na a
  fundo, por isso over-performam. O shrinkage Bayesiano corrige isto para que as
  recomendações fiquem ancoradas em aberturas mainstream.

- **A precisão (accuracy) carrega sinal de estilo para além do rating (Phase
  3).** Parsear as anotações `[%eval]` que o Lichess embute em ~12% dos jogos dá
  um centipawn loss médio por jogador (ACPL). Globalmente o ACPL acompanha o
  rating (corr −0.55), mas mantendo o rating fixo (~1428), os clusters ainda
  diferem ~16 cp/lance — o Queenside king-hunter é o menos preciso (troca
  precisão por ataque), o Underrated overperformer o mais preciso. A precisão
  nunca foi input do clustering, e mesmo assim separa os clusters como as
  identidades preveem.
  Ver [`DECISIONS.pt.md` § 8](DECISIONS.pt.md#8-phase-3--precisão-a-partir-das-anotações-eval).

![Precisão por cluster](docs/figures/07_acpl_by_cluster.png)

- **Os achados replicam-se num sample disjunto.** Um segundo sample de 5 M jogos
  (disjunto em jogos, ~59% sobreposição em jogadores) reproduz os cinco clusters
  com perfis quase idênticos (rating ±12, percentagens ±1-2pts, ACPL ±1cp).
  Ver [`DECISIONS.pt.md` § 10](DECISIONS.pt.md#10-reprodutibilidade-num-sample-disjunto).

![Reprodutibilidade entre samples](docs/figures/09_sample_reproducibility.png)

---

## Limitações

- As recomendações **não são previsões de win-rate pessoal**. Reflectem o quão
  bem *membros do cluster* se saíram com cada abertura, contra oposição de
  nível-cluster. Um jogador de 1300 projetado num cluster de 2300s não passa
  subitamente a marcar 85% com o London System.

- Os dados de treino são de **um único mês de calendário** de jogo no Lichess.
  As populações de estilo e as modas de abertura mudam ao longo do tempo — uma
  janela de treino mais longa (Phase 4) suavizaria isto.

- A distribuição de Elo do Lichess **sub-representa ligeiramente ratings muito
  baixos** (< 1200) porque poucos jogadores nessa banda têm ≥ 20 jogos por mês em
  time controls standard.

---

## Setup

```bash
# Instalar uv se ainda não tiveres
brew install uv               # ou: curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependências
uv sync

# ── Pipeline de treino (dados → modelo → recomendações) ───────────────

# 1. Descarregar o dump mensal do Lichess (~28 GB, ~25 min)
uv run python -m chess_coach.dump_download --month 2026-04

# 2. Extrair jogos que passam os filtros (~16 min com multiprocessing)
uv run python -m chess_coach.dump_extract \
    --input data/dumps/lichess_db_standard_rated_2026-04.pgn.zst \
    --output data/games.parquet \
    --max-games 5000000

# 3. Construir features por jogador (~5-8 min — parsing paralelizado)
uv run python -m chess_coach.features

# 4. Inspeccionar o sweep de K, depois treinar o K escolhido
uv run python -m chess_coach.cluster --evaluate
uv run python -m chess_coach.cluster --k 5

# 5. Pré-computar a tabela recommendations.json
uv run python -m chess_coach.precompute

# ── Inferência ────────────────────────────────────────────────────────

# CLI — recomendar para qualquer utilizador do Lichess
uv run python -m chess_coach.recommender <lichess_username>

# Ou correr a web app local
uv run uvicorn chess_coach.api:app --port 8000
# depois abrir http://localhost:8000
```

Os artefactos do modelo treinado (`data/models/`, `data/recommendations.json`,
`data/players_clustered.parquet`) são precisos no momento de inferência mas estão
gitignored — podes correr a pipeline de treino uma vez ou descarregar um release
snapshot.

---

## Estrutura do projeto

```
chess-coach/
|-- src/chess_coach/
|   |-- dump_download.py        Download em streaming do dump mensal do Lichess
|   |-- dump_extract.py         Parser PGN paralelo → games.parquet
|   |-- collector.py            Recolha via API (fallback do recomendador live)
|   |-- features.py             Vectores de playstyle por jogador (18 features)
|   |-- cluster.py              K-Means com ferramentas de elbow + silhouette
|   |-- precompute.py           Constrói a lookup recommendations.json
|   |-- recommender.py          Pipeline end-to-end para um único utilizador
|   |-- api.py                  Backend FastAPI
|   |-- openings.json           Classifier abertura → cor (dos TSVs do Lichess)
|   `-- hello_lichess.py        Sanity check
|-- scripts/
|   `-- build_openings_db.py    (Re)constrói openings.json do repo chess-openings
|-- static/                     Frontend (HTML/CSS/JS, sem build step)
|-- notebooks/
|   `-- 01_data_exploration.ipynb   EDA + visualização de clusters
|-- docs/figures/               PNGs referenciados neste README e no notebook
|-- data/                       (gitignored) ficheiros parquet + modelo treinado
|-- DECISIONS.pt.md             Log completo do racional de engenharia
`-- pyproject.toml
```

---

## Roadmap

- **Phase 1** — Recomendador end-to-end, CLI local, notebook de EDA. **Feito.**
- **Phase 2** — FastAPI + frontend mínimo, executável localmente. **Feito.**
- **Phase 2.5** — Migrar o treino para dumps mensais do Lichess; paralelizar a
  extracção de PGN; features Tier-1 ao nível do lance; recomendações ao nível da
  família com shrinkage Bayesiano. **Feito.**
- **Phase 3** — Parsear as anotações `[%eval ...]` já presentes em ~10–15% dos
  jogos do dump para extrair accuracy / blunder rate / ACPL como features extra.
  Pré-computadas pelo Lichess, logo sem Stockfish local. **Feito.**
- **Phase 4** — Treinar com 12 meses de dumps para cobertura de tendências
  temporais e um atlas de estilo mais robusto.
- **Phase 5** (especulativo) — Substituir o K-Means por um embedding contínuo
  para resolver o achado do silhouette, ou um modelo de recomendação baseado em
  vizinhos mais próximos no espaço de estilo.

---

## Stack

`uv` · `polars` · `pandas` · `pyarrow` · `zstandard` · `requests` ·
`scikit-learn` · `joblib` · `python-chess` · `matplotlib` · `seaborn` ·
`rich` · `fastapi` · `uvicorn` · `jupyter`
