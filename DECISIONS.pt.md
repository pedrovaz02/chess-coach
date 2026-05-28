# Log de decisões

*[🇬🇧 English](DECISIONS.md) · 🇵🇹 Português*

Este documento regista as decisões de design e engenharia que moldaram o
chess-coach. Cada entrada tem o contexto (qual era a situação), o que
escolhemos, que alternativas considerámos, e o resultado com a vantagem da
retrospectiva.

A intenção é tornar o projeto legível para quem lê o código, e responsabilizar
o racional perante os dados. Várias decisões aqui foram revertidas após
evidência — essas estão mantidas de propósito.

---

## Conteúdo

1. [Stack e ferramentas](#1-stack-e-ferramentas)
2. [Recolha de dados](#2-recolha-de-dados)
3. [Feature engineering](#3-feature-engineering)
4. [Clustering](#4-clustering)
5. [UX das recomendações](#5-ux-das-recomendações)
6. [Escala e performance](#6-escala-e-performance)
7. [Achados honestos e trabalho adiado](#7-achados-honestos-e-trabalho-adiado)
8. [Phase 3 — precisão a partir das anotações `[%eval]`](#8-phase-3--precisão-a-partir-das-anotações-eval)
9. [Validação da hipótese do continuum (clustering alternativo)](#9-validação-da-hipótese-do-continuum-clustering-alternativo)
10. [Reprodutibilidade num sample disjunto](#10-reprodutibilidade-num-sample-disjunto)

---

## 1. Stack e ferramentas

### 1.1 `uv` em vez de `pip` / `poetry`

**Contexto.** Preciso de um gestor de pacotes Python que trate de virtualenvs,
lockfiles e resolução de dependências.

**Decisão.** Usar `uv`.

**Alternativas.** `pip + venv` (manual), `poetry` (mais lento, mais legacy),
`pipenv` (adopção em declínio), `conda` (exagero, só ML).

**Porquê.** Moderno, 10-100x mais rápido que pip, lockfile incorporado, uma só
ferramenta para venv + deps. Standard na comunidade Python de 2025. Sinaliza
literacia de ferramentas actuais.

**Resultado.** Sem arrependimentos. `uv add`, `uv sync`, `uv run` cobriram todos
os workflows.

---

### 1.2 `polars` *e* `pandas`

**Contexto.** Preciso de manipular dados tabulares — inicialmente pequenos (10k
jogos), depois grandes (10M+ linhas).

**Decisão.** `polars` como principal, `pandas` só dentro dos notebooks para
compatibilidade com seaborn/matplotlib.

**Alternativas.** Pandas puro (familiar mas lento em 10M linhas), DuckDB
(SQL-first, menos ergonómico para feature engineering), Dask (over-engineered
para a nossa escala).

**Porquê.** Polars é 5-20x mais rápido que pandas em groupby / join / agregação,
trata parquet nativamente, e a execução lazy permite pipelines declarativas. O
pandas só entra via `.to_pandas()` para bibliotecas de plotting que o exigem.

**Resultado.** Polars aguentou 10M linhas sem esforço. `group_by("username")
.agg(...)` sobre o dataset completo termina em segundos.

---

### 1.3 FastAPI para o backend

**Contexto.** Preciso de um serviço HTTP pequeno que carrega o modelo treinado
em memória e serve um endpoint `/recommend/{username}`.

**Decisão.** FastAPI.

**Alternativas.** Flask (mais legacy, só síncrono), Starlette (a base do FastAPI
— boilerplate desnecessário), Django (demasiado pesado para um endpoint),
Streamlit (teria sido mais rápido mas sinaliza "projeto de demo").

**Porquê.** Async-native, rotas com type hints, docs OpenAPI grátis, ergonómico.
Hook de lifespan para carregar o K-Means + scaler + recommendations.json uma vez
no arranque (não por request).

**Resultado.** ~80 linhas para toda a API, incluindo o mount de ficheiros
estáticos.

---

### 1.4 Frontend em HTML/CSS/JS vanilla

**Contexto.** Preciso de uma UI para o recomendador.

**Decisão.** Um só `index.html` + `style.css` + `app.js`.

**Alternativas.** Streamlit (a escolha "desisti da UI"), React/Next.js
(over-engineered para um formulário + duas tabelas), HTMX (interessante mas
acrescenta uma dependência sem ganho claro a este tamanho).

**Porquê.** Recrutadores que abram o site em produção devem ver algo polido, não
um template de demo. Três ficheiros estáticos, sem build step, sem dependências
em runtime. Carrega em 50 ms.

**Resultado.** ~150 linhas de HTML/CSS/JS no total. Lighthouse-friendly.

---

## 2. Recolha de dados

### 2.1 API REST do Lichess primeiro, dumps depois

**Contexto.** Preciso de ~10k+ jogos rated para construir um modelo de
clustering. O Lichess oferece uma API REST pública e dumps mensais de PGN
(~30 GB cada, ~90 M jogos por mês).

**Decisão.** Começar com a API REST; migrar para dumps quando a pipeline
funcionasse end-to-end.

**Alternativas.**
- Começar com dumps (~30 GB de download, parsing complexo).
- Usar dados do chess.com (API fechada, formato menos standard).

**Porquê.** A API dá feedback instantâneo para um jogador e funciona a pequena
escala sem infraestrutura. Os dumps desbloqueiam milhões de jogos mas exigem
streaming de zstd + parsing de PGN + multiprocessing. Sequência: pipeline a
funcionar primeiro, escalar depois.

**Resultado.** A API deu-nos um dataset de 27k jogos e um recomendador deployado
em horas. A eventual migração para dumps foi limpa porque o schema era o mesmo.

**Reversão.** A API tornou-se um imposto após algumas horas: rate-limits,
ligações half-open, esperas de Retry-After de vários minutos. Na sessão 2
mudámos para dumps como fonte de treino primária. O caminho da API mantém-se
para o endpoint do recomendador live (tem de ir buscar os jogos do *utilizador*
on demand).

---

### 2.2 Scrape em duas fases (top players + adversários)

**Contexto.** O `/api/player/top/{n}/{perfType}` do Lichess devolve os top
players de um time control. Só-topo é enviesado em rating: toda a gente é 2400+.

**Decisão.** Fase 1: top 100 por perf type
(bullet/blitz/rapid/classical/correspondence). Fase 2: amostrar adversários de
rating mais baixo dos jogos da Fase 1 para alargar a distribuição de rating.

**Alternativas.** Só-topo de uma fase (banda de rating estreita demais), scraping
por equipas (API de teams do Lichess), pull de utilizadores aleatórios (não
existe tal endpoint).

**Porquê.** Sem a Fase 2, o dataset é só super-GMs. Os clusters apenas
ordenariam por rating, não por estilo. A Fase 2 captura adversários de várias
bandas de rating à boleia das listas de jogos dos top players.

**Resultado.** Chegámos a 284 jogadores entre ratings 784-3014, suficiente para
validar a pipeline. Mais tarde substituído pelo caminho dos dumps, que tem
diversidade de rating natural.

---

### 2.3 Ajuste à força via score esperado de Elo

**Contexto.** O `win_rate` em bruto correlaciona quase perfeitamente com o
rating. Jogadores fortes ganham mais, ponto final. Um modelo de clustering sobre
win rate em bruto apenas re-descobriria a coluna de rating.

**Decisão.** Usar o **score residual** — `score_real - score_esperado_elo`, onde
`score_esperado_elo = 1 / (1 + 10^((rating_adversário - rating_user) / 400))`.

**Alternativas.** Largar features correlacionadas com rating (perde informação),
incluir rating como covariável (continua a não isolar estilo).

**Porquê.** O residual mede o quão bem um jogador rende *relativamente ao seu
Elo*. Um jogador de 1400 que over-performa +0.10 por jogo é interessante pela
mesma razão que um de 2400 — ambos rendem acima do seu nível. Ortogonal à força
em bruto por construção.

**Resultado.** É a feature mais inteligente do projeto. O score residual é o que
faz o cluster "Underrated overperformer" aparecer consistentemente em todas as
escalas de dados.

---

### 2.4 Parquet em vez de CSV / JSON / pickle

**Contexto.** Preciso de um formato de serialização para datasets de ~10 M
linhas.

**Decisão.** Parquet.

**Alternativas.** CSV (enorme, text-heavy, lento de ler), JSON (pior), pickle
(inseguro, formato instável entre versões de Python), feather (menos suporte de
ecossistema).

**Porquê.** Colunar, comprimido (snappy por defeito), schema tipado preservado,
lido por polars / pandas / DuckDB / Spark sem cola. 10 M linhas → 738 MB de
parquet → lê em 2 s neste Mac.

**Resultado.** Sem arrependimentos. O dataset pode ser inspeccionado com
`polars` de qualquer máquina.

---

## 3. Feature engineering

### 3.1 Oito features → treze → dezoito

**Contexto.** O conjunto inicial de features era mínimo: residuais ajustados à
força, taxas de empate/mate/timeout, contagem média de lances, diversidade de
abertura. Os clusters ordenavam por rating + padrões de outcome e eram difíceis
de interpretar.

**Decisão.** Acrescentei features em duas vagas:

- **Vaga 2 (13 features).** Proporções de família de abertura:
  `pct_e4_as_white`, `pct_d4_as_white`, `pct_sicilian_as_black`. Computadas a
  partir dos prefixos de letra ECO — sem parsing de PGN.
- **Vaga 3 (18 features).** Sinais ao nível do lance parseados da string SAN via
  python-chess: `avg_castle_move`, `pct_queenside_castle`,
  `avg_queens_off_move`, `early_pawn_pushes`, `long_game_residual`.

**Alternativas.** Acrescentar features derivadas de Stockfish (centipawn loss,
detecção de sacrifícios) — ver [3.3](#33-adiado-features-tier-3-de-stockfish).

**Porquê.** O utilizador articulou a sua ontologia de playstyle (agressão,
propensão para sacrifícios, solidez, táctico vs posicional). Os proxies baratos
para isso são padrões ao nível do lance. O parsing de movimentos corre a ~2 800
jogos/s — mesmo 5 M jogos são processáveis em ~30 minutos.

**Resultado.** A Vaga 2 desbloqueou clusters separados por escolha de abertura
(`1.d4 specialist`, `1.e4 dogmatist`). A Vaga 3 desbloqueou o
`Queenside king-hunter` — invisível sem features de lado e timing do roque.

---

### 3.2 Não acrescentar features às cegas — esperar por um modelo mental

**Contexto.** É tentador atirar todas as features imagináveis ao problema de
clustering e esperar que algo separe.

**Decisão.** Adiar a adição de features até o utilizador conseguir articular que
dimensões de estilo queria capturar.

**Alternativas.** Feature engineering por força bruta (todos os sinais que
chess.com / Lichess expõem).

**Porquê.** Cada feature acrescenta uma dimensão a um espaço onde o silhouette já
é baixo. Features aleatórias pioram os clusters, não melhoram. O utilizador
escreveu o seu modelo mental de playstyle (agressão / sacrifício / solidez /
táctico vs posicional) — as features mapeiam depois de volta a esses conceitos.

**Resultado.** O conjunto de 18 features é opinativo e defensável. Cada feature
corresponde a um conceito de xadrez; nenhuma está lá porque "mais dados é
melhor".

---

### 3.3 Adiado: features Tier-3 de Stockfish

**Contexto.** Os sinais de playstyle mais fortes possíveis vêm da análise de
engine: centipawn loss (precisão), detecção de sacrifícios (quedas materiais com
avaliação estável), taxa de acerto táctico.

**Decisão.** Fora de scope para a Phase 2/3. A Phase 3 *vai* extrair precisão das
anotações `[%eval ...]` já presentes em ~10-15% dos jogos do dump — o Lichess
analisou-os uma vez, só temos de parsear.

**Alternativas.** Correr Stockfish localmente sobre todos os 5 M jogos (~2-3 dias
a depth 15-20 neste Mac), entrar em território de Phase 4.

**Porquê.** O valor marginal de dados de Stockfish analisados localmente sobre o
`[%eval]` pré-computado do Lichess é pequeno. O sub-sample de 10-15% de jogos
analisados dá ~500 k-700 k jogos — suficiente para uma camada de cluster
consciente da precisão.

**Resultado.** Ainda não construído na altura, mas o plano era concreto (ver
§ 8).

---

## 4. Clustering

### 4.1 K-Means com StandardScaler

**Contexto.** Preciso de particionar jogadores em grupos estilísticos.

**Decisão.** K-Means após `StandardScaler`. K escolhido por sweep sobre 2..10
inspeccionando inertia e silhouette.

**Alternativas.** Gaussian Mixture Models (clusters macios, atribuições
probabilísticas), HDBSCAN (auto-encontra nº de clusters, lida com outliers),
hierárquico / aglomerativo (dendrograma visualizável), nenhum clustering
(recomendações baseadas em similaridade contínua).

**Porquê.** K-Means é a escolha mais simples e defensável, conhecida, fácil de
explicar numa entrevista, e produz labels de cluster duros para o lookup do
recomendador. O StandardScaler é obrigatório porque as features têm escalas muito
diferentes: `avg_castle_move` vai de 9-56, `score_residual` de -0.4 a +0.6.

**Resultado.** Funciona. K=5 produz identidades interpretáveis. Mas o silhouette
fica baixo em todas as escalas de dados — ver
[4.3](#43-o-estilo-de-xadrez-é-um-continuum-não-clusters).

---

### 4.2 K=4 → K=5 à medida que features e dados escalaram

**Contexto.** A escolha de K depende dos dados.

**Decisão.** O primeiro fit foi K=4 com 8 features em 281 jogadores. Após
acrescentar features de família de abertura (vaga 2), K=5 tornou-se o cotovelo.
K=5 manteve-se quando os dados escalaram para 5 M jogos e 154 k jogadores.

**Alternativas.** Ficar em K=4 indefinidamente (perde uma identidade), explorar K
muito maior (clusters tornam-se ininteligíveis depois de 7).

**Porquê.** Cada vez que re-treinámos, re-avaliámos. O cluster "1.d4 specialist"
emergiu especificamente em K=5 — uma identidade significativa que K=4 não
conseguia isolar.

**Resultado.** K=5 é estável em todas as escalas de dados. As identidades de
cluster ajustam-se naturalmente:

| # | n      | Rating | Identidade                     |
| - | ------ | ------ | ------------------------------ |
| 0 | 28,956 | 1,431  | Quick 1.e4 amateur             |
| 1 | 28,558 | 1,540  | Underrated 1.e4 overperformer  |
| 2 | 40,932 | 1,789  | 1.e4 grinder                   |
| 3 | 20,803 | 1,415  | Queenside king-hunter          |
| 4 | 35,261 | 1,740  | 1.d4 specialist                |

---

### 4.3 O estilo de xadrez é um continuum, não clusters

**Contexto.** O score de silhouette manteve-se ~0.08-0.13 em todas as
experiências: 281 jogadores, 5 k jogadores, 154 k jogadores, 8 features, 13
features, 18 features.

**Decisão.** Documentar isto como um achado em vez de lutar contra ele.

**Alternativas.** Esforçar mais para subir o silhouette (acrescentar mais
features, re-engenheirar a loss).

**Porquê.** Um silhouette de ~0.1 é matematicamente fraco *e* não se move com
mais dados ou features. A explicação mais provável é que o playstyle de xadrez é
uma variedade suave (smooth manifold), não uma partição. O K-Means impõe
fronteiras duras numa nuvem que naturalmente não as tem.

A consequência interessante: mais dados não ajudaram. O gargalo não era o tamanho
da amostra, era a suposição do modelo estar errada sobre a geometria do problema.

**Resultado.** É o achado central do projeto. Honesto no README, defensável numa
entrevista. A próxima iteração experimentaria GMM (clusters macios) ou
similaridade contínua em vez de clustering.

---

## 5. UX das recomendações

### 5.1 Largar as percentagens de win-rate em bruto

**Contexto.** A primeira versão do recomendador mostrava "85.7%" de win rate ao
lado de cada abertura.

**Decisão.** Remover a percentagem. Mostrar só rank + tamanho de amostra.

**Alternativas.** Manter a percentagem com um aviso, mostrar "win rate esperado
para o teu rating" (computado via Elo a partir do rating real do utilizador).

**Porquê.** A percentagem é o win rate de *membros do cluster* contra *oposição
de nível-cluster*. Um jogador de 1300 projetado num cluster de 2300s não passa a
marcar 85% com o London System. Mostrar o número como se se aplicasse ao
utilizador seria desonesto.

**Resultado.** UX mais limpo, menos falsas promessas. O número entre parênteses é
agora o *tamanho de amostra* — um indicador de confiança, não uma previsão.

---

### 5.2 Mudar a métrica de ranking para score residual

**Contexto.** Ordenar por win-rate em bruto favorecia aberturas que membros do
cluster por acaso jogaram contra adversários mais fracos.

**Decisão.** Ordenar pela média do **score residual** (real − esperado-Elo),
consistente com a escolha de feature engineering em
[2.3](#23-ajuste-à-força-via-score-esperado-de-elo).

**Alternativas.** Win rate em bruto (mais simples, enviesado), Wilson lower bound
(bom para amostras pequenas mas mais difícil de explicar).

**Porquê.** A mesma abertura jogada contra oposição da mesma força produz um
sinal "o quão bem este jogador se saiu relativamente à expectativa". Revela o
efeito *verdadeiro* da abertura na performance do cluster.

**Resultado.** A qualidade do ranking subiu, especialmente para clusters com
grande dispersão de rating.

---

### 5.3 Agregar à família de abertura, não à sub-variação

**Contexto.** As recomendações emitiam coisas como "Sicilian Defense: Najdorf
Variation, Poisoned Pawn Variation, Main Line, ABBA Variation" — o mundo do
xadrez tem milhares de sub-linhas nomeadas.

**Decisão.** Agrupar recomendações por **família** (o texto antes do primeiro `:`
no nome da abertura). "Sicilian Defense: X, Y, Z" colapsam todas em "Sicilian
Defense".

**Alternativas.** Mostrar nomes completos (opacos), mostrar só códigos ECO
(pouco informativos fora dos geeks de xadrez).

**Porquê.** Os nomes ao nível da família são o que qualquer jogador de clube
reconhece. Agregar também aumenta o tamanho de amostra por recomendação em 1-2
ordens de grandeza, tornando a estimativa do residual estável.

**Resultado.** As recomendações agora têm este aspecto:

```
White                  Black
Italian Game           Sicilian Defense
Scotch Game            French Defense
King's Pawn Game       Caro-Kann Defense
Queen's Pawn Game      Scandinavian Defense
Ruy Lopez              Philidor Defense
```

Nomes sobre os quais qualquer um consegue agir.

---

### 5.4 Classifier abertura → cor (com duas paragens de debugging)

**Contexto.** "Suggested openings as White" devolvia defesas como a Tarrasch e a
Lion. Defesas são escolhas estratégicas das Pretas; as Brancas não podem
*escolher* o que as Pretas jogam.

**Decisão.** Três iterações:

1. **Regex no nome** — excluir "Defense" / "Defence" / "Indian" das recomendações
   de Brancas. Falhava casos (sub-variações da Siciliana).
2. **Reproduzir o PGN canónico de cada abertura, classificar por quem fez o
   último lance** — construído a partir de
   `github.com/lichess-org/chess-openings`. Funcionava para sub-variações mas era
   visualmente confuso ("Italian Game: Two Knights Defense, Max Lange Attack"
   classificado como Brancas, mas o nome lê "Defense").
3. **Classificar pela família** (texto antes do primeiro dois-pontos). "Sicilian
   Defense" → Pretas para todas as sub-variações. Com fallback ao nome completo
   para casos extremos onde a família não tem entrada própria no TSV (ex.
   "King's Gambit Declined" só existe como sub-variantes ":<X>").

**Alternativas no passo 2.** Podia ter curado à mão uma lista de ~50 famílias de
abertura.

**Porquê.** A iteração 1 errava nas sub-variações. A iteração 2 estava correcta
mas confundia em UX. A iteração 3 alinha com como os jogadores de xadrez pensam
de facto sobre reportório ("eu jogo a Siciliana", não "eu jogo a
Siciliana-Najdorf-English-Attack").

**Um bug subtil encontrado pelo caminho.** O script de build estava a sobrescrever
classificações quando o mesmo nome aparecia no TSV com múltiplos PGNs (ex.
"Scandinavian Defense" com `1.e4 d5` E `1.e4 d5 2.b3`). A segunda entrada ganhava,
por isso a Scandinavian, Dutch, Nimzo-Indian etc. ficavam todas guardadas como
"white". Correcção: manter a entrada com o PGN mais curto — essa é a raiz
canónica.

**Resultado.** 99.4% dos nomes de abertura no dataset são agora classificados por
lookup directo no DB; o resto cai para o regex.

---

### 5.5 Shrinkage Bayesiano no score residual

**Contexto.** Mesmo após a agregação por família, as recomendações no topo eram
obscuras: Borg Defense, Elephant Gambit, Anderssen's Opening. Estas têm score
residuals *enormes*, mas são aberturas raras jogadas por uma minúscula minoria
auto-seleccionada que as estuda a fundo.

**Decisão.** Aplicar shrinkage Bayesiano ao score residual:

```
shrunk_residual = (n / (n + k)) * raw_residual,  k = 30_000
```

**Alternativas.** Threshold mínimo de amostra duro (que também usamos, em 5,000),
ponderação por popularidade (`residual * sqrt(n)`), intervalo de Wilson.

**Porquê.** Sem shrinkage, uma abertura com 2 k jogos e residual +0.30 bate uma
com 100 k jogos e residual +0.04 — apesar de a primeira ser sobretudo
ruído / auto-selecção. O shrinkage puxa residuais de amostra pequena para 0 (a
média populacional) e mantém intactas as estimativas de amostra grande.

A escolha de `k = 30_000` é calibrada: aberturas com ~30 k amostras são ponderadas
a 50%, mainstream (50 k+ amostras) a 60%+, gambits obscuros (2-3 k amostras) a
<10%.

**Resultado.** As recomendações de topo são agora consistentemente aberturas
mainstream. A Borg Defense e companhia ainda aparecem ocasionalmente nas
recomendações de Pretas de um cluster (o residual após shrinkage ainda é
positivo), o que é defensável — *estão* a over-performar para os jogadores que as
jogam.

---

## 6. Escala e performance

### 6.1 PGN em streaming, não buffer completo

**Contexto.** Um dump mensal do Lichess tem ~30 GB comprimido, ~150 GB
descomprimido.

**Decisão.** Descomprimir em streaming via
`zstandard.ZstdDecompressor.stream_reader` + envolver em `io.TextIOWrapper`.
Percorrer o stream jogo a jogo.

**Alternativas.** Descomprimir para disco primeiro (~150 GB de ficheiro
temporário), carregar para memória (impossível).

**Porquê.** O pico de memória do extractor fica abaixo de 1 GB independentemente
do tamanho do input. O mesmo código trata um ficheiro mensal de 28 GB ou um
hipotético ficheiro anual de 360 GB.

**Resultado.** Escala trivialmente para o dataset completo da Phase 4 quando o
quisermos.

---

### 6.2 Multiprocessing para o parsing de PGN — speedup de 10x

**Contexto.** A extracção single-threaded corria a ~990 jogos/s. Um alvo de 5 M
jogos eram ~2.5 horas. O processo Python estava a 99% de CPU em *um* core dos 14
disponíveis num Mac M-series.

**Decisão.** Producer-consumer com `multiprocessing.Pool`:
- A thread principal lê o stream zstd e divide-o em strings de texto por jogo
  (barato, single-threaded).
- 14 processos worker parseiam PGN com python-chess em paralelo.
- Os resultados voltam em stream via `imap_unordered`.

**Alternativas.** Threading (bloqueado pelo GIL do Python durante o parsing),
async/await (sem benefício — workload é CPU-bound), porte para Rust (esforço
enorme).

**Porquê.** O parser de PGN em si é o gargalo. Espalhá-lo pelos workers é
embaraçosamente paralelo. A init dos workers via `Pool(initializer=)` envia a
config dos filtros uma vez por worker (não por tarefa).

**Resultado.** A taxa de extracção passou de ~990 jogos/s para ~9 800 jogos/s
(10x). Runs de 5 M jogos caíram de 2.5 horas para ~16 minutos.

Eficiência paralela real: ~61% do máximo teórico (14 workers × 990/s). Os
restantes ~40% dividem-se entre a descompressão zstd ser single-threaded,
overhead de pickle no IPC, e a agregação na thread principal.

---

### 6.3 Endurecer o collector da API após hangs de horas

**Contexto.** Vários runs iniciais do `collector.py` ficaram a 0% de CPU durante
uma hora sem crashar. Causa: `requests.get(..., stream=True)` + `iter_lines()` a
segurar um socket half-open cuja outra ponta tinha sido fechada.

**Decisão.** Várias camadas:
- Largar o streaming para o endpoint de jogos — a resposta é < 1 MB de qualquer
  forma.
- Usar um timeout em tuplo `(connect=10, read=30)` em vez de um valor único.
- `requests.Session` com `urllib3.util.Retry` configurado para respostas 429/5xx
  com backoff exponencial e `respect_retry_after_header=True`.
- Heartbeats explícitos `print(..., flush=True)` a cada 10 utilizadores para o
  progresso ser visível quando o stdout é redirecionado via `nohup`.
- Flag `python -u` para output não-bufferizado em toda a linha.

**Alternativas.** Construir um timeout duro baseado em SIGALRM (complexo, handlers
de sinal são globais), envolver cada request em `concurrent.futures` com um
deadline.

**Porquê.** O hang original não era apanhável pelo timeout do `requests` porque o
read timeout reinicia entre bytes — um servidor a enviar 1 byte a cada 29 s nunca
o accionaria. Não-streaming + retry-on-failure contorna a classe inteira do bug.

**Resultado.** Sem mais hangs silenciosos. Os erros surgem como erros HTTP que o
handler de retry ou resolve ou escala.

---

### 6.4 Mudar da API para os dumps mensais do Lichess

**Contexto.** Após múltiplos hangs da API, ciclos de debugging e backoffs de
rate-limit, a API custava-nos horas por refresh de dados. Um dump mensal do
Lichess é um download de 28 GB.

**Decisão.** Tornar o dump a fonte primária de dados de treino. Manter a API só
para o recomendador live (tem de ir buscar os jogos recentes do utilizador on
demand).

**Alternativas.** Ficar na API + melhor caching.

**Porquê.** Os dumps eliminam rate limits por completo, dão ordens de grandeza
mais dados (90 M jogos por mês vs 28 k da API), e são infinitamente
re-processáveis localmente. O custo da migração foi um único módulo novo
(`dump_extract.py`) a produzir o mesmo schema que o collector da API — o resto da
pipeline não mudou.

**Resultado.** Um download + uma extracção de 16 min substituem horas de scraping
da API. 5 M jogos mantidos / 154 k jogadores com ≥20 jogos cada — 30x mais dados
de treino do que o caminho da API nos deu em metade do tempo de relógio.

---

### 6.5 Cap em 5 M jogos mantidos, não o dump completo

**Contexto.** Um dump mensal tem ~90 M jogos. Com os nossos filtros (rated +
standard + blitz/rapid/classical + banda de rating), ~60% passam — ~54 M jogos
"mantíveis".

**Decisão.** Limitar a extracção a 5 M jogos por agora.

**Alternativas.** Sem cap (130 GB de parquet de saída, ~50 horas de parsing de
features).

**Porquê.** Retornos decrescentes. 5 M jogos dão 154 k jogadores com ≥20 jogos
cada. Já são duas ordens de grandeza mais do que precisamos para treinar clusters
K-Means estáveis — o silhouette não se mexe entre 5 k e 154 k jogadores. Gastar
compute nos 54 M completos seria escalar por vaidade.

**Resultado.** O armazenamento fica modesto (738 MB de parquet). O cap é uma flag
de CLI, fácil de aumentar para a Phase 4.

---

## 7. Achados honestos e trabalho adiado

### Achados que vale a pena destacar

- **O estilo de xadrez é um continuum.** Silhouette fixado em ~0.08-0.13 em todas
  as escalas de dados e conjuntos de features. O K-Means produz labels úteis mas
  não clusters naturais.
- **O score residual é a feature mais valiosa.** Sem ele, todos os clusters
  ordenavam por rating. Com ele, os clusters separam por estilo.
- **Features ao nível do lance desbloqueiam identidades que features de outcome
  não conseguem.** O cluster "Queenside king-hunter" — definido por 41% de
  roques longos ao lance 14+, 38% mate rate — era invisível até os dados de lado
  do roque entrarem no conjunto.
- **Aberturas obscuras têm score residuals inflacionados.** Viés de
  auto-selecção: só os raros jogadores que estudam a Borg Defense é que a jogam,
  e estudam-na a fundo, por isso over-performam. O shrinkage Bayesiano corrige
  isto.
- **A precisão (ACPL) carrega sinal de estilo para além do rating** — resultado
  da Phase 3. Ver § 8.

### Coisas deliberadamente adiadas

| Trabalho adiado                | Porque agora não é a altura certa                    |
| ------------------------------ | ---------------------------------------------------- |
| Deploy público                 | O local funciona end-to-end. Deploy é 1-2 h de trabalho. |
| Re-render do notebook          | As figuras existentes ainda contam a história certa; refresh antes do showcase. |
| Phase 3 — parsing de `[%eval]` | Trabalho da Phase 2 primeiro, depois accuracy como camada. |
| Phase 4 — dataset de 12 meses  | 5 M jogos chegam para clustering; multi-mês acrescenta análise de tendências temporais, projeto separado. |
| Comparação de algoritmos       | GMM / HDBSCAN — só vale a pena se nos comprometermos com o pivot de modelação de continuum. |

### Coisas que explicitamente escolhemos NÃO fazer

- **Features derivadas de Stockfish para a pipeline principal.** ~50 horas de
  compute para ganho marginal sobre a abordagem mais barata do `[%eval]`.
- **Um parser de PGN em Rust/C++.** O multiprocessing de Python levou-nos a ~10 k
  jogos/s, o que é rápido o suficiente. Retornos decrescentes de um porte.
- **Mais de 5 clusters.** K=5 captura as divisões significativas; K=6+ divide um
  cluster em dois sem acrescentar interpretabilidade.

---

## 8. Phase 3 — precisão a partir das anotações `[%eval]`

### 8.1 Usar os evals pré-computados do Lichess, não Stockfish local

**Contexto.** Os sinais de playstyle mais fortes vêm da análise de engine —
precisão, blunder rate, centipawn loss. Correr Stockfish localmente sobre 5 M
jogos demoraria ~2-3 dias mesmo paralelizado.

**Decisão.** Parsear os comentários `[%eval ...]` que o Lichess já embute em jogos
analisados. ~12% dos jogos do dump trazem-nos (verificado: 13.5% dos nossos jogos
extraídos).

**Porquê.** O Lichess analisou esses jogos uma vez, de graça. O python-chess lê os
evals via `node.eval()` no mesmo varrimento da mainline que reconstrói o SAN —
zero compute extra de engine. O sub-sample de 12% ainda dá 1.35 M jogos analisados
e dados de precisão para 87% dos jogadores clusterizados (um jogador só precisa de
um jogo analisado entre os seus 20+).

**Resultado.** ACPL (centipawn loss médio, capado [0, 1000] por lance, ponderado
pelos jogos analisados de um jogador) e blunder rate, computados durante a
extracção sem custo de velocidade mensurável.

### 8.2 A precisão é metadata, não feature de clustering

**Contexto.** Podíamos acrescentar o ACPL ao vector de 18 features do clustering.

**Decisão.** Mantê-lo como metadata (colunas `acpl`, `blunder_rate`,
`n_analyzed_games` ao lado de `n_games` e `avg_rating`), não em `FEATURE_COLUMNS`.

**Porquê.** O recomendador live vai buscar os jogos do utilizador à API REST, que
não devolve evals — por isso um utilizador live *não tem* dados de precisão. Se o
ACPL fosse uma feature de clustering, seria nulo na inferência e preenchido com a
média populacional, sem contribuir nada. Melhor manter o clustering em sinais
disponíveis para todos, e usar a precisão como camada de enriquecimento.

### 8.3 Achado: a precisão carrega sinal de estilo para além do rating

O payoff. O ACPL correlaciona com o rating em −0.55 (a precisão *é* uma grande
parte da força), mas as diferenças de cluster persistem **dentro de uma banda de
rating fixa**, provando que é o estilo — não a força — que as conduz.

Todos os jogadores em ~1428 de rating (banda 1350-1500):

| Cluster                       | ACPL | Blunder rate |
| ---                           | ---  | ---          |
| C1 Underrated 1.e4 overperf.  | 70.4 | 8.5%         |
| C4 1.d4 specialist            | 73.4 | 8.9%         |
| C0 Quick 1.e4 amateur         | 74.8 | 9.3%         |
| C2 1.e4 grinder               | 77.8 | 9.4%         |
| C3 Queenside king-hunter      | 86.2 | 10.5%        |

Duas validações independentes das identidades de cluster:

- O **Queenside king-hunter** é ~16 cp/lance menos preciso que o cluster
  Underrated *ao mesmo rating* — troca genuinamente precisão por hipóteses de
  ataque, exactamente o que um estilo agressivo e sacrificial deve parecer.
- O **Underrated overperformer** é o mais preciso da banda, consistente com o seu
  score residual +0.11: joga mesmo acima do seu nível de rating.

Esta é a evidência mais forte do projeto de que os clusters capturam playstyle
real e não apenas baldes de rating — a precisão nunca foi input do clustering, e
mesmo assim separa os clusters de uma forma que as identidades preveem.

---

## 9. Validação da hipótese do continuum (clustering alternativo)

### 9.1 A afirmação em teste

O silhouette do K-Means ficou em ~0.08 em todas as escalas de dados e conjuntos
de features. Isso é consistente com "o playstyle é um continuum, não clusters
discretos" — mas silhouette baixo por si só é evidência fraca (o K-Means podia
ser só um mau encaixe enquanto clusters reais existem). Por isso fizemos
stress-test com dois algoritmos independentes
(`scripts/clustering_comparison.py`).

### 9.2 Controlar a maldição da dimensionalidade

A primeira tentativa correu GMM e HDBSCAN nas 18 features standardizadas em bruto.
Os resultados eram enganadores: o GMM dava max-responsibility ≈ 1.0 (parece
clusters nítidos) e o HDBSCAN rotulava ~100% dos pontos como ruído. Ambos são
**artefactos de alta dimensão** — em 18-dim, as densidades de Gaussianas de
covariância completa tornam-se extremas (atribuições sobreconfiantes) e as
distâncias Euclidianas concentram-se (métodos baseados em densidade partem).

Correcção: correr GMM e HDBSCAN numa **projeção PCA-6** (65% da variância), onde
as probabilidades e a densidade são significativas. Agora um resultado de
"tudo-ruído" ou "atribuição-dura" reflecte a geometria dos dados, não a métrica a
partir-se.

### 9.3 Resultado: três algoritmos convergem no continuum

Na projeção PCA-6 (sub-sample de 30k jogadores):

| Método   | Sinal                                               | Leitura                          |
| ---      | ---                                                 | ---                              |
| K-Means  | silhouette 0.08–0.10, plano em K=2..8               | sem fronteiras nítidas           |
| GMM      | BIC quase plano (591k → 583k, −1.4% de K=2 a 8); max responsibility cai 0.97 → 0.71; entropia sobe 0.14 → 0.37 | sem K natural; componentes sobrepõem-se |
| HDBSCAN  | 82% ruído em min_cluster_size 50–100; 100% ruído em ≥250 | sem grupos separados por densidade |

O BIC plano do GMM é a peça mais limpa: se houvesse K clusters naturais, o BIC
mostraria um joelho claro em K. Não mostra — acrescentar componentes quase não
melhora o encaixe, exactamente o que se esperaria ao ladrilhar uma nuvem suave com
mais Gaussianas.

![Comparação de clustering](docs/figures/08_clustering_comparison.png)

### 9.4 O que isto significa para o projeto

Os clusters são **labels úteis impostos num continuum**, não grupos naturais
descobertos. É uma situação legítima e comum — os clusters K-Means continuam
interpretáveis e conduzem recomendações sensatas (o achado de precisão da Phase 3
mostra que acompanham estilo real). Mas o enquadramento honesto é "particionamos
um espaço de estilo contínuo em 5 regiões de referência", não "os jogadores de
xadrez dividem-se em 5 tipos".

Um modelo genuinamente continuum-native dispensaria o hard clustering por
completo: recomendar aberturas a partir dos *k vizinhos mais próximos* no espaço
de estilo, ou aprender um embedding contínuo de estilo. Essa é a próxima
arquitectura natural se o projeto for levado mais longe.

---

## 10. Reprodutibilidade num sample disjunto

### 10.1 O teste

Tudo até aqui foi treinado nos primeiros 5M jogos mantidos do dump de Abril-2026.
Para verificar que os achados não são artefacto dessa fatia em particular, o
`dump_extract --skip 8_500_000` extraiu um segundo sample de 5M jogos
**disjunto** (jogos 8.5M–16.5M do stream). O `scripts/compare_samples.py` depois
correu a análise completa em ambos e fez matching dos clusters entre runs pelo
centroide mais próximo (atribuição Hungarian — as labels do K-Means são
arbitrárias).

Nota: os dois samples são disjuntos em *jogos* mas sobrepõem-se ~59% em
*jogadores* (um jogador activo ao longo do mês aparece em ambos, representado por
jogos diferentes). Esse é o setup certo para testar se o *estilo* de um jogador é
estável: jogos diferentes do mesmo jogador, mais uma mistura de população
diferente, reproduzem a mesma estrutura de clusters?

### 10.2 Resultado: a estrutura replica-se quase exactamente

| Cluster (matched)        | Rating v1/v2 | score_res v1/v2 | feature-chave v1/v2 | ACPL v1/v2 |
| ---                      | ---          | ---             | ---               | ---        |
| Quick 1.e4 amateur       | 1431 / 1441  | −0.05 / −0.06   | 84% / 83% e4      | 78.6 / 78.4|
| Underrated overperformer | 1540 / 1528  | +0.11 / +0.11   | 83% / 83% e4      | 70.6 / 71.3|
| 1.e4 grinder             | 1789 / 1793  | −0.02 / −0.02   | 86% / 86% e4      | 68.2 / 68.2|
| Queenside king-hunter    | 1415 / 1414  | −0.00 / −0.00   | 41% / 42% O-O-O   | 91.2 / 91.9|
| 1.d4 specialist          | 1740 / 1753  | +0.01 / +0.01   | 48% / 48% d4      | 66.6 / 66.0|

Os cinco clusters reaparecem com perfis quase idênticos — ratings dentro de ~12
pontos, percentagens de abertura dentro de 1–2 pontos, score residual idêntico a
duas casas decimais, ACPL dentro de ~1 cp. As curvas de silhouette sobrepõem-se;
as curvas de BIC do GMM têm a mesma forma plana (escala absoluta diferente, mesmo
padrão "sem K natural").

O achado de precisão da Phase 3 também replica: o Queenside king-hunter é o
cluster menos preciso (~91 cp) e o 1.d4 specialist o mais preciso (~66 cp) em
*ambos* os samples, ordem preservada.

### 10.3 O que isto compra

Dois samples disjuntos de 5M a produzir os mesmos cinco clusters, o mesmo sinal de
continuum e a mesma ordem de precisão é evidência forte de que a estrutura é uma
propriedade real da população do Lichess, não ruído de uma fatia. Combinado com a
§ 9 (é um continuum) o resumo honesto é: **o espaço de estilo é suave e estável —
o K-Means esculpe-o em cinco regiões de referência reprodutíveis.**

---

## Lista de leitura / referências

- Base de dados aberta do Lichess: <https://database.lichess.org/>
- TSVs chess-openings do Lichess: <https://github.com/lichess-org/chess-openings>
- Docs do python-chess: <https://python-chess.readthedocs.io/>
- Guia do utilizador de polars: <https://docs.pola.rs/>
- "Como escolher K em K-Means" — ver o output de
  `chess_coach/cluster.py --evaluate` e o racional em
  [4.2](#42-k4--k5-à-medida-que-features-e-dados-escalaram).
