"""Generate a comprehensive terminal-commands reference PDF for chess-coach."""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak,
    Table, TableStyle, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT


# ── Styling ──────────────────────────────────────────────────────────────
ACCENT = HexColor("#6a7d4f")
CODE_BG = HexColor("#f5f3ee")
CODE_BORDER = HexColor("#d8d3c4")
MUTED = HexColor("#6e6a60")
TEXT = HexColor("#2a2825")

styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=22, leading=28, textColor=TEXT,
    spaceBefore=0, spaceAfter=6, fontName="Helvetica-Bold",
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=15, leading=20, textColor=ACCENT,
    spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold",
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"],
    fontSize=11.5, leading=15, textColor=TEXT,
    spaceBefore=12, spaceAfter=3, fontName="Helvetica-Bold",
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontSize=10, leading=14, textColor=TEXT,
    spaceBefore=2, spaceAfter=4, fontName="Helvetica",
)
NOTE = ParagraphStyle(
    "Note", parent=styles["BodyText"],
    fontSize=9, leading=12, textColor=MUTED,
    spaceBefore=2, spaceAfter=8, fontName="Helvetica-Oblique",
)
CODE = ParagraphStyle(
    "Code", parent=styles["Code"],
    fontSize=9, leading=12, textColor=TEXT,
    leftIndent=10, rightIndent=10,
    spaceBefore=4, spaceAfter=8,
    fontName="Courier", backColor=CODE_BG,
    borderColor=CODE_BORDER, borderWidth=0.5, borderPadding=6,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["BodyText"],
    fontSize=11, leading=15, textColor=MUTED,
    alignment=TA_LEFT, spaceBefore=0, spaceAfter=18,
    fontName="Helvetica",
)
TOC_ITEM = ParagraphStyle(
    "TocItem", parent=styles["BodyText"],
    fontSize=10.5, leading=16, textColor=TEXT,
    spaceBefore=0, spaceAfter=0, fontName="Helvetica",
)


def section(title: str) -> Paragraph:
    return Paragraph(title, H2)


def subsection(title: str) -> Paragraph:
    return Paragraph(title, H3)


def body(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def note(text: str) -> Paragraph:
    return Paragraph(text, NOTE)


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


story = []

# ── Cover ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 4 * cm))
story.append(Paragraph("Chess Coach", H1))
story.append(Paragraph(
    "Terminal commands reference — every command used or useful "
    "during the development, training, deployment and operation of "
    "the chess-coach project.",
    SUBTITLE,
))
story.append(Spacer(1, 1 * cm))

story.append(Paragraph("Contents", H2))
toc = [
    "1. Setup &amp; environment",
    "2. uv — package &amp; project management",
    "3. Git — version control",
    "4. Git LFS &amp; binary files",
    "5. Hugging Face Spaces deploy",
    "6. Data pipeline (training)",
    "7. Inference (CLI &amp; web)",
    "8. Background processes &amp; monitoring",
    "9. Log inspection",
    "10. Process management",
    "11. Data inspection (Polars one-liners)",
    "12. Network &amp; API testing",
    "13. Disk &amp; performance",
    "14. Useful one-off recipes",
]
for item in toc:
    story.append(Paragraph(item, TOC_ITEM))

story.append(PageBreak())


# ── 1. Setup & environment ───────────────────────────────────────────────
story.append(section("1. Setup &amp; environment"))

story.append(subsection("Install uv (Python package manager)"))
story.append(code("""
# Option A — via Homebrew (recommended on macOS)
brew install uv

# Option B — universal install script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
"""))

story.append(subsection("Install other system tools"))
story.append(code("""
# Git LFS (needed for HF Spaces with binary files)
brew install git-lfs

# Docker (optional — only if you want to test container locally)
brew install --cask docker

# watch (optional — for live file monitoring)
brew install watch

# zstd (optional — if you want to inspect .pgn.zst manually)
brew install zstd
"""))

story.append(subsection("Project skeleton (one-time)"))
story.append(code("""
mkdir -p ~/dev && cd ~/dev
uv init --package chess-coach
cd chess-coach
"""))


# ── 2. uv ────────────────────────────────────────────────────────────────
story.append(section("2. uv — package &amp; project management"))

story.append(subsection("Daily commands"))
story.append(code("""
# Install all locked dependencies into .venv
uv sync

# Install only runtime deps (no --dev), for production
uv sync --frozen --no-dev

# Add a runtime dependency
uv add polars

# Add a dev-only dependency (notebooks, tests, etc.)
uv add --dev pytest reportlab

# Remove a dependency
uv remove pandas

# Run a script inside the project venv
uv run python -m chess_coach.recommender pedrovaz02

# Run any command in the venv
uv run uvicorn chess_coach.api:app --port 8000

# Run with unbuffered output (for log-friendly long-running scripts)
uv run python -u -m chess_coach.dump_extract --input ...

# Update lockfile to latest compatible versions
uv lock --upgrade
"""))


# ── 3. Git ───────────────────────────────────────────────────────────────
story.append(section("3. Git — version control"))

story.append(subsection("Daily workflow"))
story.append(code("""
git status
git diff
git log --oneline -10

git add path/to/file.py
git add -A                              # stage everything

git commit -m "Short imperative summary"

git push                                # default remote (origin)
git push origin main                    # explicit
git push --force                        # ONLY when history was rewritten
"""))

story.append(subsection("Remotes"))
story.append(code("""
git remote -v
git remote add origin git@github.com:USER/REPO.git
git remote add huggingface https://huggingface.co/spaces/USER/REPO
git remote set-url huggingface https://huggingface.co/spaces/USER/REPO
git remote remove huggingface
"""))

story.append(subsection("Branches"))
story.append(code("""
git branch                              # list local branches
git branch -M main                      # rename current branch to main
git checkout -b feature/x               # create + switch
git checkout main                       # switch
git merge feature/x                     # merge into current branch
"""))

story.append(subsection("Inspect history"))
story.append(code("""
git log --oneline -20
git log --stat path/to/file.py
git log --since="2 days ago"
git show <commit_sha>
git blame src/chess_coach/cluster.py
"""))

story.append(subsection("Undoing things (use with care)"))
story.append(code("""
git restore path/to/file.py             # discard unstaged changes
git reset HEAD path/to/file.py          # unstage but keep edits
git reset --soft HEAD~1                 # undo last commit, keep changes staged
git reset --hard HEAD~1                 # undo last commit AND discard changes
"""))


# ── 4. Git LFS ───────────────────────────────────────────────────────────
story.append(section("4. Git LFS &amp; binary files"))

story.append(body(
    "HF Spaces rejects regular git pushes that contain binary files "
    "(models, images). Use Git LFS to track them."
))

story.append(subsection("Initial setup (once per repo)"))
story.append(code("""
git lfs install

# Tell LFS which patterns to track
git lfs track "*.joblib"
git lfs track "*.png"
git lfs track "*.parquet"

# .gitattributes was created — commit it
git add .gitattributes
git commit -m "Track binary patterns via Git LFS"
"""))

story.append(subsection("Migrate existing history (when files were committed as binary first)"))
story.append(code("""
# Rewrite all past commits to put matching files in LFS
git lfs migrate import --include="*.joblib,*.png" --everything

# History is now rewritten — force-push to both remotes
git push --force origin main
git push --force huggingface main
"""))

story.append(subsection("Inspect LFS state"))
story.append(code("""
git lfs ls-files                        # list tracked files
git lfs status                          # show LFS pending uploads
git lfs version
"""))


# ── 5. HF Spaces deploy ──────────────────────────────────────────────────
story.append(section("5. Hugging Face Spaces deploy"))

story.append(subsection("One-time auth setup"))
story.append(code("""
# Generate a write token at huggingface.co/settings/tokens
# Then save it once into macOS Keychain so future pushes are silent:

git config --global credential.helper osxkeychain
printf "protocol=https\\nhost=huggingface.co\\nusername=YOUR_HF_USER\\npassword=hf_YOUR_TOKEN\\n\\n" \\
  | git credential approve
"""))

story.append(subsection("Push to deploy"))
story.append(code("""
git push huggingface main
# HF auto-detects Dockerfile and starts a build
"""))

story.append(subsection("Check build &amp; runtime logs via API"))
story.append(code("""
HF_TOKEN=hf_YOUR_TOKEN

# Build logs (Docker layer build)
curl -sN -H "Authorization: Bearer $HF_TOKEN" \\
  "https://huggingface.co/api/spaces/USER/SPACE/logs/build"

# Run logs (your app's stdout/stderr)
curl -sN -H "Authorization: Bearer $HF_TOKEN" \\
  "https://huggingface.co/api/spaces/USER/SPACE/logs/run"

# Space status JSON (stage, hardware, last-modified)
curl -s -H "Authorization: Bearer $HF_TOKEN" \\
  "https://huggingface.co/api/spaces/USER/SPACE" | python3 -m json.tool
"""))

story.append(subsection("Test the deployed endpoints"))
story.append(code("""
curl -s -o /dev/null -w "%{http_code} %{time_total}s\\n" \\
  https://USER-SPACE.hf.space/

curl -s https://USER-SPACE.hf.space/health

curl -s https://USER-SPACE.hf.space/recommend/pedrovaz02 | python3 -m json.tool
"""))


# ── 6. Data pipeline (training) ──────────────────────────────────────────
story.append(section("6. Data pipeline (training)"))

story.append(subsection("Download a monthly Lichess dump (~28 GB)"))
story.append(code("""
# Latest month available
uv run python -m chess_coach.dump_download --month 2026-04

# Custom output directory
uv run python -m chess_coach.dump_download \\
  --month 2026-04 --output-dir /path/to/dumps

# Force re-download (overwrites partial file)
uv run python -m chess_coach.dump_download --month 2026-04 --force
"""))

story.append(subsection("Extract games to parquet (parallel, ~16 min for 5M games)"))
story.append(code("""
uv run python -m chess_coach.dump_extract \\
  --input data/dumps/lichess_db_standard_rated_2026-04.pgn.zst \\
  --output data/games.parquet \\
  --max-games 5000000 \\
  --workers 14

# Tune filters
uv run python -m chess_coach.dump_extract \\
  --input ... --output ... --max-games 1000000 \\
  --min-elo 1200 --max-elo 2400 \\
  --time-controls blitz rapid
"""))

story.append(subsection("Alternative — API-based collector (smaller, slower)"))
story.append(code("""
uv run python -m chess_coach.collector \\
  --perf-types ultraBullet bullet blitz rapid classical \\
  --top-n 100 --games 100 --low-rated 300 --sleep 1.0
"""))

story.append(subsection("Build per-player features"))
story.append(code("""
uv run python -m chess_coach.features
# Defaults: reads data/games.parquet → data/features.parquet
# Drops players with < 20 games

# Custom path / threshold
uv run python -m chess_coach.features \\
  --games data/games.parquet \\
  --output data/features.parquet \\
  --min-games 30
"""))

story.append(subsection("Pick K, then train K-Means"))
story.append(code("""
# Inertia + silhouette sweep
uv run python -m chess_coach.cluster --evaluate --k-min 2 --k-max 10

# Fit the chosen K (saves kmeans.joblib + scaler.joblib + players_clustered.parquet)
uv run python -m chess_coach.cluster --k 5
"""))

story.append(subsection("Precompute recommendations JSON"))
story.append(code("""
uv run python -m chess_coach.precompute --top-n 5 --min-opening-games 5000
"""))

story.append(subsection("(Re)build the openings classifier database"))
story.append(code("""
# Downloads Lichess's chess-openings TSVs and produces openings.json
uv run python scripts/build_openings_db.py
"""))


# ── 7. Inference ─────────────────────────────────────────────────────────
story.append(section("7. Inference (CLI &amp; web)"))

story.append(subsection("Single-user CLI recommendation"))
story.append(code("""
uv run python -m chess_coach.recommender pedrovaz02
uv run python -m chess_coach.recommender pedrovaz02 --top-n 8 --min-opening-games 3000
uv run python -m chess_coach.recommender pedrovaz02 --verbose
"""))

story.append(subsection("Sanity check — fetch raw games from Lichess"))
story.append(code("""
uv run python -m chess_coach.hello_lichess pedrovaz02
"""))

story.append(subsection("Run the local web app"))
story.append(code("""
uv run uvicorn chess_coach.api:app --port 8000
# Open http://localhost:8000

# Auto-reload on code change (dev)
uv run uvicorn chess_coach.api:app --port 8000 --reload

# Bind to all interfaces (LAN access)
uv run uvicorn chess_coach.api:app --host 0.0.0.0 --port 8000
"""))


# ── 8. Background processes ──────────────────────────────────────────────
story.append(section("8. Background processes &amp; monitoring"))

story.append(subsection("Launch detached (survives terminal close)"))
story.append(code("""
nohup uv run python -u -m chess_coach.dump_extract \\
  --input data/dumps/lichess_db_standard_rated_2026-04.pgn.zst \\
  --output data/games.parquet \\
  --max-games 5000000 \\
  > /tmp/extract.log 2>&1 &

echo "PID: $!"
"""))

story.append(subsection("Run job in background (dies when terminal closes)"))
story.append(code("""
uv run python -m chess_coach.collector --top-n 50 --games 80 > /tmp/collect.log 2>&1 &
"""))


# ── 9. Log inspection ────────────────────────────────────────────────────
story.append(section("9. Log inspection"))

story.append(subsection("Tail logs"))
story.append(code("""
tail -f /tmp/extract.log                # live stream
tail -F /tmp/extract.log                # like -f but follows recreation
tail -50 /tmp/collect.log               # last 50 lines
cat /tmp/extract.log                    # full content (small files only)
less /tmp/extract.log                   # paginate (arrows, q to quit)
"""))

story.append(subsection("Grep / filter"))
story.append(code("""
grep ERROR /tmp/extract.log
grep -i error /tmp/*.log                # case-insensitive across files
grep "rate " /tmp/extract.log | tail -5
"""))

story.append(subsection("Count / inspect"))
story.append(code("""
wc -l /tmp/extract.log
ls -lh /tmp/*.log
"""))


# ── 10. Process management ───────────────────────────────────────────────
story.append(section("10. Process management"))

story.append(code("""
# Find your processes
ps aux | grep chess_coach | grep -v grep
pgrep -f chess_coach

# Process details
ps -p PID -o pid,%cpu,%mem,etime,command

# Find child processes
pgrep -P PARENT_PID

# Network sockets owned by a process
lsof -p PID | grep TCP

# Kill politely (SIGTERM)
kill PID

# Force-kill (SIGKILL)
kill -9 PID

# Kill by name
pkill -f chess_coach.dump_extract
"""))


# ── 11. Polars one-liners ────────────────────────────────────────────────
story.append(section("11. Data inspection (Polars one-liners)"))

story.append(subsection("Quick shape + sample"))
story.append(code("""
uv run python -c "
import polars as pl
df = pl.read_parquet('data/games.parquet')
print('Shape:', df.shape)
print('Columns:', df.columns)
print(df.head(3))
"
"""))

story.append(subsection("Aggregations"))
story.append(code("""
uv run python -c "
import polars as pl
df = pl.read_parquet('data/games.parquet')
print(df.group_by('result').len())
print(df.group_by('status').len().sort('len', descending=True))
print(df.group_by('speed').len())
"
"""))

story.append(subsection("Per-player game counts"))
story.append(code("""
uv run python -c "
import polars as pl
df = pl.read_parquet('data/games.parquet')
counts = df.group_by('username').len().sort('len', descending=True)
print(counts.describe())
for n in [5, 10, 20, 50, 100]:
    print(f'>= {n} games:', (counts['len'] >= n).sum())
"
"""))

story.append(subsection("Feature distribution"))
story.append(code("""
uv run python -c "
import polars as pl
f = pl.read_parquet('data/features.parquet')
print(f.select(['score_residual','draw_rate','avg_moves']).describe())
"
"""))


# ── 12. Network & API ────────────────────────────────────────────────────
story.append(section("12. Network &amp; API testing"))

story.append(subsection("Test Lichess API directly"))
story.append(code("""
# Status check on a single user
curl -s -o /dev/null -w "HTTP %{http_code} | time %{time_total}s\\n" \\
  "https://lichess.org/api/games/user/pedrovaz02?max=5&rated=true" \\
  -H "Accept: application/x-ndjson"

# Fetch a few games and pipe through jq
curl -s "https://lichess.org/api/games/user/pedrovaz02?max=3&opening=true" \\
  -H "Accept: application/x-ndjson" | head -3
"""))

story.append(subsection("Test our local API"))
story.append(code("""
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/recommend/pedrovaz02 | python3 -m json.tool

curl -s -o /dev/null -w "%{http_code} %{time_total}s\\n" \\
  http://127.0.0.1:8000/recommend/SOMEONE
"""))

story.append(subsection("Check what's listening locally"))
story.append(code("""
lsof -i :8000
lsof -nP -iTCP -sTCP:LISTEN | grep python
"""))


# ── 13. Disk & performance ───────────────────────────────────────────────
story.append(section("13. Disk &amp; performance"))

story.append(code("""
# Disk free
df -h ~/dev
df -h /

# Project size
du -sh ~/dev/chess-coach
du -sh ~/dev/chess-coach/data
du -sh ~/dev/chess-coach/data/dumps

# Sort directory contents by size
du -h --max-depth=1 ~/dev/chess-coach | sort -h

# How big is each parquet?
ls -lhS ~/dev/chess-coach/data/*.parquet
"""))


# ── 14. Recipes ──────────────────────────────────────────────────────────
story.append(section("14. Useful one-off recipes"))

story.append(subsection("Smoke-test the full pipeline (small data)"))
story.append(code("""
uv run python -m chess_coach.collector --perf-types blitz --top-n 3 --games 10 \\
  --low-rated 2 --low-rating-max 2200 --output /tmp/smoke.parquet --sleep 0.5

uv run python -m chess_coach.features --games /tmp/smoke.parquet \\
  --output /tmp/smoke_features.parquet --min-games 1

uv run python -m chess_coach.cluster --features /tmp/smoke_features.parquet \\
  --evaluate --k-min 2 --k-max 5
"""))

story.append(subsection("Tail the most recent Lichess dump available"))
story.append(code("""
for m in $(seq -f "%02g" 1 12); do
  url="https://database.lichess.org/standard/lichess_db_standard_rated_2026-${m}.pgn.zst"
  status=$(curl -s -o /dev/null -w "%{http_code}" -I "$url")
  echo "2026-$m: HTTP $status"
done
"""))

story.append(subsection("Free disk space — delete the dump after extract"))
story.append(code("""
ls -lh data/dumps/
rm data/dumps/lichess_db_standard_rated_2026-04.pgn.zst
"""))

story.append(subsection("Sync repo to both remotes (GitHub + HF) after history rewrite"))
story.append(code("""
git push --force origin main
git push --force huggingface main
"""))

story.append(subsection("Inspect HF Space artifacts shipped in the container"))
story.append(code("""
# Lists what the Dockerfile actually copies
grep -E '^COPY' Dockerfile

# Size accounting
ls -lh data/recommendations.json data/models/
"""))

story.append(subsection("Quickly regenerate recommendations after editing precompute logic"))
story.append(code("""
uv run python -m chess_coach.precompute && \\
  uv run python -m chess_coach.recommender pedrovaz02
"""))


# ── Build ────────────────────────────────────────────────────────────────
OUT = Path("/Users/pedrovaz/dev/chess-coach/docs/chess-coach-commands.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=1.7 * cm,
    rightMargin=1.7 * cm,
    topMargin=1.7 * cm,
    bottomMargin=1.7 * cm,
    title="Chess Coach — terminal commands",
    author="pedrovaz02",
)

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(
        A4[0] - 1.7 * cm, 1 * cm,
        f"chess-coach — terminal commands · p. {doc.page}",
    )
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
