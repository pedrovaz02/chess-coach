# Hugging Face Spaces Docker container for chess-coach.
# HF requires the container to run as a non-root user with UID 1000.

FROM python:3.14-slim

# Install uv (modern Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HF Spaces requirement: non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy

WORKDIR /home/user/app

# Copy dependency manifests first for Docker layer cache
COPY --chown=user pyproject.toml uv.lock README.md ./

# Resolve and install dependencies into /home/user/app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code, frontend, and inference artefacts
COPY --chown=user src/ ./src/
COPY --chown=user static/ ./static/
COPY --chown=user data/recommendations.json ./data/recommendations.json
COPY --chown=user data/models/ ./data/models/

# Install the project itself (registers entry points)
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD [".venv/bin/uvicorn", "chess_coach.api:app", "--host", "0.0.0.0", "--port", "8000"]
