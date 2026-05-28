"""FastAPI backend for the chess-coach recommender.

Routes:
    GET  /                   -> static/index.html
    GET  /recommend/{user}   -> JSON: cluster + top openings + feature comparison
    GET  /health             -> liveness probe

Heavy artifacts (kmeans model, scaler, recommendations.json) are loaded once
at startup via a lifespan handler and reused across requests.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from chess_coach.collector import fetch_games, game_to_row
from chess_coach.features import FEATURE_COLUMNS, build_player_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"


class AppState:
    """Loaded once at startup, read-only at request time."""
    kmeans: Any = None
    scaler: Any = None
    recommendations: dict | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.kmeans = joblib.load(DATA_DIR / "models" / "kmeans.joblib")
    state.scaler = joblib.load(DATA_DIR / "models" / "scaler.joblib")
    state.recommendations = json.loads(
        (DATA_DIR / "recommendations.json").read_text()
    )
    yield


app = FastAPI(
    title="Chess Coach",
    description="Opening recommender based on Lichess playstyle clustering.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "k": state.recommendations["k"] if state.recommendations else None,
    }


@app.get("/recommend/{username}")
def recommend(username: str, max_games: int = 100) -> JSONResponse:
    # Bound the cost an anonymous caller can impose on Lichess.
    max_games = max(20, min(max_games, 200))

    try:
        raw_games = fetch_games(username, max_games)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(
            status_code=code,
            detail=f"Lichess API error fetching games for '{username}'",
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc

    rows = [r for r in (game_to_row(g, username) for g in raw_games) if r is not None]
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No rated games found for '{username}' on Lichess.",
        )

    games_df = pl.DataFrame(rows)
    features = build_player_features(games_df, min_games=1)
    user_row = features.row(0, named=True)

    X = features.select(FEATURE_COLUMNS).to_numpy()
    X_scaled = state.scaler.transform(X)
    cluster_id = int(state.kmeans.predict(X_scaled)[0])

    # Look up the matching cluster's recommendations
    cluster_meta = next(
        c for c in state.recommendations["clusters"] if c["id"] == cluster_id
    )

    # Feature comparison (user vs cluster mean) — useful for explainability
    cluster_means = cluster_meta["feature_means"]
    feature_comparison = [
        {
            "feature": col,
            "user": float(user_row[col]),
            "cluster_mean": float(cluster_means[col]),
            "delta": float(user_row[col]) - float(cluster_means[col]),
        }
        for col in FEATURE_COLUMNS
    ]

    return JSONResponse({
        "username": username,
        "n_games_used": games_df.height,
        "user_rating": float(user_row["avg_rating"]),
        "cluster": {
            "id": cluster_meta["id"],
            "name": cluster_meta["name"],
            "blurb": cluster_meta["blurb"],
            "size": cluster_meta["size"],
            "avg_rating": cluster_meta["avg_rating"],
            "accuracy": cluster_meta.get("accuracy"),
        },
        "top_openings": cluster_meta["top_openings"],
        "feature_comparison": feature_comparison,
    })


# Mount static last so the API routes above take precedence.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
