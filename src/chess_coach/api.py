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
from chess_coach.features import build_player_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"


class AppState:
    """Loaded once at startup, read-only at request time."""
    kmeans: Any = None
    scaler: Any = None
    recommendations: dict | None = None
    knn: Any = None  # ServingIndex for the continuum recommender


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from chess_coach.knn_recommender import load_serving_index

    state.kmeans = joblib.load(DATA_DIR / "models" / "kmeans.joblib")
    state.scaler = joblib.load(DATA_DIR / "models" / "scaler.joblib")
    state.recommendations = json.loads(
        (DATA_DIR / "recommendations.json").read_text()
    )
    # kNN serving artifacts are optional — the cluster recommender works
    # without them, so a missing index just disables continuum mode.
    try:
        state.knn = load_serving_index()
    except FileNotFoundError:
        state.knn = None
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
def recommend(username: str, max_games: int = 100, mode: str = "cluster") -> JSONResponse:
    """Opening recommendations for a Lichess user.

    mode="cluster" (default): snap to the nearest K-Means cluster, return that
    cluster's precomputed openings.
    mode="knn": pool the openings of the k nearest players in style space —
    the continuum-native recommender (no hard cluster boundary).
    """
    if mode not in ("cluster", "knn"):
        raise HTTPException(status_code=400, detail="mode must be 'cluster' or 'knn'")
    if mode == "knn" and state.knn is None:
        raise HTTPException(status_code=503, detail="kNN index not available")

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

    # Production feature order from recommendations.json — the set the saved
    # scaler/model were trained on. Decoupled from the working-tree
    # FEATURE_COLUMNS, which may carry experimental extra features.
    prod_cols = state.recommendations["feature_columns"]
    X = features.select(prod_cols).to_numpy()
    X_scaled = state.scaler.transform(X)

    if mode == "knn":
        return JSONResponse(_knn_response(username, games_df, user_row, X_scaled))

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
        for col in prod_cols
    ]

    return JSONResponse({
        "username": username,
        "mode": "cluster",
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


def _knn_response(username: str, games_df, user_row, X_scaled) -> dict:
    """Build the /recommend response for mode=knn.

    Reuses the same response shape as the cluster path (the frontend renders
    one card), where the 'cluster' block describes the style neighbourhood
    instead of a discrete cluster.
    """
    import numpy as np

    from chess_coach.knn_recommender import recommend_served

    idx = state.knn
    # Leave-one-out if the query player is in the training index.
    hit = np.where(idx.usernames == username)[0]
    exclude = int(hit[0]) if hit.size else None
    query = X_scaled.astype(np.float32)

    out = recommend_served(idx, query, exclude_idx=exclude)
    blurb = (
        f"Openings pooled from the {out['k']:,} players most similar to you in "
        f"style space (ratings {out['neighbour_rating_min']:.0f}–"
        f"{out['neighbour_rating_max']:.0f}, mean {out['neighbour_rating_mean']:.0f}). "
        f"No hard cluster boundary — a smooth neighbourhood on the playstyle continuum."
    )
    return {
        "username": username,
        "mode": "knn",
        "n_games_used": games_df.height,
        "user_rating": float(user_row["avg_rating"]),
        "cluster": {
            "id": None,
            "name": "Your style neighbourhood",
            "blurb": blurb,
            "size": out["k"],
            "avg_rating": out["neighbour_rating_mean"],
            "accuracy": None,
        },
        "top_openings": {"white": out["white"], "black": out["black"]},
        "feature_comparison": [],
    }


@app.get("/versus/{player_a}/{player_b}")
def versus(player_a: str, player_b: str) -> JSONResponse:
    """Compare two players' styles and surface openings that suit both."""
    from chess_coach import versus as versus_mod

    def profile(name: str) -> dict:
        try:
            return versus_mod._profile(name, state.scaler, state.kmeans, state.recommendations)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 502
            raise HTTPException(status_code=code, detail=f"Lichess error for '{name}'") from exc
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc

    a = profile(player_a)
    b = profile(player_b)
    return JSONResponse(versus_mod.build_versus_payload(a, b))


# Mount static last so the API routes above take precedence.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
