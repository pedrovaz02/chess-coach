"""Cluster players by their playstyle feature vector using K-Means.

Two modes:
    --evaluate          Sweep K in a range, print inertia + silhouette scores.
                        Use this to pick K before committing to a model.
    (default)           Fit K-Means with the chosen K, save model + scaler,
                        and write features.parquet enriched with a `cluster`
                        column to data/players_clustered.parquet.

Why standardize first:
    K-Means uses Euclidean distance. avg_moves spans ~60..90 while win_rate
    spans 0..1 — without scaling, avg_moves dominates every distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from chess_coach.features import FEATURE_COLUMNS


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = DATA_DIR / "models"
RANDOM_STATE = 42


def fit_clusters(
    features: pl.DataFrame, k: int
) -> tuple[pl.DataFrame, KMeans, StandardScaler]:
    """Fit K-Means and return (df+cluster, model, scaler).

    The scaler is needed at inference time to project new players into the
    same feature space the model was trained on.
    """
    X = features.select(FEATURE_COLUMNS).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)

    enriched = features.with_columns(pl.Series("cluster", labels))
    return enriched, model, scaler


def evaluate_k_range(features: pl.DataFrame, k_min: int, k_max: int) -> pl.DataFrame:
    """For each K, return inertia (lower=tighter) and silhouette (-1..1, higher=better)."""
    X = features.select(FEATURE_COLUMNS).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else float("nan")
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pl.DataFrame(rows)


def summarise_clusters(enriched: pl.DataFrame) -> pl.DataFrame:
    """One row per cluster with mean of each feature + size."""
    return (
        enriched.group_by("cluster")
        .agg(
            size=pl.len(),
            **{col: pl.col(col).mean() for col in FEATURE_COLUMNS},
        )
        .sort("cluster")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster players by playstyle")
    parser.add_argument(
        "--features", type=Path, default=DATA_DIR / "features.parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "players_clustered.parquet"
    )
    parser.add_argument("--k", type=int, default=4, help="Number of clusters")
    parser.add_argument(
        "--evaluate", action="store_true", help="Sweep K and print scores instead of fitting"
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    console = Console()
    features = pl.read_parquet(args.features)
    console.print(f"Loaded [bold]{features.height}[/bold] players from {args.features}")

    if args.evaluate:
        scores = evaluate_k_range(features, args.k_min, args.k_max)
        table = Table(title="K sweep")
        table.add_column("K", justify="right")
        table.add_column("Inertia", justify="right")
        table.add_column("Silhouette", justify="right")
        for row in scores.iter_rows(named=True):
            table.add_row(
                str(row["k"]),
                f"{row['inertia']:.2f}",
                f"{row['silhouette']:.3f}",
            )
        console.print(table)
        console.print(
            "\n[dim]Pick K at the elbow of inertia (where the drop slows down) "
            "or where silhouette peaks.[/dim]"
        )
        return

    enriched, model, scaler = fit_clusters(features, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(args.output)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    console.print(f"\nFit K-Means with K=[bold]{args.k}[/bold]")
    console.print(f"Saved labelled players to [bold]{args.output}[/bold]")
    console.print(f"Saved model + scaler to [bold]{MODELS_DIR}[/bold]")

    summary = summarise_clusters(enriched)
    console.print("\n[bold]Cluster summary (mean of each feature):[/bold]")
    console.print(summary)


if __name__ == "__main__":
    main()
