from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_confidence_histogram(uncertainty_csv: str | Path, output_png: str | Path) -> None:
    df = pd.read_csv(uncertainty_csv)
    plt.figure(figsize=(9, 5))
    plt.hist(df["confidence"].to_numpy(), bins=40)
    plt.xlabel("Cluster confidence")
    plt.ylabel("Number of samples")
    plt.title("V2 K=4 Cluster Confidence")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close()


def save_anomaly_distribution(anomaly_csv: str | Path, output_png: str | Path) -> None:
    df = pd.read_csv(anomaly_csv)
    plt.figure(figsize=(9, 5))
    plt.hist(df["combined_anomaly_score"].to_numpy(), bins=50)
    plt.xlabel("Combined anomaly score percentile")
    plt.ylabel("Number of samples")
    plt.title("V2 Multi-Signal Anomaly Distribution")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close()


def save_consensus_agreement(consensus_csv: str | Path, output_png: str | Path) -> None:
    df = pd.read_csv(consensus_csv)
    plt.figure(figsize=(9, 5))
    plt.hist(df["consensus_agreement"].to_numpy(), bins=np.linspace(0.25, 1.0, 16))
    plt.xlabel("Fraction of models agreeing with consensus")
    plt.ylabel("Number of samples")
    plt.title("V2 Model Consensus Agreement")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close()


def save_top_anomalies(anomaly_csv: str | Path, output_png: str | Path, top_n: int = 2500) -> None:
    df = pd.read_csv(anomaly_csv).head(top_n)
    if not {"row", "col"}.issubset(df.columns):
        return
    plt.figure(figsize=(10, 8))
    plt.scatter(df["col"], df["row"], s=4, alpha=0.45, c=df["combined_anomaly_score"], cmap="viridis")
    plt.xlabel("Radar grid column")
    plt.ylabel("Radar grid row")
    plt.title(f"Top {len(df):,} V2 Candidate Anomalies")
    plt.gca().invert_yaxis()
    plt.colorbar(label="Combined anomaly score")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close()
