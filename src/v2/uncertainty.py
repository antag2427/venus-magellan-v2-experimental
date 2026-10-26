from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES
from .report import ensure_new, ensure_outputs_new, require_inputs, write_csv
from .model_artifacts import validate_features


def fit_distance_uncertainty(
    sample_csv: str | Path,
    output_csv: str | Path,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 20,
    temperature: float = 1.0,
    *, max_iter: int = 500,
) -> dict[str, float | int | str]:
    ensure_outputs_new(output_csv)
    require_inputs(sample_csv)
    df = pd.read_csv(sample_csv, float_precision="round_trip")
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = df[features].to_numpy(dtype=np.float64)
    X = validate_features(X, features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(Xs)
    distances = model.transform(Xs)
    order = np.argsort(distances, axis=1)
    nearest = distances[np.arange(len(df)), order[:, 0]]
    second = distances[np.arange(len(df)), order[:, 1]]
    margin = second - nearest
    logits = -distances / max(float(temperature), 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(k)
    confidence = 1.0 - entropy

    out = df[[c for c in ("row", "col") if c in df.columns]].copy()
    out["cluster"] = labels
    out["nearest_centroid_distance"] = nearest
    out["second_centroid_distance"] = second
    out["distance_margin"] = margin
    out["confidence"] = confidence
    out["entropy"] = entropy
    for c in range(k):
        out[f"p_cluster_{c}"] = probs[:, c]

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, output, index=False)

    return {
        "rows": int(len(out)),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "mean_entropy": float(entropy.mean()),
        "ambiguous_fraction_entropy_gt_0.5": float((entropy > 0.5).mean()),
        "output": str(output),
        "estimator_parameters": model.get_params(), "temperature": temperature,
    }
