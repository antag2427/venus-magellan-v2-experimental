from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES, validate_prefix
from .report import ensure_new, ensure_outputs_new, require_inputs, write_csv
from .model_artifacts import validate_features


def _contingency(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    table = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(reference, candidate):
        if 0 <= a < k and 0 <= b < k:
            table[a, b] += 1
    return table


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    table = _contingency(reference, candidate, k)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    aligned = np.array([mapping.get(int(x), int(x)) for x in candidate], dtype=np.int16)
    return aligned, mapping


def normalized_entropy(labels_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return per-sample H(labels) / log(k) for a (samples, models) matrix."""
    labels_matrix = np.asarray(labels_matrix)
    if labels_matrix.ndim != 2 or labels_matrix.shape[1] == 0:
        raise ValueError("labels_matrix must be 2D with at least one model.")
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("k must be an integer greater than or equal to 2.")
    if not np.isin(labels_matrix, np.arange(k)).all():
        raise ValueError("labels must be integer cluster IDs in [0, k).")

    n_samples, n_models = labels_matrix.shape
    counts = np.zeros((n_samples, k), dtype=np.float64)
    for model_idx in range(n_models):
        for c in range(k):
            counts[:, c] += (labels_matrix[:, model_idx] == c)
    probs = counts / n_models
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return ent / np.log(k)


def run_consensus(
    sample_csv: str | Path,
    output_dir: str | Path,
    k: int = 4,
    subset_size: int = 20_000,
    random_state: int = 42,
    n_init: int = 20,
    *, max_iter: int = 500, gmm_n_init: int = 3, gmm_max_iter: int = 300,
    birch_threshold: float = 0.5, minibatch_batch_size: int = 2048, output_prefix: str = "v2",
) -> dict[str, str | int | float]:
    output = Path(output_dir)
    validate_prefix(output_prefix)
    names = ["consensus_pairwise_ari.csv", "consensus_labels.csv", "advanced_scaler.csv",
             "consensus_summary.csv", "label_mappings.json"]
    ensure_outputs_new(*(output / f"{output_prefix}_{name}" for name in names))
    require_inputs(sample_csv)
    df = pd.read_csv(sample_csv, float_precision="round_trip")
    feature_columns = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = df[feature_columns].to_numpy(dtype=np.float64)
    X = validate_features(X, feature_columns)
    if min(subset_size, len(X)) < k:
        raise ValueError("Consensus subset must contain at least k samples.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rng = np.random.default_rng(random_state)
    subset_n = min(subset_size, len(df))
    subset_idx = np.sort(rng.choice(len(df), size=subset_n, replace=False))
    Xsub = Xs[subset_idx]

    models = {
        "kmeans": KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter),
        "minibatch_kmeans": MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            batch_size=minibatch_batch_size,
            max_iter=max_iter,
        ),
        "gmm": GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=gmm_n_init,
            max_iter=gmm_max_iter,
        ),
        "birch": Birch(n_clusters=k, threshold=birch_threshold),
    }

    output.mkdir(parents=True, exist_ok=True)
    raw_labels: dict[str, np.ndarray] = {}
    for name, model in models.items():
        raw_labels[name] = model.fit_predict(Xsub)

    reference_name = "kmeans"
    reference = raw_labels[reference_name]
    aligned = {reference_name: reference.copy()}
    mappings: dict[str, dict[int, int]] = {reference_name: {i: i for i in range(k)}}
    for name, labels in raw_labels.items():
        if name == reference_name:
            continue
        aligned[name], mapping = align_labels(reference, labels, k)
        mappings[name] = mapping

    label_matrix = np.column_stack([aligned[name] for name in models])
    consensus = np.apply_along_axis(lambda row: np.bincount(row, minlength=k).argmax(), 1, label_matrix)
    agreement = (label_matrix == consensus[:, None]).mean(axis=1)
    entropy = normalized_entropy(label_matrix, k)

    pairwise = []
    for a, b in combinations(models, 2):
        pairwise.append({"model_a": a, "model_b": b, "ARI": adjusted_rand_score(raw_labels[a], raw_labels[b])})
    pairwise_df = pd.DataFrame(pairwise)
    write_csv(pairwise_df, output / f"{output_prefix}_consensus_pairwise_ari.csv", index=False)

    labels_df = pd.DataFrame({"sample_index": subset_idx})
    for name in models:
        labels_df[f"label_{name}"] = raw_labels[name]
        labels_df[f"aligned_{name}"] = aligned[name]
    labels_df["consensus_label"] = consensus
    labels_df["consensus_agreement"] = agreement
    labels_df["consensus_entropy"] = entropy
    write_csv(labels_df, output / f"{output_prefix}_consensus_labels.csv", index=False)

    scaler_df = pd.DataFrame({"feature": feature_columns, "mean": scaler.mean_, "scale": scaler.scale_})
    write_csv(scaler_df, output / f"{output_prefix}_advanced_scaler.csv", index=False)

    metrics = {
        "sample_rows": len(df),
        "consensus_rows": subset_n,
        "models": list(models),
        "mean_consensus_agreement": float(agreement.mean()),
        "median_consensus_agreement": float(np.median(agreement)),
        "mean_consensus_entropy": float(entropy.mean()),
        "min_pairwise_ARI": float(pairwise_df["ARI"].min()),
        "mean_pairwise_ARI": float(pairwise_df["ARI"].mean()),
        "max_pairwise_ARI": float(pairwise_df["ARI"].max()),
        "random_state": random_state,
        "estimator_parameters": {name: model.get_params() for name, model in models.items()},
    }
    write_csv(pd.DataFrame([metrics]), output / f"{output_prefix}_consensus_summary.csv", index=False)
    from .report import write_json
    write_json(output / f"{output_prefix}_label_mappings.json", mappings)
    return {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items()}
