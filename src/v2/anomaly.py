from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ALL_FEATURES
from .report import ensure_outputs_new, ensure_new, ensure_separate_artifacts, write_csv
from .model_artifacts import validate_features, preprocessing_config, sha256_file
from .anomaly_reference import (
    empirical_percentile, fit_anomaly_reference, load_anomaly_reference,
)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Compatibility helper: score a sample against its own right-inclusive ECDF."""
    return empirical_percentile(values, values)


def run_anomaly_detection(
    sample_csv: str | Path,
    output_csv: str | Path,
    k: int = 4,
    random_state: int = 42,
    contamination: float = 0.01,
    n_estimators: int = 300,
    fit_subset: int = 50_000,
    *,
    reference_artifact: str | Path | None = None,
    reference_output_dir: str | Path | None = None,
    local_radius: int = 2,
    n_init: int = 20, max_iter: int = 500,
) -> dict[str, float | int | str]:
    """Fit/save a NEW separate anomaly experiment, or reuse an explicit saved one.

    Cluster IDs here belong to the anomaly KMeans, not production classification.
    A fixed model/reference pair is passed to map inference; it never refits.
    """
    output = Path(output_csv).expanduser().resolve()
    ensure_outputs_new(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite anomaly CSV: {output}")
    if reference_artifact is not None and reference_output_dir is not None:
        raise ValueError("Supply either an existing reference or a new reference destination.")
    if reference_artifact is None:
        destination = reference_output_dir or output.with_name(output.stem + "_reference")
        ensure_new(destination)
        ensure_separate_artifacts(output, destination)
        artifact = fit_anomaly_reference(
            sample_csv, destination, k=k, random_state=random_state,
            contamination=contamination, n_estimators=n_estimators,
            fit_subset=fit_subset, local_radius=local_radius, n_init=n_init, max_iter=max_iter,
        )
    else:
        ensure_separate_artifacts(output, reference_artifact)
        artifact = load_anomaly_reference(
            reference_artifact, expected_preprocessing=preprocessing_config(local_radius)
        )
    artifact.validate_configuration(k=k, random_state=random_state, contamination=contamination,
                                    n_estimators=n_estimators, fit_subset=fit_subset, n_init=n_init, max_iter=max_iter)
    before = sha256_file(sample_csv)
    df = pd.read_csv(sample_csv, float_precision="round_trip")
    missing = [name for name in ALL_FEATURES if name not in df]
    if missing:
        raise ValueError(f"Anomaly sample is missing features: {missing}")
    X = validate_features(df[ALL_FEATURES].to_numpy(dtype=np.float64), ALL_FEATURES)
    artifact.validate_sample(X, sample_csv)
    if sha256_file(sample_csv) != before:
        raise ValueError("Anomaly sample changed while reading.")
    scores = artifact.score(X, ALL_FEATURES)
    out = df[[c for c in ("row", "col") if c in df.columns]].copy()
    for key, values in scores.items():
        out[key] = values
    out["anomaly_model_id"] = artifact.metadata["model_id"]
    out["anomaly_reference_id"] = artifact.metadata["reference_id"]
    out["anomaly_manifest_sha256"] = sha256_file(artifact.directory / "manifest.json")
    out = out.sort_values("combined_anomaly_score", ascending=False).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, output, index=False, float_format="%.17g")
    combined = scores["combined_anomaly_score"]
    return {
        "rows": int(len(out)), "fit_subset": artifact.metadata["fit_subset_rows"],
        "top_1pct_count": int(max(1, round(len(out) * 0.01))),
        "mean_combined_score": float(combined.mean()),
        "max_combined_score": float(combined.max()), "output": str(output),
        "reference_artifact": str(artifact.directory),
        "estimator_parameters": artifact.metadata["estimator_parameters"],
        "model_id": artifact.metadata["model_id"], "reference_id": artifact.metadata["reference_id"],
    }
