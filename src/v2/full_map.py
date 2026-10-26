from __future__ import annotations

from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .config import ALL_FEATURES, validate_prefix
from .report import ensure_new, ensure_separate_artifacts
from .anomaly_reference import empirical_percentile, load_anomaly_reference
from .model_artifacts import (
    fit_production_kmeans, load_production_kmeans, validate_features,
    matrix_sha256, preprocessing_config, sha256_file,
)
from .raster_features import (
    aligned_rasters, derive_local_features, iter_blocks,
    _read_aligned_window, _feature_window, plan_core_window,
)

CLUSTER_NODATA = 255
FLOAT_NODATA = -9999.0


def _percentile_reference(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return empirical_percentile(values, query)


def predict_full_v2_maps(
    model_artifact: str | Path,
    sample_csv: str | Path,
    output_dir: str | Path,
    k: int | None = None,
    random_state: int = 42,
    block_size: int = 512,
    local_radius: int = 2,
    isolation_contamination: float = 0.01,
    isolation_estimators: int = 300,
    isolation_fit_subset: int = 50_000,
    *,
    anomaly_reference: str | Path | None = None,
    output_prefix: str = "v2",
    n_init: int | None = None, max_iter: int | None = None,
    raster_inputs=None, core_window=None, halo_policy="clip",
) -> dict[str, str | int | float]:
    """Predict using saved production and separate anomaly artifacts; never refit.

    Classification/confidence use production KMeans. Anomaly scores use the
    saved anomaly scaler/KMeans/IsolationForest and their fixed references.
    """
    validate_prefix(output_prefix)
    ensure_new(output_dir)
    ensure_separate_artifacts(output_dir, model_artifact, anomaly_reference)
    production = load_production_kmeans(
        model_artifact, expected_preprocessing=preprocessing_config(local_radius)
    )
    scaler, model = production.scaler, production.model
    if k is not None and k != model.n_clusters:
        raise ValueError("Requested k does not match the saved production model.")
    k = model.n_clusters
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite map destination: {out}")
    sample = pd.read_csv(sample_csv, float_precision="round_trip")
    features = production.metadata["feature_names"]
    missing = [name for name in features if name not in sample.columns]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = validate_features(sample[features].to_numpy(dtype=np.float64), features)
    if matrix_sha256(X) != production.metadata["input_sample"]["feature_matrix_sha256"]:
        raise ValueError("Anomaly reference sample differs from the production training features.")
    sample_hash = sha256_file(sample_csv)
    recorded_hash = production.metadata["input_sample"].get("sha256")
    if recorded_hash and recorded_hash != sample_hash:
        raise ValueError("Reference sample file hash differs from the production artifact.")
    anomaly = load_anomaly_reference(
        anomaly_reference, expected_preprocessing=preprocessing_config(local_radius)
    )
    anomaly.validate_sample(X, sample_csv)
    anomaly.validate_configuration(
        k=k, random_state=random_state, contamination=isolation_contamination,
        n_estimators=isolation_estimators, fit_subset=isolation_fit_subset,
        n_init=n_init, max_iter=max_iter,
    )

    if ((n_init is not None and model.n_init != n_init) or
        (max_iter is not None and model.max_iter != max_iter)):
        raise ValueError("Requested production KMeans configuration differs from saved model.")
    cluster_path = out / f"venus_{output_prefix}_clusters_k{k}.tif"
    confidence_path = out / f"venus_{output_prefix}_confidence.tif"
    anomaly_path = out / f"venus_{output_prefix}_anomaly.tif"

    ctx, radar, sources = aligned_rasters() if raster_inputs is None else aligned_rasters(raster_inputs)
    with ctx:
        plan = plan_core_window(radar,core_window,local_radius=local_radius,halo_policy=halo_policy) if core_window is not None else None
        region = plan.core if plan is not None else Window(0,0,radar.width,radar.height)
        out.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(model.cluster_centers_, columns=features).assign(cluster=np.arange(k)).to_csv(
            out / f"{output_prefix}_kmeans_centroids_standardized.csv", index=False
        )
        pd.DataFrame(scaler.mean_, index=features, columns=["mean"]).assign(scale=scaler.scale_).to_csv(
            out / f"{output_prefix}_scaler_parameters.csv"
        )
        profile = radar.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=CLUSTER_NODATA, compress="lzw", predictor=2)
        if plan is not None:
            profile.update(width=int(region.width),height=int(region.height),transform=plan.core_transform)
        float_profile = radar.profile.copy()
        float_profile.update(count=1, dtype="float32", nodata=FLOAT_NODATA, compress="lzw", predictor=3)
        if plan is not None:
            float_profile.update(width=int(region.width),height=int(region.height),transform=plan.core_transform)
        with rasterio.open(cluster_path, "w", **profile) as dst_cluster, \
             rasterio.open(confidence_path, "w", **float_profile) as dst_conf, \
             rasterio.open(anomaly_path, "w", **float_profile) as dst_anomaly:
            for destination in (dst_cluster, dst_conf, dst_anomaly):
                destination.update_tags(
                    MODEL_ID=production.model_id,
                    MODEL_ARTIFACT_SCHEMA=str(production.metadata["schema_version"]),
                    MODEL_MANIFEST_SHA256=sha256_file(production.directory / "manifest.json"),
                    MODEL_ROLE="production_full_map_kmeans",
                )
            dst_anomaly.update_tags(
                ANOMALY_MODEL_ID=anomaly.metadata["model_id"],
                ANOMALY_REFERENCE_ID=anomaly.metadata["reference_id"],
                ANOMALY_CENTROID_MODEL_ID=anomaly.metadata["centroid_model_id"],
                ISOLATION_MODEL_ID=anomaly.metadata["isolation_model_id"],
                ANOMALY_MANIFEST_SHA256=sha256_file(anomaly.directory / "manifest.json"),
                ISOLATION_FOREST_ROLE="saved_separate_anomaly_experiment",
            )
            px_x, px_y = abs(radar.res[0]), abs(radar.res[1])
            for local_row, local_col, local_window in iter_blocks(int(region.width), int(region.height), block_size):
                row = local_row + int(region.row_off)
                col = local_col + int(region.col_off)
                base_window = Window(col,row,local_window.width,local_window.height)
                read_window = _feature_window(base_window, radar.width, radar.height, local_radius)
                arrays, masks = _read_aligned_window(sources, read_window)
                valid = np.logical_and.reduce([masks[name] for name in features[:5]])
                derived = derive_local_features(arrays, masks, local_radius, px_x, px_y)
                local_features = {**arrays, **derived}

                r0 = row - int(read_window.row_off)
                c0 = col - int(read_window.col_off)
                h = int(base_window.height)
                w = int(base_window.width)
                center_valid = valid[r0:r0 + h, c0:c0 + w]

                cluster_block = np.full((h, w), CLUSTER_NODATA, dtype=np.uint8)
                confidence_block = np.full((h, w), FLOAT_NODATA, dtype=np.float32)
                anomaly_block = np.full((h, w), FLOAT_NODATA, dtype=np.float32)

                if np.any(center_valid):
                    stack = np.column_stack([
                        local_features[name][r0:r0 + h, c0:c0 + w][center_valid]
                        for name in ALL_FEATURES
                    ])
                    block_scaled = production.transform(stack, ALL_FEATURES)
                    labels = model.predict(block_scaled)
                    distances = model.transform(block_scaled)
                    nearest = distances.min(axis=1)
                    order = np.argsort(distances, axis=1)
                    second = distances[np.arange(len(nearest)), order[:, 1]]
                    logits = -distances
                    logits -= logits.max(axis=1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= probs.sum(axis=1, keepdims=True)
                    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(k)
                    confidence = 1.0 - entropy

                    combined = anomaly.score(stack, ALL_FEATURES)["combined_anomaly_score"]

                    flat_idx = np.flatnonzero(center_valid)
                    cluster_block.ravel()[flat_idx] = labels.astype(np.uint8)
                    confidence_block.ravel()[flat_idx] = confidence.astype(np.float32)
                    anomaly_block.ravel()[flat_idx] = combined.astype(np.float32)

                dst_cluster.write(cluster_block, 1, window=local_window)
                dst_conf.write(confidence_block, 1, window=local_window)
                dst_anomaly.write(anomaly_block, 1, window=local_window)

    association = {
        "model_id": production.model_id,
        "model_artifact": str(production.directory),
        "model_manifest_sha256": sha256_file(production.directory / "manifest.json"),
        "feature_names": features,
        "window": plan.to_dict() if plan is not None else None,
        "raster_inputs": {name: str(Path(path).resolve()) for name,path in raster_inputs.items()} if raster_inputs is not None else None,
        "kmeans_parameters": model.get_params(),
        "preprocessing": production.metadata["preprocessing"],
        "recorded_sample_provenance": production.provenance,
        "reference_sample": {"path": str(Path(sample_csv).resolve()), "sha256": sample_hash},
        "outputs": {
            "clusters": str(cluster_path), "confidence": str(confidence_path),
            "anomaly": str(anomaly_path),
            "centroids": str(out / f"{output_prefix}_kmeans_centroids_standardized.csv"),
            "scaler": str(out / f"{output_prefix}_scaler_parameters.csv"),
        },
        "anomaly_reference": {
            "artifact": str(anomaly.directory),
            "manifest_sha256": sha256_file(anomaly.directory / "manifest.json"),
            "model_id": anomaly.metadata["model_id"],
            "reference_id": anomaly.metadata["reference_id"],
            "centroid_model_id": anomaly.metadata["centroid_model_id"],
            "isolation_model_id": anomaly.metadata["isolation_model_id"],
            "role": "separate saved anomaly experiment; not the production classifier",
            "percentile": anomaly.metadata["percentile"],
            "composite": anomaly.metadata["composite"],
        },
    }
    with (out / f"{output_prefix}_production_model.json").open("x", encoding="utf-8") as stream:
        json.dump(association, stream, indent=2)
    return {
        "model_id": production.model_id,
        "model_artifact": str(production.directory),
        "model_association": str(out / f"{output_prefix}_production_model.json"),
        "rows": int(len(sample)),
        "k": int(k),
        "cluster_raster": str(cluster_path),
        "confidence_raster": str(confidence_path),
        "anomaly_raster": str(anomaly_path),
    }



def fit_full_map_kmeans(
    sample_csv: str | Path,
    artifact_dir: str | Path,
    *,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 500,
    local_radius: int = 2,
):
    """Fit/save a separate production experiment; no raster inputs are opened."""
    ensure_new(artifact_dir)
    destination = Path(artifact_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact destination: {destination}")
    sample_path = Path(sample_csv).expanduser().resolve()
    before = sha256_file(sample_path)
    sample = pd.read_csv(sample_path, float_precision="round_trip")
    missing = [name for name in ALL_FEATURES if name not in sample.columns]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = validate_features(sample[ALL_FEATURES].to_numpy(dtype=np.float64), ALL_FEATURES)
    if sha256_file(sample_path) != before:
        raise ValueError("Training sample changed while it was being read.")
    return fit_production_kmeans(
        X, ALL_FEATURES, destination, k=k, random_state=random_state,
        n_init=n_init, max_iter=max_iter, local_radius=local_radius,
        sample_provenance={
            "path": str(sample_path), "sha256": before,
            "preprocessing_lineage": "Caller-declared; not independently recoverable from this CSV.",
        },
    )


def build_full_v2_maps(
    sample_csv: str | Path,
    output_dir: str | Path,
    k: int = 4,
    random_state: int = 42,
    block_size: int = 512,
    local_radius: int = 2,
    isolation_contamination: float = 0.01,
    isolation_estimators: int = 300,
    isolation_fit_subset: int = 50_000,
    *,
    model_artifact: str | Path | None = None,
    anomaly_reference: str | Path | None = None,
    output_prefix: str = "v2", n_init: int = 20, max_iter: int = 500,
    raster_inputs=None, core_window=None, halo_policy="clip",
) -> dict[str, str | int | float]:
    """Compatibility orchestrator: explicit fit/save, then artifact-based mapping.

    Existing main_v2.py callers retain fit-then-map behaviour. Call
    predict_full_v2_maps with an existing artifact for production inference only.
    The consensus, uncertainty and anomaly-analysis modules are not shared fits.
    """
    validate_prefix(output_prefix)
    ensure_new(output_dir)
    ensure_separate_artifacts(output_dir, model_artifact, anomaly_reference)
    # Validate the explicit anomaly reference before fitting any production model.
    anomaly = load_anomaly_reference(
        anomaly_reference, expected_preprocessing=preprocessing_config(local_radius)
    )
    anomaly.validate_configuration(
        k=k, random_state=random_state, contamination=isolation_contamination,
        n_estimators=isolation_estimators, fit_subset=isolation_fit_subset,
        n_init=n_init, max_iter=max_iter,
    )
    sample = pd.read_csv(sample_csv, float_precision="round_trip")
    missing = [name for name in ALL_FEATURES if name not in sample]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    anomaly.validate_sample(validate_features(sample[ALL_FEATURES].to_numpy(dtype=np.float64),
                                               ALL_FEATURES), sample_csv)
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite map destination: {out}")
    if model_artifact is None:
        directory = out.parent / "models" / ("full_map_kmeans_" + uuid.uuid4().hex)
        production = fit_full_map_kmeans(
            sample_csv, directory, k=k, random_state=random_state,
            n_init=n_init, max_iter=max_iter, local_radius=local_radius,
        )
        model_artifact = production.directory
    return predict_full_v2_maps(
        model_artifact, sample_csv, out, k=k, random_state=random_state,
        block_size=block_size, local_radius=local_radius,
        isolation_contamination=isolation_contamination,
        isolation_estimators=isolation_estimators,
        isolation_fit_subset=isolation_fit_subset,
        anomaly_reference=anomaly.directory,
        output_prefix=output_prefix, n_init=n_init, max_iter=max_iter,
        raster_inputs=raster_inputs, core_window=core_window, halo_policy=halo_policy,
    )
