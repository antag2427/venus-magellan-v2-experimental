"""Versioned production full-map KMeans artifacts; unrelated experiments stay separate.

Pickle preserves fitted estimators, so load only trusted local artifacts.
A final manifest is the completion marker; pending files are never loadable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import uuid

import numpy as np
import pandas as pd
import rasterio
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import ALL_FEATURES
from .report import ensure_new

ARTIFACT_TYPE = "venus_v2_production_full_map_kmeans"
SCHEMA_VERSION = 1
MODEL_FILE = "model.pkl"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_sha256(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def preprocessing_config(local_radius: int = 2) -> dict:
    if not isinstance(local_radius, (int, np.integer)) or local_radius < 0:
        raise ValueError("local_radius must be a nonnegative integer.")
    return {
        "schema_id": "v2_17_float64_imputation_derivative_halo_v1",
        "local_radius": int(local_radius),
        "halo_radius": int(local_radius) + 1,
        "companion_dtype": "float64",
        "resampling": "bilinear",
        "destination_nodata": -32768.0,
        "reference_grid": "radar CRS, transform, width and height",
        "validity": "finite base values; per-layer NoData; common five-base-feature mask",
        "calibration": "stored numerical values; no additional physical conversion",
        "raster_features_sha256": sha256_file(Path(__file__).with_name("raster_features.py")),
        "rasterio_version": rasterio.__version__,
        "gdal_version": rasterio.__gdal_version__,
    }


def validate_features(values: np.ndarray, feature_names) -> np.ndarray:
    if list(feature_names) != ALL_FEATURES:
        raise ValueError("Feature names/order must exactly match the V2 17-feature schema.")
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be a numeric matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(ALL_FEATURES):
        raise ValueError("Feature matrix dimension must be (samples, 17).")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains nonfinite values.")
    return matrix


def _new_destination(directory: Path) -> None:
    ensure_new(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact destination: {directory}")


def _validate_metadata(metadata: dict) -> None:
    required = {
        "artifact_type", "schema_version", "model_id", "feature_names",
        "feature_count", "kmeans_parameters", "scaling", "preprocessing",
        "input_sample", "environment", "model_file", "model_sha256",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Malformed artifact metadata: missing required fields.")
    if metadata["artifact_type"] != ARTIFACT_TYPE or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Incompatible artifact type or schema version; V1 artifacts are not accepted.")
    if not isinstance(metadata["model_id"], str) or not metadata["model_id"].strip():
        raise ValueError("Malformed model identifier.")
    if metadata["feature_names"] != ALL_FEATURES or metadata["feature_count"] != len(ALL_FEATURES):
        raise ValueError("Artifact feature schema/order or dimension mismatch.")
    if metadata["model_file"] != MODEL_FILE:
        raise ValueError("Malformed artifact model-file reference.")
    if not isinstance(metadata["model_sha256"], str) or len(metadata["model_sha256"]) != 64:
        raise ValueError("Malformed model hash.")
    environment = metadata["environment"]
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(key), str) or not environment[key]
        for key in ("python", "sklearn", "numpy", "scipy", "pandas")
    ):
        raise ValueError("Malformed environment metadata.")
    if environment["sklearn"] != sklearn.__version__:
        raise ValueError("Incompatible scikit-learn environment version.")
    if not isinstance(metadata["kmeans_parameters"], dict) or not isinstance(metadata["scaling"], dict):
        raise ValueError("Malformed estimator/scaling parameters.")
    sample = metadata["input_sample"]
    required_sample = {"rows", "feature_matrix_sha256", "path", "sha256"}
    if not isinstance(sample, dict) or not required_sample.issubset(sample):
        raise ValueError("Malformed input-sample provenance fields.")
    if not isinstance(sample["rows"], int) or sample["rows"] < 1:
        raise ValueError("Malformed input-sample row count.")
    digest = sample["feature_matrix_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Malformed training feature-matrix hash.")
    if sample["path"] is not None:
        digest = sample["sha256"]
        if not isinstance(sample["path"], str) or not sample["path"].strip():
            raise ValueError("Malformed input-sample file reference.")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed input-sample file hash.")
    elif sample["sha256"] is not None:
        raise ValueError("Input-sample hash has no file reference.")
    config = metadata["preprocessing"]
    if not isinstance(config, dict):
        raise ValueError("Malformed preprocessing configuration.")
    radius = config.get("local_radius")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Malformed preprocessing radius.")
    expected = preprocessing_config(radius)
    if set(config) != set(expected):
        raise ValueError("Malformed preprocessing fields.")
    # Scientific semantics must be recognized. Source/version fingerprints are
    # recorded here and checked against the runtime by raster-map inference.
    for key in expected.keys() - {"raster_features_sha256", "rasterio_version", "gdal_version"}:
        if config[key] != expected[key]:
            raise ValueError(f"Incompatible preprocessing field: {key}")


def _validate_fitted(scaler, model, metadata: dict) -> None:
    if not isinstance(scaler, StandardScaler) or not isinstance(model, KMeans):
        raise ValueError("Artifact must contain fitted StandardScaler and KMeans objects.")
    try:
        check_is_fitted(scaler, ["mean_", "scale_", "var_", "n_features_in_"])
        check_is_fitted(model, ["cluster_centers_", "n_features_in_"])
    except Exception as exc:
        raise ValueError("Artifact contains an unfitted estimator.") from exc
    n = len(ALL_FEATURES)
    if scaler.n_features_in_ != n or model.n_features_in_ != n:
        raise ValueError("Incompatible fitted scaler/model feature dimension.")
    if scaler.mean_.shape != (n,) or scaler.scale_.shape != (n,) or scaler.var_.shape != (n,):
        raise ValueError("Incompatible fitted scaler dimension.")
    if model.cluster_centers_.shape != (model.n_clusters, n):
        raise ValueError("Incompatible KMeans centre dimension.")
    for array in (scaler.mean_, scaler.scale_, scaler.var_, model.cluster_centers_):
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite fitted model parameters.")
    if (scaler.scale_ <= 0).any() or (scaler.var_ < 0).any():
        raise ValueError("Invalid fitted scaler parameters.")
    if model.get_params() != metadata["kmeans_parameters"]:
        raise ValueError("Fitted model parameters do not match the manifest.")
    if scaler.get_params() != metadata["scaling"]:
        raise ValueError("Fitted scaler parameters do not match the manifest.")


def _read_state(path: Path, metadata: dict):
    _validate_metadata(metadata)
    if not path.is_file():
        raise ValueError(f"Incomplete artifact: missing {MODEL_FILE}.")
    if sha256_file(path) != metadata["model_sha256"]:
        raise ValueError("Artifact model hash mismatch.")
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception as exc:
        raise ValueError("Malformed serialized model artifact.") from exc
    if not isinstance(payload, dict) or set(payload) != {"scaler", "model"}:
        raise ValueError("Malformed serialized estimator payload.")
    try:
        _validate_fitted(payload["scaler"], payload["model"], metadata)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Malformed fitted estimator state.") from exc
    # A fitted-attribute check alone cannot establish that private estimator
    # state is sufficient for prediction. This probe does not fit either object.
    try:
        probe = payload["scaler"].transform(payload["scaler"].mean_[None, :])
        labels = payload["model"].predict(probe)
        distances = payload["model"].transform(probe)
        if labels.shape != (1,) or distances.shape != (1, payload["model"].n_clusters):
            raise ValueError("Unexpected predictor output dimensions.")
        if not np.isfinite(distances).all() or not 0 <= labels[0] < payload["model"].n_clusters:
            raise ValueError("Invalid predictor output.")
    except Exception as exc:
        raise ValueError("Artifact estimator state cannot predict reliably.") from exc
    return payload["scaler"], payload["model"]


@dataclass(frozen=True)
class ProductionKMeansArtifact:
    directory: Path
    scaler: StandardScaler
    model: KMeans
    metadata: dict

    @property
    def model_id(self) -> str:
        return self.metadata["model_id"]

    @property
    def provenance(self) -> dict:
        source = self.metadata["input_sample"]
        if not source.get("path"):
            return {"status": "in_memory_only", "limitation": "Training matrix hashed; no source sample file supplied."}
        path = Path(source["path"])
        if not path.is_file():
            return {"status": "unavailable", "limitation": f"Recorded input sample cannot be verified: {path}"}
        if not source.get("sha256") or sha256_file(path) != source["sha256"]:
            return {"status": "hash_mismatch", "limitation": f"Recorded input sample hash does not match: {path}"}
        return {
            "status": "verified",
            "limitation": "File identity verified; CSV preprocessing lineage is caller-declared, not recoverable from values alone.",
        }

    def transform(self, values: np.ndarray, feature_names) -> np.ndarray:
        matrix = validate_features(values, feature_names)
        if not len(matrix):
            return np.empty_like(matrix)
        return self.scaler.transform(matrix)

    def predict(self, values: np.ndarray, feature_names) -> np.ndarray:
        scaled = self.transform(values, feature_names)
        if not len(scaled):
            return np.empty(0, dtype=np.int32)
        return self.model.predict(scaled)


def load_production_kmeans(directory: str | Path, *, expected_preprocessing: dict | None = None) -> ProductionKMeansArtifact:
    directory = Path(directory).expanduser().resolve()
    manifest = directory / MANIFEST_FILE
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or malformed completed artifact manifest: {manifest}") from exc
    _validate_metadata(metadata)
    if expected_preprocessing is not None and metadata["preprocessing"] != expected_preprocessing:
        raise ValueError("Artifact preprocessing does not match the requested raster pipeline.")
    scaler, model = _read_state(directory / MODEL_FILE, metadata)
    return ProductionKMeansArtifact(directory, scaler, model, metadata)


def fit_production_kmeans(
    values: np.ndarray,
    feature_names,
    output_dir: str | Path,
    *,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 500,
    local_radius: int = 2,
    sample_provenance: dict | None = None,
) -> ProductionKMeansArtifact:
    """Fit a NEW production experiment. Never call this from inference-only code."""
    directory = Path(output_dir).expanduser().resolve()
    _new_destination(directory)  # Reject conflicts before fitting or writing.
    matrix = validate_features(values, feature_names)
    if not isinstance(k, (int, np.integer)) or not 2 <= k <= 255 or len(matrix) < k:
        raise ValueError("KMeans requires 2 <= k <= 255 and at least k training samples.")
    preprocessing = preprocessing_config(local_radius)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter).fit(scaled)
    info = dict(sample_provenance or {})
    info.setdefault("path", None)
    info.setdefault("sha256", None)
    info.update(rows=len(matrix), feature_matrix_sha256=matrix_sha256(matrix))
    metadata = {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "model_id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(ALL_FEATURES), "feature_count": len(ALL_FEATURES),
        "kmeans_parameters": model.get_params(), "scaling": scaler.get_params(),
        "preprocessing": preprocessing, "input_sample": info,
        "environment": {
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        },
        "model_file": MODEL_FILE, "model_sha256": "",
    }
    # Reserve an entirely new directory. On failure, pending data is identifiable
    # and deliberately retained; without manifest.json it cannot appear complete.
    directory.mkdir(parents=True, exist_ok=False)
    pending_model = directory / (MODEL_FILE + ".pending")
    pending_manifest = directory / (MANIFEST_FILE + ".pending")
    with pending_model.open("xb") as stream:
        pickle.dump({"scaler": scaler, "model": model}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    metadata["model_sha256"] = sha256_file(pending_model)
    with pending_manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    restored_scaler, restored_model = _read_state(pending_model, metadata)
    np.testing.assert_array_equal(model.predict(scaled), restored_model.predict(restored_scaler.transform(matrix)))
    pending_model.rename(directory / MODEL_FILE)
    pending_manifest.rename(directory / MANIFEST_FILE)  # Publish completion last.
    artifact = load_production_kmeans(directory, expected_preprocessing=preprocessing)
    np.testing.assert_array_equal(artifact.predict(matrix, ALL_FEATURES), model.predict(scaled))
    return artifact
