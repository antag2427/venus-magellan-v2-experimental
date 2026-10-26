"""Persistent, separate anomaly scorer and fixed empirical references.

Only load trusted local pickle artifacts. The final manifest marks completion.
Production classification estimators remain in their own KMeans artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import uuid

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES
from .report import ensure_new
from .model_artifacts import (
    validate_features, preprocessing_config, matrix_sha256, sha256_file,
)

ARTIFACT_TYPE = "venus_v2_anomaly_reference"
SCHEMA_VERSION = 1
SCORES = {
    "centroid_distance": "min(KMeans.transform(StandardScaler.transform(X)), axis=1); higher is more anomalous",
    "isolation_anomaly_score": "-IsolationForest.score_samples(StandardScaler.transform(X)); higher is more anomalous",
}
PERCENTILE = {
    "definition": "count(reference_scores <= query_score) / len(reference_scores)",
    "tie_policy": "right-inclusive",
    "below_minimum": 0, "at_or_above_maximum": 1,
}
COMPOSITE = {"centroid_weight": 0.5, "isolation_weight": 0.5,
             "definition": "0.5 * centroid_percentile + 0.5 * isolation_percentile"}


def _scores(values, name, *, nonempty=False):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array.") from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one-dimensional and finite.")
    if nonempty and not len(array):
        raise ValueError(f"{name} must not be empty.")
    return array


def empirical_percentile(reference_scores, query_scores):
    """Right-inclusive ECDF; empty queries allowed, empty references rejected.

    Below-minimum queries map to zero; at/above maximum to one. All inputs
    must be finite 1D arrays. Ties are independent of input order.
    """
    reference = _scores(reference_scores, "Reference scores", nonempty=True)
    query = _scores(query_scores, "Query scores")
    return np.searchsorted(np.sort(reference), query, side="right") / len(reference)


@dataclass(frozen=True)
class AnomalyReference:
    directory: Path
    metadata: dict
    scaler: StandardScaler
    kmeans: KMeans
    isolation: IsolationForest
    centroid_reference: np.ndarray
    isolation_reference: np.ndarray

    def score_raw(self, centroid_distance, isolation_raw):
        centroid = _scores(centroid_distance, "Centroid scores")
        isolation = _scores(isolation_raw, "Isolation scores")
        if centroid.shape != isolation.shape:
            raise ValueError("Raw score arrays must have the same shape.")
        cp = empirical_percentile(self.centroid_reference, centroid)
        ip = empirical_percentile(self.isolation_reference, isolation)
        return {
            "centroid_distance": centroid,
            "isolation_anomaly_score": isolation,
            "centroid_anomaly_percentile": cp,
            "isolation_anomaly_percentile": ip,
            "combined_anomaly_score": 0.5 * cp + 0.5 * ip,
        }

    def score(self, values, feature_names):
        matrix = validate_features(values, feature_names)
        if not len(matrix):
            return dict(cluster=np.empty(0, dtype=int),
                        **self.score_raw(np.empty(0), np.empty(0)))
        scaled = self.scaler.transform(matrix)
        return dict(
            cluster=self.kmeans.predict(scaled),
            **self.score_raw(self.kmeans.transform(scaled).min(axis=1),
                             -self.isolation.score_samples(scaled)),
        )

    def validate_sample(self, values, sample_path):
        info = self.metadata["input_sample"]
        if len(values) != info["rows"] or matrix_sha256(values) != info["feature_matrix_sha256"]:
            raise ValueError("Anomaly reference sample identity does not match.")
        if sha256_file(sample_path) != info["sha256"]:
            raise ValueError("Anomaly reference sample file hash does not match.")

    def validate_configuration(self, *, k, random_state, contamination,
                               n_estimators, fit_subset, n_init=None, max_iter=None):
        expected = {"k": k, "random_state": random_state, "contamination": contamination,
                    "n_estimators": n_estimators, "fit_subset": fit_subset}
        params = self.metadata["estimator_parameters"]["kmeans"]
        if ((n_init is not None and params["n_init"] != n_init) or
            (max_iter is not None and params["max_iter"] != max_iter)):
            raise ValueError("Anomaly KMeans configuration does not match requested run.")
        if expected != self.metadata["configuration"]:
            raise ValueError("Anomaly reference configuration does not match requested run.")


def _validate_metadata(meta):
    required = {"artifact_type", "schema_version", "model_id", "reference_id",
                "centroid_model_id", "isolation_model_id", "feature_names", "preprocessing",
                "score_definitions", "percentile", "composite", "input_sample",
                "configuration", "environment", "estimator_parameters", "state_file",
                "state_sha256", "fit_subset_rows", "fit_subset_indices_sha256", "created_utc"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("Malformed anomaly reference metadata.")
    if meta["artifact_type"] != ARTIFACT_TYPE or meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Incompatible anomaly reference artifact version or type.")
    if meta["feature_names"] != ALL_FEATURES:
        raise ValueError("Incompatible anomaly reference feature schema/order.")
    if meta["score_definitions"] != SCORES or meta["percentile"] != PERCENTILE or meta["composite"] != COMPOSITE:
        raise ValueError("Incompatible anomaly score definitions, percentile or composite.")
    for key in ("model_id", "reference_id", "centroid_model_id", "isolation_model_id"):
        try:
            uuid.UUID(hex=meta[key])
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Malformed anomaly identity: {key}.") from exc
    env = meta["environment"]
    if not isinstance(env, dict) or any(not isinstance(env.get(k), str) or not env[k]
                                       for k in ("python", "sklearn", "numpy", "scipy", "pandas")):
        raise ValueError("Malformed anomaly software environment.")
    if env["sklearn"] != sklearn.__version__:
        raise ValueError("Incompatible scikit-learn version.")
    if meta["state_file"] != "state.pkl":
        raise ValueError("Malformed anomaly state filename.")
    for digest in (meta["state_sha256"], meta["fit_subset_indices_sha256"]):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed anomaly state/reference hash.")
    info = meta["input_sample"]
    if not isinstance(info, dict) or not {"path", "sha256", "rows", "feature_matrix_sha256"}.issubset(info):
        raise ValueError("Malformed anomaly sample provenance.")
    if not isinstance(info["path"], str) or not info["path"] or not isinstance(info["rows"], int) or info["rows"] < 1:
        raise ValueError("Malformed anomaly sample identity.")
    for key in ("sha256", "feature_matrix_sha256"):
        digest = info[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed anomaly sample hash.")
    config = meta["preprocessing"]
    if not isinstance(config, dict) or "local_radius" not in config:
        raise ValueError("Malformed anomaly preprocessing.")
    if config != preprocessing_config(config["local_radius"]):
        raise ValueError("Incompatible anomaly preprocessing.")


def load_anomaly_reference(directory, *, expected_model_id=None,
                           expected_reference_id=None, expected_preprocessing=None):
    """Load saved state without fitting; hashes detect corruption, not hostile pickle."""
    if directory is None:
        raise ValueError("An explicit saved anomaly reference is required.")
    directory = Path(directory).expanduser().resolve()
    try:
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Missing or malformed completed anomaly reference manifest.") from exc
    _validate_metadata(meta)
    if ((expected_model_id is not None and expected_model_id != meta["model_id"]) or
        (expected_reference_id is not None and expected_reference_id != meta["reference_id"])):
        raise ValueError("Anomaly model/reference identity mismatch.")
    if expected_preprocessing is not None and expected_preprocessing != meta["preprocessing"]:
        raise ValueError("Anomaly reference preprocessing mismatch.")
    return _read_state(directory / "state.pkl", meta, directory)


def _read_state(state, meta, directory):
    """Validate the entire serialized contract before publishing or using it."""
    if not state.is_file() or sha256_file(state) != meta["state_sha256"]:
        raise ValueError("Missing anomaly state or state hash mismatch.")
    try:
        with state.open("rb") as stream:
            payload = pickle.load(stream)
        # Bind every semantic manifest field and all model identities to the state.
        contract = {k: v for k, v in meta.items() if k != "state_sha256"}
        if payload["contract"] != contract:
            raise ValueError("Anomaly state/manifest identity or contract mismatch.")
        scaler, km, iso = payload["scaler"], payload["kmeans"], payload["isolation"]
        if not isinstance(scaler, StandardScaler) or not isinstance(km, KMeans) or not isinstance(iso, IsolationForest):
            raise ValueError("Unexpected anomaly estimator types.")
        for estimator in (scaler, km, iso):
            if estimator.n_features_in_ != len(ALL_FEATURES):
                raise ValueError("Incompatible anomaly estimator dimensions.")
        params = {"scaler": scaler.get_params(), "kmeans": km.get_params(), "isolation": iso.get_params()}
        if params != meta["estimator_parameters"]:
            raise ValueError("Anomaly fitted parameters differ from manifest.")
        cfg = meta["configuration"]
        if cfg != {"k": km.n_clusters, "random_state": km.random_state,
                   "contamination": iso.contamination, "n_estimators": iso.n_estimators,
                   "fit_subset": cfg["fit_subset"]} or iso.random_state != km.random_state:
            raise ValueError("Inconsistent anomaly configuration.")
        if meta["fit_subset_rows"] != min(cfg["fit_subset"], meta["input_sample"]["rows"]):
            raise ValueError("Inconsistent Isolation Forest subset size.")
        for a in (scaler.mean_, scaler.var_, scaler.scale_, km.cluster_centers_):
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite anomaly fitted state.")
        if np.any(scaler.scale_ <= 0):
            raise ValueError("Invalid anomaly scaler.")
        cr = _scores(payload["centroid_reference"], "Centroid reference", nonempty=True)
        ir = _scores(payload["isolation_reference"], "Isolation reference", nonempty=True)
        if len(cr) != meta["input_sample"]["rows"] or len(ir) != len(cr):
            raise ValueError("Anomaly reference size differs from sample.")
        artifact = AnomalyReference(directory, meta, scaler, km, iso, cr, ir)
        artifact.score(scaler.mean_[None, :], ALL_FEATURES)  # bounded fitted-state probe
    except Exception as exc:
        raise ValueError(f"Invalid anomaly reference state: {exc}") from exc
    return artifact


def fit_anomaly_reference(sample_csv, output_dir, *, k=4, random_state=42,
                          contamination=0.01, n_estimators=300, fit_subset=50_000,
                          local_radius=2, n_init=20, max_iter=500):
    """Explicit NEW anomaly experiment. Inference must only load this artifact."""
    directory = Path(output_dir).expanduser().resolve()
    ensure_new(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Refusing to overwrite anomaly artifact: {directory}")
    if not isinstance(fit_subset, int) or fit_subset < 1:
        raise ValueError("fit_subset must be a positive integer.")
    sample_path = Path(sample_csv).expanduser().resolve()
    before = sha256_file(sample_path)
    sample = pd.read_csv(sample_path, float_precision="round_trip")
    missing = [name for name in ALL_FEATURES if name not in sample]
    if missing:
        raise ValueError(f"Anomaly sample is missing features: {missing}")
    X = validate_features(sample[ALL_FEATURES].to_numpy(dtype=np.float64), ALL_FEATURES)
    if not isinstance(k, int) or not 2 <= k <= len(X):
        raise ValueError("Anomaly KMeans requires 2 <= k <= sample count.")
    preprocessing = preprocessing_config(local_radius)
    if sha256_file(sample_path) != before:
        raise ValueError("Reference sample changed during reading.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter).fit(scaled)
    indices = np.random.default_rng(random_state).choice(len(X), size=min(fit_subset, len(X)), replace=False)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=random_state, n_jobs=-1).fit(scaled[indices])
    cr = km.transform(scaled).min(axis=1)
    ir = -iso.score_samples(scaled)
    meta = {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION,
        "model_id": uuid.uuid4().hex, "reference_id": uuid.uuid4().hex,
        "centroid_model_id": uuid.uuid4().hex, "isolation_model_id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(ALL_FEATURES), "preprocessing": preprocessing,
        "score_definitions": dict(SCORES), "percentile": dict(PERCENTILE),
        "composite": dict(COMPOSITE),
        "configuration": {"k": k, "random_state": random_state, "contamination": contamination,
                          "n_estimators": n_estimators, "fit_subset": fit_subset},
        "estimator_parameters": {"scaler": scaler.get_params(), "kmeans": km.get_params(),
                                 "isolation": iso.get_params()},
        "input_sample": {"path": str(sample_path), "sha256": before, "rows": len(X),
                         "feature_matrix_sha256": matrix_sha256(X),
                         "limitation": "File identity verified; CSV preprocessing lineage is caller-declared."},
        "fit_subset_rows": len(indices), "fit_subset_indices_sha256": matrix_sha256(indices),
        "environment": {"python": sys.version, "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
        "state_file": "state.pkl",
    }
    payload = {"contract": dict(meta), "scaler": scaler, "kmeans": km, "isolation": iso,
               "centroid_reference": cr, "isolation_reference": ir}
    directory.mkdir(parents=True, exist_ok=False)
    pending = directory / "state.pkl.pending"
    with pending.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    meta["state_sha256"] = sha256_file(pending)
    _validate_metadata(meta)
    manifest = directory / "manifest.json.pending"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Full load validation against pending state; no complete manifest exists yet.
    restored = _read_state(pending, meta, directory)
    if not np.array_equal(restored.centroid_reference, cr) or not np.array_equal(restored.isolation_reference, ir):
        raise ValueError("Serialized anomaly reference values differ from fitted reference scores.")
    actual = restored.score(X, ALL_FEATURES)
    np.testing.assert_array_equal(actual["cluster"], km.predict(scaled))
    np.testing.assert_array_equal(actual["centroid_distance"], cr)
    np.testing.assert_array_equal(actual["isolation_anomaly_score"], ir)
    pending.rename(directory / "state.pkl")
    manifest.rename(directory / "manifest.json")  # published last
    artifact = load_anomaly_reference(directory, expected_preprocessing=preprocessing)
    actual = artifact.score(X, ALL_FEATURES)
    np.testing.assert_array_equal(actual["centroid_distance"], cr)
    np.testing.assert_array_equal(actual["isolation_anomaly_score"], ir)
    return artifact
