from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.v2.config import ALL_FEATURES
from src.v2.model_artifacts import (
    fit_production_kmeans, load_production_kmeans, sha256_file,
)


@pytest.fixture
def tiny_features():
    rng = np.random.default_rng(42)
    return rng.normal(size=(48, 17)) + np.repeat(np.arange(4) * 6, 12)[:, None]


@pytest.fixture
def artifact(tmp_path, tiny_features):
    with threadpool_limits(limits=1):
        return fit_production_kmeans(tiny_features, ALL_FEATURES, tmp_path / "model")


def test_artifact_fit_save_load_native_predictions(artifact, tiny_features):
    loaded = load_production_kmeans(artifact.directory)
    np.testing.assert_array_equal(
        artifact.predict(tiny_features, ALL_FEATURES),
        loaded.predict(tiny_features, ALL_FEATURES),
    )
    assert loaded.model_id == artifact.model_id
    assert loaded.metadata["feature_names"] == ALL_FEATURES
    assert loaded.metadata["feature_count"] == 17
    assert loaded.metadata["kmeans_parameters"]["n_init"] == 20
    assert loaded.metadata["kmeans_parameters"]["max_iter"] == 500
    assert loaded.metadata["preprocessing"]["local_radius"] == 2
    assert loaded.metadata["preprocessing"]["halo_radius"] == 3
    assert loaded.metadata["preprocessing"]["companion_dtype"] == "float64"
    assert loaded.metadata["model_file"] == "model.pkl"
    assert loaded.metadata["model_sha256"] == sha256_file(artifact.directory / "model.pkl")
    assert loaded.provenance["status"] == "in_memory_only"
    assert loaded.metadata["environment"]["sklearn"]


@pytest.mark.parametrize("mode", ["reordered", "missing", "nonfinite"])
def test_artifact_rejects_input_schema_and_nonfinite(artifact, tiny_features, mode):
    names = ALL_FEATURES.copy()
    values = tiny_features.copy()
    if mode == "reordered":
        names[0], names[1] = names[1], names[0]
    elif mode == "missing":
        names = names[:-1]
        values = values[:, :-1]
    else:
        values[0, 0] = np.nan
    with pytest.raises(ValueError):
        artifact.predict(values, names)


@pytest.mark.parametrize("mode", ["version", "feature_order", "feature_count", "kind", "missing_key", "malformed"])
def test_artifact_rejects_bad_metadata(artifact, mode):
    path = artifact.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if mode == "version":
        manifest["schema_version"] = 999
    elif mode == "feature_order":
        manifest["feature_names"][0], manifest["feature_names"][1] = manifest["feature_names"][1], manifest["feature_names"][0]
    elif mode == "feature_count":
        manifest["feature_count"] = 5
    elif mode == "kind":
        manifest["artifact_type"] = "V1"
    elif mode == "missing_key":
        del manifest["preprocessing"]
    else:
        path.write_text("{not json")
    if mode != "malformed":
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_production_kmeans(artifact.directory)


def test_artifact_refuses_overwrite_before_fitting(artifact, tiny_features, monkeypatch):
    before = {p.name: p.read_bytes() for p in artifact.directory.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing destination must be rejected before fitting")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    with pytest.raises(FileExistsError):
        fit_production_kmeans(tiny_features, ALL_FEATURES, artifact.directory)
    assert {p.name: p.read_bytes() for p in artifact.directory.iterdir()} == before


@pytest.mark.parametrize("component", ["scaler", "model"])
def test_artifact_detects_incompatible_fitted_dimensions(artifact, component):
    model_path = artifact.directory / "model.pkl"
    with model_path.open("rb") as f:
        objects = pickle.load(f)
    objects[component].n_features_in_ = 16
    with model_path.open("wb") as f:
        pickle.dump(objects, f)
    manifest_path = artifact.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = sha256_file(model_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dimension"):
        load_production_kmeans(artifact.directory)


def test_artifact_inference_never_refits(artifact, tiny_features, monkeypatch):
    expected = artifact.predict(tiny_features, ALL_FEATURES)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference attempted model fitting")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(KMeans, "fit", forbidden)
    loaded = load_production_kmeans(artifact.directory)
    np.testing.assert_array_equal(loaded.predict(tiny_features, ALL_FEATURES), expected)


def test_artifact_missing_or_incomplete(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        load_production_kmeans(tmp_path / "missing")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "model.pkl.pending").write_bytes(b"partial")
    with pytest.raises(ValueError, match="manifest"):
        load_production_kmeans(incomplete)


def test_artifact_write_failure_has_no_complete_manifest(tmp_path, tiny_features, monkeypatch):
    import src.v2.model_artifacts as artifacts
    def fail(*args, **kwargs):
        raise OSError("simulated serialization interruption")
    monkeypatch.setattr(artifacts.pickle, "dump", fail)
    directory = tmp_path / "interrupted"
    with threadpool_limits(limits=1), pytest.raises(OSError):
        fit_production_kmeans(tiny_features, ALL_FEATURES, directory)
    assert not (directory / "manifest.json").exists()
    with pytest.raises(ValueError):
        load_production_kmeans(directory)


def test_artifact_detects_corrupt_state(artifact):
    with (artifact.directory / "model.pkl").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_production_kmeans(artifact.directory)


def test_artifact_rejects_preprocessing_mismatch(artifact):
    with pytest.raises(ValueError, match="preprocessing"):
        load_production_kmeans(artifact.directory, expected_preprocessing={"local_radius": 3})


def test_full_map_fit_wrapper_records_sample_provenance(tmp_path, tiny_features):
    from src.v2.full_map import fit_full_map_kmeans
    sample = tmp_path / "sample.csv"
    pd.DataFrame(tiny_features, columns=ALL_FEATURES).assign(row=np.arange(48), col=0).to_csv(sample, index=False)
    with threadpool_limits(limits=1):
        artifact = fit_full_map_kmeans(sample, tmp_path / "production")
    assert artifact.metadata["input_sample"]["sha256"] == sha256_file(sample)
    assert artifact.provenance["status"] == "verified"
    sample.unlink()
    loaded = load_production_kmeans(artifact.directory)
    assert loaded.provenance["status"] == "unavailable"
    assert "limitation" in loaded.provenance


def test_full_map_prediction_preflight_rejects_missing_artifact(tmp_path):
    from src.v2.full_map import predict_full_v2_maps
    with pytest.raises(ValueError, match="manifest"):
        predict_full_v2_maps(tmp_path / "missing", tmp_path / "sample.csv", tmp_path / "maps")
    assert not (tmp_path / "maps").exists()


def test_full_map_prediction_has_no_production_fit_calls():
    import ast
    import inspect
    from src.v2.full_map import predict_full_v2_maps
    tree = ast.parse(inspect.getsource(predict_full_v2_maps))
    # Prediction must not fit production OR anomaly estimators.
    forbidden = {"fit_transform", "fit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden


@pytest.mark.parametrize("mode", ["empty", "bad_matrix_hash", "bad_source", "missing_python"])
def test_artifact_rejects_malformed_provenance(artifact, mode):
    path = artifact.directory / "manifest.json"
    data = json.loads(path.read_text())
    if mode == "empty":
        data["input_sample"] = {}
    elif mode == "bad_matrix_hash":
        data["input_sample"]["feature_matrix_sha256"] = "not-a-hash"
    elif mode == "bad_source":
        data["input_sample"]["path"] = 123
    else:
        del data["environment"]["python"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_production_kmeans(artifact.directory)


def test_artifact_rejects_incomplete_predictor_state(artifact):
    state = artifact.directory / "model.pkl"
    with state.open("rb") as stream:
        objects = pickle.load(stream)
    del objects["model"]._n_threads
    with state.open("wb") as stream:
        pickle.dump(objects, stream)
    path = artifact.directory / "manifest.json"
    data = json.loads(path.read_text())
    data["model_sha256"] = sha256_file(state)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="predict"):
        load_production_kmeans(artifact.directory)


def test_full_map_mocked_raster_path_uses_saved_model(tmp_path, tiny_features, monkeypatch):
    from contextlib import ExitStack
    from types import SimpleNamespace
    import src.v2.full_map as maps

    sample = tmp_path / "sample.csv"
    pd.DataFrame(tiny_features, columns=ALL_FEATURES).to_csv(sample, index=False)
    with threadpool_limits(limits=1):
        fitted = maps.fit_full_map_kmeans(sample, tmp_path / "production")
    values = pd.read_csv(sample, float_precision="round_trip")[ALL_FEATURES].to_numpy()
    expected = fitted.predict(values[:4], ALL_FEATURES).reshape(2, 2)
    expected[1, 1] = 255
    cube = values[:4].reshape(2, 2, 17)
    arrays = {name: cube[..., i] for i, name in enumerate(ALL_FEATURES[:5])}
    masks = {name: np.ones((2, 2), bool) for name in arrays}
    masks["height"][1, 1] = False
    derived = {name: cube[..., i] for i, name in enumerate(ALL_FEATURES) if i >= 5}
    radar = SimpleNamespace(width=2, height=2, res=(2025, 2025), profile={})
    monkeypatch.setattr(maps, "aligned_rasters", lambda: (ExitStack(), radar, {}))
    monkeypatch.setattr(maps, "_read_aligned_window", lambda *args: (arrays, masks))
    monkeypatch.setattr(maps, "derive_local_features", lambda *args: derived)

    from src.v2.anomaly_reference import fit_anomaly_reference
    from sklearn.ensemble import IsolationForest
    with threadpool_limits(limits=1):
        anomaly = fit_anomaly_reference(sample, tmp_path / "anomaly", n_estimators=5)
    expected_anomaly = anomaly.score(values[:3], ALL_FEATURES)["combined_anomaly_score"]

    destinations = {}
    class MemoryDestination:
        def __init__(self):
            self.tags = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update_tags(self, **kwargs):
            self.tags.update(kwargs)
        def write(self, array, band, window):
            self.values = array.copy()

    def memory_open(path, *args, **kwargs):
        destination = MemoryDestination()
        destinations[path.name] = destination
        return destination

    monkeypatch.setattr(maps.rasterio, "open", memory_open)
    def forbidden(*args, **kwargs):
        raise AssertionError("Raster prediction refitted production state")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(KMeans, "fit", forbidden)
    monkeypatch.setattr(IsolationForest, "fit", forbidden)
    result = maps.predict_full_v2_maps(fitted.directory, sample, tmp_path / "maps",
                                      anomaly_reference=anomaly.directory, isolation_estimators=5)
    np.testing.assert_array_equal(destinations["venus_v2_clusters_k4.tif"].values, expected)
    assert result["model_id"] == fitted.model_id
    assert all(x.tags["MODEL_ID"] == fitted.model_id for x in destinations.values())
    metadata = json.loads((tmp_path / "maps/v2_production_model.json").read_text())
    assert metadata["model_id"] == fitted.model_id
    assert metadata["anomaly_reference"]["role"].startswith("separate")
    assert metadata["anomaly_reference"]["reference_id"] == anomaly.metadata["reference_id"]
    assert metadata["anomaly_reference"]["model_id"] != fitted.model_id
    dst = destinations["venus_v2_anomaly.tif"]
    np.testing.assert_allclose(dst.values.ravel()[:3], expected_anomaly, atol=3e-8)
    assert dst.values[1,1] == -9999.
    assert dst.tags["ANOMALY_REFERENCE_ID"] == anomaly.metadata["reference_id"]
