import numpy as np
import pytest

from src.v2.anomaly import _percentile_rank
from src.v2.full_map import _percentile_reference


def test_sample_and_map_tied_percentiles_match():
    scores = np.array([1., 1., 2., 3.])
    expected = [.5, .5, .75, 1.]
    np.testing.assert_array_equal(_percentile_rank(scores), expected)
    np.testing.assert_array_equal(_percentile_reference(scores, scores), expected)

import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.v2.config import ALL_FEATURES


@pytest.mark.parametrize("reference,query,expected", [
    ([1,1,2,3], [3,1,2,1], [1,.5,.75,.5]),
    ([3,1,1,2], [-1,1,2,3,4], [0,.5,.75,1,1]),
    ([7], [6,7,8], [0,1,1]),
    ([1,2], [], []),
])
def test_ecdf_shapes_bounds_and_ties(reference, query, expected):
    from src.v2.anomaly_reference import empirical_percentile
    result = empirical_percentile(reference, query)
    assert result.shape == (len(query),)
    assert np.isfinite(result).all()
    assert ((0 <= result) & (result <= 1)).all()
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(_percentile_reference(reference, query), result)


@pytest.mark.parametrize("reference,query", [
    ([], [1]), ([np.nan], [1]), ([np.inf], [1]),
    ([[1,2]], [1]), (1, [1]), ([1], [[1]]), ([1], 1),
    ([1], [np.nan]), ([1], [np.inf]), (["bad"], [1]),
])
def test_ecdf_invalid_inputs(reference, query):
    from src.v2.anomaly_reference import empirical_percentile
    with pytest.raises(ValueError):
        empirical_percentile(reference, query)


@pytest.fixture
def sample_file(tmp_path):
    import pandas as pd
    X = np.random.default_rng(42).normal(size=(32,17))
    path = tmp_path / "sample.csv"
    pd.DataFrame(X, columns=ALL_FEATURES).assign(row=np.arange(32), col=0).to_csv(path,index=False)
    return path


@pytest.fixture
def anomaly_artifact(tmp_path, sample_file):
    from src.v2.anomaly_reference import fit_anomaly_reference
    with threadpool_limits(limits=1):
        return fit_anomaly_reference(sample_file, tmp_path/"reference",
                                     k=4, n_estimators=5, fit_subset=24)


def test_reference_roundtrip_scores_and_no_refit(anomaly_artifact, sample_file, monkeypatch):
    import pandas as pd
    from src.v2.anomaly_reference import load_anomaly_reference
    X = pd.read_csv(sample_file, float_precision="round_trip")[ALL_FEATURES].to_numpy()
    expected = anomaly_artifact.score(X, ALL_FEATURES)
    def forbidden(*a, **kw):
        raise AssertionError("Inference must not refit")
    for cls in (StandardScaler, KMeans, IsolationForest):
        monkeypatch.setattr(cls, "fit", forbidden)
    loaded = load_anomaly_reference(
        anomaly_artifact.directory, expected_model_id=anomaly_artifact.metadata["model_id"],
        expected_reference_id=anomaly_artifact.metadata["reference_id"])
    actual = loaded.score(X, ALL_FEATURES)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(actual["combined_anomaly_score"],
        .5*actual["centroid_anomaly_percentile"]+.5*actual["isolation_anomaly_percentile"])
    # Query population cannot establish a new reference.
    single = loaded.score(X[:1], ALL_FEATURES)
    for key in actual:
        np.testing.assert_allclose(single[key], actual[key][:1], atol=1e-12)
    assert loaded.metadata["input_sample"]["sha256"]
    assert loaded.metadata["percentile"]["tie_policy"] == "right-inclusive"


def test_reference_refuses_overwrite(anomaly_artifact, sample_file, monkeypatch):
    from src.v2.anomaly_reference import fit_anomaly_reference
    def forbidden(*a, **kw):
        raise AssertionError("Must reject before fitting")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    with pytest.raises(FileExistsError):
        fit_anomaly_reference(sample_file, anomaly_artifact.directory)


@pytest.mark.parametrize("field,value", [
    ("schema_version",999), ("feature_names",["radar"]),
    ("model_id","wrong"), ("reference_id","wrong"),
    ("score_definitions",{}), ("percentile",{}), ("composite",{}),
    ("input_sample",{}), ("environment",{}),
])
def test_reference_rejects_bad_manifest(anomaly_artifact, field, value):
    from src.v2.anomaly_reference import load_anomaly_reference
    p = anomaly_artifact.directory/"manifest.json"
    meta = json.loads(p.read_text())
    meta[field] = value
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_anomaly_reference(anomaly_artifact.directory)


@pytest.mark.parametrize("keyword", ["expected_model_id","expected_reference_id"])
def test_reference_rejects_wrong_expected_identity(anomaly_artifact, keyword):
    from src.v2.anomaly_reference import load_anomaly_reference
    with pytest.raises(ValueError, match="identity"):
        load_anomaly_reference(anomaly_artifact.directory, **{keyword:"wrong"})


def test_reference_missing_partial_corrupt(tmp_path, anomaly_artifact):
    from src.v2.anomaly_reference import load_anomaly_reference
    with pytest.raises(ValueError,match="manifest"):
        load_anomaly_reference(tmp_path/"missing")
    partial=tmp_path/"partial"
    partial.mkdir()
    (partial/"state.pkl.pending").write_bytes(b"incomplete")
    with pytest.raises(ValueError,match="manifest"):
        load_anomaly_reference(partial)
    with (anomaly_artifact.directory/"state.pkl").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError,match="hash"):
        load_anomaly_reference(anomaly_artifact.directory)


def test_sample_reuses_reference_and_protects_output(anomaly_artifact, sample_file, tmp_path, monkeypatch):
    import pandas as pd
    from src.v2.anomaly import run_anomaly_detection
    X=pd.read_csv(sample_file, float_precision="round_trip")[ALL_FEATURES].to_numpy()
    expected=anomaly_artifact.score(X, ALL_FEATURES)
    def forbidden(*a,**kw):
        raise AssertionError("Reusing reference must not fit")
    monkeypatch.setattr(IsolationForest,"fit",forbidden)
    out=tmp_path/"scored.csv"
    result=run_anomaly_detection(sample_file,out,reference_artifact=anomaly_artifact.directory,
                                n_estimators=5,fit_subset=24)
    scored=pd.read_csv(out, float_precision="round_trip").sort_values("row")
    for key,value in expected.items():
        np.testing.assert_allclose(scored[key],value,atol=1e-14)
    assert scored["anomaly_reference_id"].nunique()==1
    assert result["reference_id"]==anomaly_artifact.metadata["reference_id"]
    with pytest.raises(FileExistsError):
        run_anomaly_detection(sample_file,out,reference_artifact=anomaly_artifact.directory)


def test_map_missing_reference_fails_before_writes(tmp_path):
    from src.v2.full_map import build_full_v2_maps
    with pytest.raises(ValueError,match="anomaly reference"):
        build_full_v2_maps(tmp_path/"missing.csv",tmp_path/"maps")
    assert not (tmp_path/"maps").exists()
    assert not (tmp_path/"models").exists()


def test_root_passes_sample_reference_to_mapping(monkeypatch, tmp_path, sample_file):
    import main_v2 as entry
    from src.v2.config import V2Config
    monkeypatch.setattr(entry,"V2_PROCESSED",sample_file.parent)
    target=sample_file.parent/"venus_advanced_training_sample.csv"
    target.write_bytes(sample_file.read_bytes())
    monkeypatch.setattr(entry,"V2_ANALYSIS",tmp_path/"analysis")
    clustering=tmp_path/"clustering";clustering.mkdir()
    (clustering/"v2_consensus_labels.csv").write_text("consensus_agreement\n1\n")
    monkeypatch.setattr(entry,"V2_CLUSTERING",clustering)
    monkeypatch.setattr(entry,"_require_rasters",lambda:None)
    captured={}
    def sample(*a,**kw):
        captured["radius"]=kw["local_radius"]
        return {"reference_artifact":"saved-reference"}
    def maps(*a,**kw):
        assert kw["anomaly_reference"]=="saved-reference"
        return {}
    monkeypatch.setattr(entry,"run_anomaly_detection",sample)
    monkeypatch.setattr(entry,"build_full_v2_maps",maps)
    for name in ("fit_distance_uncertainty","save_confidence_histogram","save_anomaly_distribution",
                 "save_consensus_agreement","save_top_anomalies","write_json","write_stage_report"):
        monkeypatch.setattr(entry,name,lambda *a,**kw: {})
    monkeypatch.setattr(entry,"discover_and_run_associations",lambda *a,**kw: [])
    monkeypatch.setattr(entry,"K4_RASTER",tmp_path/"absent.tif")
    entry.stage3(V2Config(local_radius=3))
    assert captured["radius"]==3


def test_raw_score_shape_validation(anomaly_artifact):
    with pytest.raises(ValueError,match="same shape"):
        anomaly_artifact.score_raw([1,2],[1])
    result=anomaly_artifact.score_raw([1,1,2,3],[1,1,2,3])
    restored=anomaly_artifact.score_raw([1,1,2,3],[1,1,2,3])
    for key in result:
        np.testing.assert_array_equal(result[key],restored[key])


@pytest.mark.parametrize("field,empty", [("centroid_reference",True),("centroid_reference",False),("isolation_reference",False)])
def test_bad_serialized_reference_never_publishes_manifest(tmp_path, sample_file, monkeypatch, field, empty):
    import src.v2.anomaly_reference as refs
    original=refs.pickle.dump
    def damaged(payload,stream,**kwargs):
        payload=dict(payload)
        payload[field] = np.array([]) if empty else np.zeros_like(payload[field])
        return original(payload,stream,**kwargs)
    monkeypatch.setattr(refs.pickle,"dump",damaged)
    directory=tmp_path/"damaged"
    with threadpool_limits(limits=1), pytest.raises(ValueError):
        refs.fit_anomaly_reference(sample_file,directory,n_estimators=3)
    assert not (directory/"manifest.json").exists()
    with pytest.raises(ValueError,match="manifest"):
        refs.load_anomaly_reference(directory)


def test_write_interruption_leaves_no_complete_reference(tmp_path,sample_file,monkeypatch):
    import src.v2.anomaly_reference as refs
    def fail(*a,**kw):
        raise OSError("interrupted")
    monkeypatch.setattr(refs.pickle,"dump",fail)
    directory=tmp_path/"interrupted"
    with threadpool_limits(limits=1), pytest.raises(OSError,match="interrupted"):
        refs.fit_anomaly_reference(sample_file,directory,n_estimators=3)
    assert not (directory/"manifest.json").exists()


def test_reference_rejects_wrong_sample_or_preprocessing(anomaly_artifact,sample_file):
    import pandas as pd
    from src.v2.anomaly_reference import load_anomaly_reference
    X=pd.read_csv(sample_file, float_precision="round_trip")[ALL_FEATURES].to_numpy()
    with pytest.raises(ValueError,match="identity"):
        anomaly_artifact.validate_sample(X[::-1],sample_file)
    with pytest.raises(ValueError,match="preprocessing"):
        load_anomaly_reference(anomaly_artifact.directory,expected_preprocessing={})
    with pytest.raises(ValueError,match="configuration"):
        anomaly_artifact.validate_configuration(k=3,random_state=42,contamination=.01,
                                                n_estimators=5,fit_subset=24)


def test_new_sample_scoring_saves_identifiable_reference(tmp_path,sample_file):
    from src.v2.anomaly import run_anomaly_detection
    from src.v2.anomaly_reference import load_anomaly_reference
    with threadpool_limits(limits=1):
        result=run_anomaly_detection(sample_file,tmp_path/"new_scores.csv",
                                     n_estimators=3,fit_subset=20)
    saved=load_anomaly_reference(result["reference_artifact"])
    assert saved.metadata["reference_id"]==result["reference_id"]
    assert saved.metadata["configuration"]["n_estimators"]==3
