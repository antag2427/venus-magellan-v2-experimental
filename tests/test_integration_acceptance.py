"""Acceptance gate: synthetic rasters/models only; every output uses pytest tmp dirs."""
from contextlib import ExitStack
from functools import wraps
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.v2.config import V2Config, ALL_FEATURES
from src.v2.model_artifacts import sha256_file, load_production_kmeans
from src.v2.anomaly_reference import fit_anomaly_reference, load_anomaly_reference
from src.v2.uncertainty import fit_distance_uncertainty


def hashes(directory):
    return {str(p.relative_to(directory)):sha256_file(p) for p in directory.rglob("*") if p.is_file()}


@pytest.fixture
def synthetic_world(tmp_path,monkeypatch):
    import main_v2 as entry
    from src.v2 import raster_features as rf
    root=tmp_path/"TEST_ONLY_SYNTHETIC_WORLD"
    raw=root/"data/raw";raw.mkdir(parents=True)
    crs="+proj=eqc +lat_ts=0 +lon_0=0 +R=6051000 +units=m +no_defs"
    rng=np.random.default_rng(42)
    y,x=np.mgrid[:16,:16]
    radar=(20+3*x+5*y+rng.integers(0,8,(16,16))).astype("uint8")
    radar[0,:]=0;radar[5,5]=0
    paths={}
    def write(name,array,transform,nodata):
        path=raw/(name+".tif")
        with rasterio.open(path,"w",driver="GTiff",count=1,width=array.shape[1],height=array.shape[0],
                           dtype=array.dtype.name,crs=crs,transform=transform,nodata=nodata,
                           tiled=True,blockxsize=16,blockysize=16) as dst:
            dst.write(array,1)
        paths[name]=path
    write("radar",radar,from_origin(0,16,1,1),0)
    y,x=np.mgrid[:8,:8]
    for i,name in enumerate(["height","emissivity","slope","reflectivity"]):
        dtype="int16" if name in ("height","emissivity") else "uint8"
        data=(20+7*i+2*x+3*y+x*y+rng.integers(0,8,(8,8))).astype(dtype)
        nodata=-32768 if dtype=="int16" else 0
        data[6:,6:]=nodata
        write(name,data,from_origin(0,16,2,2),nodata)
    monkeypatch.setattr(rf,"RADAR_FILE",paths["radar"])
    monkeypatch.setattr(rf,"BASE_PATHS",paths)
    monkeypatch.setattr(entry,"BASE_PATHS",paths)
    monkeypatch.setattr(entry,"PROJECT_ROOT",root)
    monkeypatch.setattr(entry,"V2_PROCESSED",root/"data/processed/v2")
    monkeypatch.setattr(entry,"V2_CLUSTERING",root/"outputs/v2/clustering")
    monkeypatch.setattr(entry,"V2_ANALYSIS",root/"outputs/v2/analysis")
    monkeypatch.setattr(entry,"V2_SPATIAL",root/"outputs/v2/spatial")
    monkeypatch.setattr(entry,"V2_FIGURES",root/"figures/v2")
    monkeypatch.setattr(entry,"CONFIG_PATH",root/"unused_default.json")
    monkeypatch.setattr(entry,"K4_RASTER",root/"ABSENT_V1_RASTER.tif")
    cfg=V2Config(sample_size=64,block_size=8,k=4,consensus_subset=48,
                 anomaly_subset=32,isolation_estimators=7,kmeans_n_init=3,kmeans_max_iter=37,
                 gmm_n_init=2,gmm_max_iter=80,birch_threshold=.7,minibatch_batch_size=8,
                 object_downsample_max_dim=16,output_prefix="gateA")
    config=root/"synthetic_config.json"
    cfg.save_json(config)
    return entry,rf,root,paths,cfg,config


def invoke(monkeypatch,entry,stage,config):
    monkeypatch.setattr(sys,"argv",["main_v2.py",stage,"--config",str(config)])
    with threadpool_limits(limits=1):
        entry.main()


def spy_initializers(monkeypatch):
    seen={cls.__name__:[] for cls in (KMeans,MiniBatchKMeans,GaussianMixture,Birch,IsolationForest)}
    seen["__calls__"]=[]
    for cls in (KMeans,MiniBatchKMeans,GaussianMixture,Birch,IsolationForest):
        original=cls.__init__
        def factory(original,cls):
            @wraps(original)
            def spy(self,*args,**kwargs):
                caller=inspect.currentframe().f_back.f_globals["__name__"]
                seen["__calls__"].append({"class":cls.__name__,"caller":caller,"kwargs":dict(kwargs)})
                seen[cls.__name__].append(dict(kwargs))
                original(self,*args,**kwargs)
            return spy
        monkeypatch.setattr(cls,"__init__",factory(original,cls))
    return seen


def forbid_fitting(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Saved-artifact inference attempted fitting")
    for cls in (StandardScaler,KMeans,IsolationForest):
        monkeypatch.setattr(cls,"fit",forbidden)
    monkeypatch.setattr(StandardScaler,"fit_transform",forbidden)
    monkeypatch.setattr(KMeans,"fit_transform",forbidden)
    monkeypatch.setattr(KMeans,"fit_predict",forbidden)


def test_uncertainty_real_estimator_contract(tmp_path,monkeypatch):
    import src.v2.uncertainty as uncertainty
    rng=np.random.default_rng(7)
    X=rng.normal(size=(48,17))
    table=pd.DataFrame(X,columns=ALL_FEATURES).assign(row=np.arange(48),col=13)
    sample=tmp_path/"sample.csv";table.to_csv(sample,index=False,float_format="%.17g")
    seen=spy_initializers(monkeypatch)
    def wrong_path(*args,**kwargs):
        raise AssertionError("Exploratory uncertainty must not load a production inference artifact")
    import src.v2.model_artifacts as models
    monkeypatch.setattr(models,"load_production_kmeans",wrong_path)
    with threadpool_limits(limits=1):
        result=uncertainty.fit_distance_uncertainty(sample,tmp_path/"uncertainty.csv",
                                                     k=4,n_init=3,max_iter=37)
    assert seen["KMeans"]==[dict(n_clusters=4,random_state=42,n_init=3,max_iter=37)]
    assert result["estimator_parameters"]["n_init"]==3
    assert result["estimator_parameters"]["max_iter"]==37
    out=pd.read_csv(tmp_path/"uncertainty.csv",float_precision="round_trip")
    expected=["row","col","cluster","nearest_centroid_distance","second_centroid_distance",
              "distance_margin","confidence","entropy"]+[f"p_cluster_{i}" for i in range(4)]
    assert out.columns.tolist()==expected and out.shape==(48,len(expected))
    np.testing.assert_array_equal(out[["row","col"]],table[["row","col"]])
    assert np.isfinite(out.to_numpy()).all()
    assert out.cluster.between(0,3).all()
    assert out.confidence.between(0,1).all() and out.entropy.between(0,1).all()
    np.testing.assert_allclose(out.confidence,1-out.entropy,rtol=0,atol=1e-15)
    assert (out.second_centroid_distance>=out.nearest_centroid_distance).all()
    np.testing.assert_allclose(out.distance_margin,out.second_centroid_distance-out.nearest_centroid_distance,
                               rtol=0,atol=1e-15)
    np.testing.assert_allclose(out[[f"p_cluster_{i}" for i in range(4)]].sum(axis=1),1,rtol=0,atol=1e-15)


@pytest.mark.parametrize("mode",["missing_file","missing_feature","nan","infinity"])
def test_uncertainty_invalid_input_precedes_fitting(tmp_path,monkeypatch,mode):
    path=tmp_path/"input.csv"
    table=pd.DataFrame(np.ones((8,17)),columns=ALL_FEATURES)
    if mode=="missing_feature":table=table.drop(columns="height")
    elif mode=="nan":table.loc[0,"height"]=np.nan
    elif mode=="infinity":table.loc[0,"height"]=np.inf
    if mode!="missing_file":table.to_csv(path,index=False)
    forbid_fitting(monkeypatch)
    with pytest.raises((ValueError,FileNotFoundError)):
        fit_distance_uncertainty(path,tmp_path/"output.csv")
    assert not (tmp_path/"output.csv").exists()


def test_synthetic_cli_all_stages_and_saved_inference(synthetic_world,monkeypatch):
    entry,rf,root,paths,cfg,config=synthetic_world
    source_before={str(p):sha256_file(p) for p in paths.values()}
    seen=spy_initializers(monkeypatch)
    invoke(monkeypatch,entry,"all",config)
    processed,clustering,analysis,_,figures=entry._layout(cfg)
    manifests=[json.loads((d/f"stage{i}_run.json").read_text())
               for i,d in enumerate([processed,clustering,analysis],1)]
    assert all(m["status"]=="complete" for m in manifests)
    assert len({m["run_id"] for m in manifests})==3
    assert not list(root.rglob("*.pending"))
    assert not list(root.rglob("*.publishing"))
    for manifest in manifests:
        assert manifest["configuration"]==cfg.to_dict()
        for record in manifest["inputs"]:
            assert record["sha256"]==sha256_file(record["path"])
    project_calls=[call for call in seen["__calls__"] if call["class"]=="KMeans" and call["caller"].startswith("src.v2.")]
    from collections import Counter
    assert Counter(call["caller"] for call in project_calls)=={
        "src.v2.consensus":1,"src.v2.discovery":10,"src.v2.uncertainty":1,
        "src.v2.anomaly_reference":1,"src.v2.model_artifacts":1}
    assert all(call["kwargs"]["n_init"]==3 and call["kwargs"]["max_iter"]==37 for call in project_calls)
    internal_calls=[call for call in seen["__calls__"] if call["class"]=="KMeans" and not call["caller"].startswith("src.v2.")]
    assert len(internal_calls)==cfg.gmm_n_init
    assert all(call["caller"].startswith("sklearn.mixture.") and call["kwargs"]["n_init"]==1 for call in internal_calls)
    assert seen["MiniBatchKMeans"]==[dict(n_clusters=4,random_state=42,n_init=10,batch_size=8,max_iter=37)]
    assert seen["GaussianMixture"]==[dict(n_components=4,covariance_type="full",random_state=42,n_init=2,max_iter=80)]
    assert seen["Birch"]==[dict(n_clusters=4,threshold=.7)]
    assert seen["IsolationForest"]==[dict(n_estimators=7,contamination=.01,random_state=42,n_jobs=-1)]
    assert manifests[1]["results"]["consensus"]["estimator_parameters"]["kmeans"]["max_iter"]==37
    assert manifests[2]["results"]["uncertainty"]["estimator_parameters"]["n_init"]==3
    sample=processed/"venus_advanced_training_sample.csv"
    table=pd.read_csv(sample,float_precision="round_trip")
    assert len(table)==64 and np.isfinite(table[ALL_FEATURES].to_numpy()).all()
    assert len(ALL_FEATURES)==17
    assert (clustering/"gateA_consensus_labels.csv").is_file()
    assert all((figures/f"gateA_{name}.png").is_file() for name in
               ("confidence_distribution","anomaly_distribution","consensus_agreement","top_anomalies"))
    assert manifests[2]["results"]["geology_associations"]==[]
    assert manifests[2]["results"]["coarse_objects"] is None
    summary=manifests[2]["results"]
    production=load_production_kmeans(summary["full_map"]["model_artifact"])
    anomaly=load_anomaly_reference(summary["anomaly"]["reference_artifact"])
    assert production.model_id==summary["full_map"]["model_id"]
    assert anomaly.metadata["model_id"]==summary["anomaly"]["model_id"]
    assert production.model_id!=anomaly.metadata["model_id"]
    assert anomaly.metadata["reference_id"]==summary["anomaly"]["reference_id"]
    assert production.model.get_params()["n_init"]==anomaly.kmeans.get_params()["n_init"]==3
    assert production.model.get_params()["max_iter"]==anomaly.kmeans.get_params()["max_iter"]==37
    association=json.loads(Path(summary["full_map"]["model_association"]).read_text())
    assert association["anomaly_reference"]["reference_id"]==anomaly.metadata["reference_id"]
    assert association["model_id"]==production.model_id
    # Native labels, alignment, and NoData checks on the entire tiny synthetic grid.
    ctx,radar,sources=rf.aligned_rasters()
    with ctx:
        arrays,masks=rf._read_aligned_window(sources,rasterio.windows.Window(0,0,16,16))
        valid=np.logical_and.reduce([masks[n] for n in rf.BASE_FEATURES])
        expected_transform=radar.transform
        expected_crs=radar.crs
    with rasterio.open(summary["full_map"]["cluster_raster"]) as dst:
        labels=dst.read(1)
        assert dst.shape==(16,16) and dst.transform==expected_transform and dst.crs==expected_crs
        assert dst.nodata==255
        np.testing.assert_array_equal(labels!=255,valid)
        np.testing.assert_array_equal(labels[table.row,table.col],
                                      production.predict(table[ALL_FEATURES].to_numpy(),ALL_FEATURES))
    restored_scores=anomaly.score(table[ALL_FEATURES].to_numpy(),ALL_FEATURES)
    saved_scores=pd.read_csv(summary["anomaly"]["output"],float_precision="round_trip").set_index(["row","col"])
    saved_scores=saved_scores.loc[pd.MultiIndex.from_frame(table[["row","col"]])]
    for key,expected in restored_scores.items():
        np.testing.assert_array_equal(saved_scores[key].to_numpy(),expected)
    with rasterio.open(summary["full_map"]["anomaly_raster"]) as dst:
        scores=dst.read(1)
        assert dst.nodata==-9999
        np.testing.assert_array_equal(scores==-9999,~valid)
        assert np.isfinite(scores[valid]).all()
        assert ((scores[valid]>=0)&(scores[valid]<=1)).all()
        assert dst.tags()["ANOMALY_REFERENCE_ID"]==anomaly.metadata["reference_id"]
        # Compare native sample locations; only float32 raster storage gets tolerance.
        np.testing.assert_allclose(scores[table.row,table.col],
            restored_scores["combined_anomaly_score"],rtol=0,atol=3e-8)
    with rasterio.open(summary["full_map"]["confidence_raster"]) as dst:
        conf=dst.read(1)
        np.testing.assert_array_equal(conf==-9999,~valid)
        assert ((conf[valid]>=0)&(conf[valid]<=1)).all()
    # Same algorithm under a second --config namespace gets a different run ID.
    cfg_b=V2Config(**dict(cfg.to_dict(),output_prefix="gateB"))
    config_b=root/"second_config.json";cfg_b.save_json(config_b)
    invoke(monkeypatch,entry,"features",config_b)
    second=json.loads((entry._layout(cfg_b)[0]/"stage1_run.json").read_text())
    assert second["run_id"]!=manifests[0]["run_id"]
    assert second["configuration"]["output_prefix"]=="gateB"
    assert {str(p):sha256_file(p) for p in paths.values()}==source_before
    protected=hashes(root)
    from src.v2 import full_map as maps
    with monkeypatch.context() as guard, threadpool_limits(limits=1):
        forbid_fitting(guard)
        # CLI reruns must reject their existing destinations before any fitting.
        for stage in ("features","discover","interpret"):
            with pytest.raises(FileExistsError):invoke(guard,entry,stage,config)
        # Real inference from saved synthetic artifacts, with every fit forbidden.
        result=maps.predict_full_v2_maps(production.directory,sample,root/"saved_inference_only",
                  k=4,block_size=8,local_radius=2,isolation_estimators=7,isolation_fit_subset=32,
                  anomaly_reference=anomaly.directory,output_prefix="reload",n_init=3,max_iter=37)
        with rasterio.open(result["cluster_raster"]) as dst:np.testing.assert_array_equal(dst.read(1),labels)
        with rasterio.open(result["anomaly_raster"]) as dst:np.testing.assert_array_equal(dst.read(1),scores)
        with rasterio.open(result["confidence_raster"]) as dst:np.testing.assert_array_equal(dst.read(1),conf)
    for rel,digest in protected.items():assert sha256_file(root/rel)==digest


@pytest.fixture(scope="module")
def synthetic_saved_states(tmp_path_factory):
    from src.v2.full_map import fit_full_map_kmeans
    root=tmp_path_factory.mktemp("TEST_ONLY_saved_states")
    sample=root/"synthetic_sample.csv"
    pd.DataFrame(np.random.default_rng(42).normal(size=(32,17)),columns=ALL_FEATURES).to_csv(
        sample,index=False,float_format="%.17g")
    with threadpool_limits(limits=1):
        model=fit_full_map_kmeans(sample,root/"synthetic_classifier",n_init=3,max_iter=37)
        anomaly=fit_anomaly_reference(sample,root/"synthetic_anomaly",n_init=3,max_iter=37,
                                      n_estimators=3,fit_subset=16)
    return root,sample,model,anomaly


@pytest.mark.parametrize("mode",["stale","corrupt","incomplete","wrong_role"])
def test_bad_artifact_rejected_before_mapping(synthetic_saved_states,tmp_path,monkeypatch,mode):
    from src.v2 import full_map as maps
    _,sample,model,anomaly=synthetic_saved_states
    protected={**{str(p):sha256_file(p) for p in model.directory.iterdir()},
               **{str(p):sha256_file(p) for p in anomaly.directory.iterdir()}}
    bad=tmp_path/"bad"
    if mode=="incomplete":
        bad.mkdir();(bad/"model.pkl.pending").write_bytes(b"incomplete")
    elif mode=="wrong_role":
        bad=anomaly.directory
    else:
        shutil.copytree(model.directory,bad)
        if mode=="stale":
            manifest=json.loads((bad/"manifest.json").read_text());manifest["schema_version"]=999
            (bad/"manifest.json").write_text(json.dumps(manifest))
        else:
            with (bad/"model.pkl").open("ab") as stream:stream.write(b"corrupt")
    def forbidden(*a,**kw):raise AssertionError("Raster mapping reached invalid artifact")
    monkeypatch.setattr(maps,"aligned_rasters",forbidden)
    forbid_fitting(monkeypatch)
    with pytest.raises(ValueError):
        maps.predict_full_v2_maps(bad,sample,tmp_path/"maps",anomaly_reference=anomaly.directory)
    assert not (tmp_path/"maps").exists()
    for path,digest in protected.items():assert sha256_file(path)==digest


@pytest.mark.parametrize("mode",["unknown","invalid"])
def test_cli_bad_config_precedes_stages(synthetic_world,monkeypatch,mode):
    entry,_,root,_,cfg,_=synthetic_world
    payload=cfg.to_dict()
    if mode=="unknown":payload["kmeans_typo"]=3
    else:payload["kmeans_max_iter"]=0
    config=root/"bad_config.json";config.write_text(json.dumps(payload))
    def forbidden(*a,**kw):raise AssertionError("A stage ran for invalid configuration")
    for name in ("stage1","stage2","stage3"):monkeypatch.setattr(entry,name,forbidden)
    with pytest.raises(ValueError):invoke(monkeypatch,entry,"all",config)
    assert not list(root.rglob("stage*_run.json"))


def test_interrupted_stage_publication_has_no_completion(synthetic_world,monkeypatch):
    entry,rf,root,_,cfg,config=synthetic_world
    original=pd.DataFrame.to_csv
    def interrupted(self,path,*args,**kwargs):
        if str(getattr(path,"name",path)).endswith("venus_advanced_training_sample.csv.pending"):
            path.write("row,col,partial\n")
            raise OSError("simulated publication interruption")
        return original(self,path,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,"to_csv",interrupted)
    with pytest.raises(OSError,match="interruption"):invoke(monkeypatch,entry,"features",config)
    processed=entry._layout(cfg)[0]
    assert not (processed/"venus_advanced_training_sample.csv").exists()
    assert (processed/"venus_advanced_training_sample.csv.pending").exists()
    assert not (processed/"stage1_run.json").exists()
    pending=json.loads((processed/"stage1_run.json.pending").read_text())
    assert pending["status"]=="running"
    def forbidden(*a,**kw):raise AssertionError("Raster processing reached existing output")
    monkeypatch.setattr(rf,"aligned_rasters",forbidden)
    with pytest.raises(FileExistsError):invoke(monkeypatch,entry,"features",config)


def test_corrupt_sample_does_not_publish_discovery_completion(synthetic_world,monkeypatch):
    entry,_,_,_,cfg,config=synthetic_world
    processed,clustering,*_=entry._layout(cfg)
    processed.mkdir(parents=True)
    (processed/"venus_advanced_training_sample.csv").write_text("row,col,radar\n1,2,3\n")
    def forbidden(*a,**kw):raise AssertionError("Discovery fitting reached corrupt sample")
    monkeypatch.setattr(entry,"run_consensus",forbidden)
    with pytest.raises(ValueError,match="missing features"):invoke(monkeypatch,entry,"discover",config)
    assert not (clustering/"stage2_run.json").exists()
