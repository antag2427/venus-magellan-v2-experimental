from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from src.v2.config import V2Config,ALL_FEATURES


@pytest.fixture
def sample(tmp_path):
    p=tmp_path/"sample.csv"
    pd.DataFrame(np.random.default_rng(42).normal(size=(16,17)),columns=ALL_FEATURES).assign(row=np.arange(16),col=0).to_csv(p,index=False,float_format="%.17g")
    return p


@pytest.mark.parametrize("option,value",[("kmeans_n_init",0),("kmeans_max_iter",-1),("gmm_n_init",0),
    ("birch_threshold",0),("sample_min_candidate_factor",2.),("sample_batch_factor",2.),
    ("output_prefix","../oops"),("block_size",0),("k",1),("local_radius",-1)])
def test_invalid_or_unsupported_configuration(option,value):
    with pytest.raises(ValueError,match=option):
        V2Config(**{option:value})


def test_unknown_configuration_not_silently_ignored(tmp_path):
    p=tmp_path/"bad.json"
    p.write_text('{"kmeans_n_innit":7}')
    with pytest.raises(ValueError,match="kmeans_n_innit"):
        V2Config.from_json(p)


def test_defaults_preserved():
    c=V2Config()
    assert (c.kmeans_n_init,c.kmeans_max_iter,c.gmm_n_init,c.gmm_max_iter)==(20,500,3,300)
    assert (c.birch_threshold,c.minibatch_batch_size,c.output_prefix)==(.5,2048,"v2")
    assert (c.sample_min_candidate_factor,c.sample_batch_factor)==(1.1,1.5)


class FakeModel:
    def __init__(self,**kw):self.kw=kw
    def fit_predict(self,X):return np.arange(len(X))%self.kw.get("n_clusters",self.kw.get("n_components",4))
    def get_params(self):return self.kw
    def transform(self,X):return np.tile(np.arange(self.kw["n_clusters"],dtype=float)+1,(len(X),1))


def test_consensus_constructor_configuration_and_prefix(sample,tmp_path,monkeypatch):
    from src.v2 import consensus
    captured={}
    for name in ["KMeans","MiniBatchKMeans","GaussianMixture","Birch"]:
        def factory(_name=name,**kw):
            captured[_name]=kw
            return FakeModel(**kw)
        monkeypatch.setattr(consensus,name,factory)
    result=consensus.run_consensus(sample,tmp_path/"consensus",n_init=7,max_iter=29,
        gmm_n_init=6,gmm_max_iter=31,birch_threshold=.7,minibatch_batch_size=19,output_prefix="trial")
    assert captured["KMeans"]["n_init"]==7 and captured["KMeans"]["max_iter"]==29
    assert captured["MiniBatchKMeans"]["batch_size"]==19
    assert captured["MiniBatchKMeans"]["n_init"]==10
    assert captured["MiniBatchKMeans"]["max_iter"]==29
    assert captured["GaussianMixture"]["n_init"]==6 and captured["GaussianMixture"]["max_iter"]==31
    assert captured["Birch"]["threshold"]==.7
    assert (tmp_path/"consensus/trial_consensus_labels.csv").is_file()
    assert result["estimator_parameters"]["kmeans"]==captured["KMeans"]


@pytest.mark.parametrize("module_name,function",[
    ("discovery","kmeans_stability"),("uncertainty","fit_distance_uncertainty")])
def test_other_kmeans_consumers(sample,tmp_path,monkeypatch,module_name,function):
    import importlib
    module=importlib.import_module("src.v2."+module_name)
    captured=[]
    def constructor(**kw):
        captured.append(kw);return FakeModel(**kw)
    monkeypatch.setattr(module,"KMeans",constructor)
    kw={"n_init":7,"max_iter":29}
    if module_name=="discovery":kw["seeds"]=(1,2)
    getattr(module,function)(sample,tmp_path/(module_name+".csv"),**kw)
    assert captured and all(m["n_init"]==7 and m["max_iter"]==29 for m in captured)


@pytest.mark.parametrize("module_name",["anomaly_reference","model_artifacts"])
def test_artifact_fitting_constructor_options(sample,tmp_path,monkeypatch,module_name):
    import importlib
    module=importlib.import_module("src.v2."+module_name)
    captured={}
    class Reached(Exception):pass
    def constructor(**kw):
        captured.update(kw);raise Reached()
    monkeypatch.setattr(module,"KMeans",constructor)
    with pytest.raises(Reached):
        if module_name=="anomaly_reference":
            module.fit_anomaly_reference(sample,tmp_path/"ref",n_init=7,max_iter=29)
        else:
            module.fit_production_kmeans(pd.read_csv(sample)[ALL_FEATURES].to_numpy(),ALL_FEATURES,
                                        tmp_path/"model",n_init=7,max_iter=29)
    assert captured["n_init"]==7 and captured["max_iter"]==29


@pytest.mark.parametrize("module_name,function",[
    ("discovery","kmeans_stability"),("uncertainty","fit_distance_uncertainty")])
def test_existing_outputs_rejected_before_fitting(sample,tmp_path,monkeypatch,module_name,function):
    import importlib
    module=importlib.import_module("src.v2."+module_name)
    output=tmp_path/"existing.csv";output.write_text("untouched")
    def forbidden(*a,**kw):raise AssertionError("fit reached")
    monkeypatch.setattr(module,"KMeans",forbidden)
    with pytest.raises(FileExistsError):
        getattr(module,function)(sample,output)
    assert output.read_text()=="untouched"


def test_feature_output_conflict_before_raster_reads(tmp_path,monkeypatch):
    from src.v2 import raster_features
    output=tmp_path/"existing.csv";output.write_text("untouched")
    def forbidden(*a,**kw):raise AssertionError("raster opened")
    monkeypatch.setattr(raster_features,"aligned_rasters",forbidden)
    with pytest.raises(FileExistsError):
        raster_features.build_advanced_training_sample(output)


def test_reports_refuse_overwrite_and_pending_is_not_complete(tmp_path):
    from src.v2.report import write_json,write_stage_report,stage_run
    p=tmp_path/"report.json"
    write_json(p,{"first":1})
    with pytest.raises(FileExistsError):write_json(p,{"second":2})
    write_stage_report(tmp_path,"Test",[],filename="report.md")
    with pytest.raises(FileExistsError):write_stage_report(tmp_path,"Again",[],filename="report.md")
    with pytest.raises(RuntimeError):
        with stage_run(tmp_path,"test",V2Config(),[],[]) as run:
            raise RuntimeError("interrupted")
    assert not (tmp_path/"test_run.json").exists()
    assert (tmp_path/"test_run.json.pending").exists()


def test_stage3_missing_consensus_precedes_any_fit(tmp_path,sample,monkeypatch):
    import main_v2 as entry
    processed=tmp_path/"processed";processed.mkdir()
    (processed/"venus_advanced_training_sample.csv").write_bytes(sample.read_bytes())
    monkeypatch.setattr(entry,"V2_PROCESSED",processed)
    monkeypatch.setattr(entry,"V2_CLUSTERING",tmp_path/"clustering")
    def forbidden(*a,**kw):raise AssertionError("fit reached")
    monkeypatch.setattr(entry,"fit_distance_uncertainty",forbidden)
    with pytest.raises(FileNotFoundError,match="consensus"):
        entry.stage3(V2Config())


def test_stage2_propagates_all_options_and_records_completion(tmp_path,sample,monkeypatch):
    import main_v2 as entry
    processed=tmp_path/"processed";processed.mkdir()
    (processed/"venus_advanced_training_sample.csv").write_bytes(sample.read_bytes())
    monkeypatch.setattr(entry,"V2_PROCESSED",processed)
    monkeypatch.setattr(entry,"V2_CLUSTERING",tmp_path/"clustering")
    calls={}
    def consensus(*a,**kw):calls["consensus"]=kw;return {"estimator_parameters":{"checked":True}}
    def stability(*a,**kw):calls["stability"]=kw;return {}
    monkeypatch.setattr(entry,"run_consensus",consensus)
    monkeypatch.setattr(entry,"kmeans_stability",stability)
    entry.stage2(V2Config(kmeans_n_init=7,kmeans_max_iter=29,gmm_n_init=6,gmm_max_iter=31,birch_threshold=.7,minibatch_batch_size=19))
    assert calls["consensus"]["n_init"]==7 and calls["consensus"]["max_iter"]==29
    assert calls["consensus"]["gmm_n_init"]==6 and calls["consensus"]["gmm_max_iter"]==31
    assert calls["consensus"]["birch_threshold"]==.7 and calls["consensus"]["minibatch_batch_size"]==19
    assert calls["stability"]["max_iter"]==29
    manifest=json.loads((tmp_path/"clustering/stage2_run.json").read_text())
    assert manifest["status"]=="complete"
    assert manifest["configuration"]["kmeans_max_iter"]==29
    assert manifest["results"]["consensus"]["estimator_parameters"]["checked"]
    with pytest.raises(FileExistsError):entry.stage2(V2Config())


def test_anomaly_wrapper_propagates_and_validates_kmeans_options(sample,tmp_path,monkeypatch):
    from src.v2 import anomaly
    captured={}
    class Reached(Exception):pass
    def fit(*a,**kw):captured.update(kw);raise Reached()
    monkeypatch.setattr(anomaly,"fit_anomaly_reference",fit)
    with pytest.raises(Reached):
        anomaly.run_anomaly_detection(sample,tmp_path/"scores.csv",n_init=7,max_iter=29)
    assert captured["n_init"]==7 and captured["max_iter"]==29


def test_build_wrapper_output_conflict_before_production_fit(sample,tmp_path,monkeypatch):
    from src.v2 import full_map
    output=tmp_path/"maps";output.mkdir()
    def forbidden(*a,**kw):raise AssertionError("artifact loaded or fit before conflict check")
    monkeypatch.setattr(full_map,"load_anomaly_reference",forbidden)
    monkeypatch.setattr(full_map,"fit_full_map_kmeans",forbidden)
    with pytest.raises(FileExistsError):
        full_map.build_full_v2_maps(sample,output)


def test_sampling_controls_rejected_before_any_raster_read(tmp_path,monkeypatch):
    from src.v2 import raster_features as rf
    def forbidden(*a,**kw):raise AssertionError("raster read reached")
    monkeypatch.setattr(rf,"aligned_rasters",forbidden)
    with pytest.raises(ValueError,match="sample_batch_factor"):
        rf.build_advanced_training_sample(tmp_path/"sample.csv",sample_batch_factor=2.)


def test_full_map_wrapper_propagates_fit_options_without_sharing_models(sample,tmp_path,monkeypatch):
    from src.v2 import full_map as maps
    calls={}
    class Reference:
        directory=tmp_path/"anomaly"
        def validate_configuration(self,**kw):calls["anomaly_check"]=kw
        def validate_sample(self,*a):pass
    monkeypatch.setattr(maps,"load_anomaly_reference",lambda *a,**kw:Reference())
    def fit(*a,**kw):
        calls["production_fit"]=kw
        return SimpleNamespace(directory=tmp_path/"production")
    def predict(*a,**kw):
        calls["inference"]=(a,kw)
        return {}
    monkeypatch.setattr(maps,"fit_full_map_kmeans",fit)
    monkeypatch.setattr(maps,"predict_full_v2_maps",predict)
    maps.build_full_v2_maps(sample,tmp_path/"maps",anomaly_reference=tmp_path/"anomaly",
                            n_init=7,max_iter=29,output_prefix="trial")
    assert calls["production_fit"]["n_init"]==7 and calls["production_fit"]["max_iter"]==29
    args,kw=calls["inference"]
    assert args[0]==tmp_path/"production" and kw["anomaly_reference"]==tmp_path/"anomaly"
    assert kw["output_prefix"]=="trial"
    assert kw["n_init"]==7 and kw["max_iter"]==29


def test_build_existing_artifact_never_fits(sample,tmp_path,monkeypatch):
    from src.v2 import full_map as maps
    class Reference:
        directory=tmp_path/"anomaly"
        def validate_configuration(self,**kw):pass
        def validate_sample(self,*a):pass
    monkeypatch.setattr(maps,"load_anomaly_reference",lambda *a,**kw:Reference())
    def forbidden(*a,**kw):raise AssertionError("production fitting reached")
    monkeypatch.setattr(maps,"fit_full_map_kmeans",forbidden)
    monkeypatch.setattr(maps,"predict_full_v2_maps",lambda *a,**kw:{"used":str(a[0])})
    result=maps.build_full_v2_maps(sample,tmp_path/"maps",model_artifact=tmp_path/"production",
                                  anomaly_reference=tmp_path/"anomaly")
    assert result["used"]==str(tmp_path/"production")


def test_stage3_nondefault_propagation_and_prefix(tmp_path,sample,monkeypatch):
    import main_v2 as entry
    prefix="trial"
    monkeypatch.setattr(entry,"V2_PROCESSED",tmp_path/"processed"/"v2")
    monkeypatch.setattr(entry,"V2_CLUSTERING",tmp_path/"outputs"/"v2"/"clustering")
    monkeypatch.setattr(entry,"V2_ANALYSIS",tmp_path/"outputs"/"v2"/"analysis")
    monkeypatch.setattr(entry,"V2_SPATIAL",tmp_path/"outputs"/"v2"/"spatial")
    monkeypatch.setattr(entry,"V2_FIGURES",tmp_path/"figures"/"v2")
    cfg=V2Config(output_prefix=prefix,kmeans_n_init=7,kmeans_max_iter=29)
    processed,clustering,analysis,spatial,figures=entry._layout(cfg)
    processed.mkdir(parents=True);clustering.mkdir(parents=True)
    (processed/"venus_advanced_training_sample.csv").write_bytes(sample.read_bytes())
    (clustering/"trial_consensus_labels.csv").write_text("consensus_agreement\n1\n")
    monkeypatch.setattr(entry,"_require_rasters",lambda:None)
    monkeypatch.setattr(entry,"K4_RASTER",tmp_path/"missing.tif")
    calls={}
    def uncertainty(*a,**kw):calls["uncertainty"]=(a,kw);return {"estimator_parameters":kw}
    def anomaly(*a,**kw):
        calls["anomaly"]=(a,kw)
        return {"reference_artifact":str(analysis/"trial_anomaly_candidates_reference"),
                "model_id":"anomaly-id","reference_id":"reference-id"}
    def mapping(*a,**kw):calls["map"]=(a,kw);return {"model_id":"production-id"}
    monkeypatch.setattr(entry,"fit_distance_uncertainty",uncertainty)
    monkeypatch.setattr(entry,"run_anomaly_detection",anomaly)
    monkeypatch.setattr(entry,"build_full_v2_maps",mapping)
    plot_outputs=[]
    for name in ["save_confidence_histogram","save_anomaly_distribution","save_consensus_agreement","save_top_anomalies"]:
        monkeypatch.setattr(entry,name,lambda source,out:plot_outputs.append(out))
    monkeypatch.setattr(entry,"discover_and_run_associations",lambda *a:[])
    entry.stage3(cfg)
    assert all(calls[name][1]["n_init"]==7 and calls[name][1]["max_iter"]==29 for name in calls)
    assert calls["anomaly"][0][1].name=="trial_anomaly_candidates.csv"
    assert calls["map"][1]["anomaly_reference"].endswith("trial_anomaly_candidates_reference")
    assert all(p.parent==figures and p.name.startswith("trial_") for p in plot_outputs)
    manifest=json.loads((analysis/"stage3_run.json").read_text())
    assert manifest["results"]["anomaly"]["reference_id"]=="reference-id"
    assert manifest["results"]["full_map"]["model_id"]=="production-id"


def test_stage1_passes_reserved_controls_and_records_sources(tmp_path,monkeypatch):
    import main_v2 as entry
    monkeypatch.setattr(entry,"V2_PROCESSED",tmp_path/"processed")
    monkeypatch.setattr(entry,"_require_rasters",lambda:None)
    captured={}
    def build(path,**kw):
        captured.update(kw)
        pd.DataFrame(np.zeros((kw["sample_size"],17)),columns=ALL_FEATURES).to_csv(path,index=False)
        return {"output":str(path)}
    monkeypatch.setattr(entry,"build_advanced_training_sample",build)
    monkeypatch.setattr(entry,"run_feature_diagnostics",lambda *a,**kw:{})
    entry.stage1(V2Config(sample_size=32,block_size=16,local_radius=3))
    assert captured["sample_size"]==32 and captured["block_size"]==16 and captured["local_radius"]==3
    assert captured["sample_min_candidate_factor"]==1.1 and captured["sample_batch_factor"]==1.5


def test_missing_sample_rejected_before_fit(tmp_path,monkeypatch):
    from src.v2 import discovery
    def forbidden(*a,**kw):raise AssertionError("fitting reached")
    monkeypatch.setattr(discovery,"KMeans",forbidden)
    with pytest.raises(FileNotFoundError,match="missing.csv"):
        discovery.kmeans_stability(tmp_path/"missing.csv",tmp_path/"scores.csv")


@pytest.mark.parametrize("consumer",["anomaly","anomaly_reference","full_map","model_artifacts"])
def test_blocked_parent_rejected_before_fitting(sample,tmp_path,monkeypatch,consumer):
    import importlib
    module=importlib.import_module("src.v2."+consumer)
    parent=tmp_path/"blocked";parent.write_text("file")
    def forbidden(*a,**kw):raise AssertionError("fitting reached")
    if consumer=="anomaly":
        monkeypatch.setattr(module,"fit_anomaly_reference",forbidden)
        call=lambda:module.run_anomaly_detection(sample,parent/"scores.csv")
    elif consumer=="anomaly_reference":
        monkeypatch.setattr(module,"StandardScaler",forbidden)
        call=lambda:module.fit_anomaly_reference(sample,parent/"model")
    elif consumer=="full_map":
        monkeypatch.setattr(module,"fit_production_kmeans",forbidden)
        call=lambda:module.fit_full_map_kmeans(sample,parent/"model")
    else:
        monkeypatch.setattr(module,"StandardScaler",forbidden)
        call=lambda:module.fit_production_kmeans(pd.read_csv(sample)[ALL_FEATURES].to_numpy(),ALL_FEATURES,parent/"model")
    with pytest.raises(NotADirectoryError):call()


def test_anomaly_output_cannot_overlap_artifact(sample,tmp_path,monkeypatch):
    from src.v2 import anomaly
    def forbidden(*a,**kw):raise AssertionError("fitting reached")
    monkeypatch.setattr(anomaly,"fit_anomaly_reference",forbidden)
    with pytest.raises(ValueError,match="overlap"):
        anomaly.run_anomaly_detection(sample,tmp_path/"reference"/"scores.csv",
                                      reference_output_dir=tmp_path/"reference")


def test_pending_summary_rejected_before_stage_fit(sample,tmp_path,monkeypatch):
    import main_v2 as entry
    processed=tmp_path/"processed";processed.mkdir()
    (processed/"venus_advanced_training_sample.csv").write_bytes(sample.read_bytes())
    clustering=tmp_path/"clustering";clustering.mkdir()
    (clustering/"stage2_summary.json.pending").write_text("partial")
    monkeypatch.setattr(entry,"V2_PROCESSED",processed)
    monkeypatch.setattr(entry,"V2_CLUSTERING",clustering)
    def forbidden(*a,**kw):raise AssertionError("fitting reached")
    monkeypatch.setattr(entry,"run_consensus",forbidden)
    with pytest.raises(FileExistsError,match="pending"):entry.stage2(V2Config())


def test_interrupted_csv_publication_has_no_final_table(tmp_path,monkeypatch):
    from src.v2.report import write_csv
    table=pd.DataFrame({"value":[1.,2.]})
    def fail(*a,**kw):raise OSError("interrupted")
    monkeypatch.setattr(table,"to_csv",fail)
    destination=tmp_path/"table.csv"
    with pytest.raises(OSError,match="interrupted"):write_csv(table,destination,index=False)
    assert not destination.exists()
    assert (tmp_path/"table.csv.pending").exists()


def test_saved_anomaly_nondefaults_and_role_rejection(sample,tmp_path):
    from threadpoolctl import threadpool_limits
    from src.v2.anomaly_reference import fit_anomaly_reference,load_anomaly_reference
    from src.v2.model_artifacts import load_production_kmeans
    with threadpool_limits(limits=1):
        artifact=fit_anomaly_reference(sample,tmp_path/"anomaly",n_init=3,max_iter=11,n_estimators=3,fit_subset=8)
    loaded=load_anomaly_reference(artifact.directory)
    assert loaded.kmeans.n_init==3 and loaded.kmeans.max_iter==11
    loaded.validate_configuration(k=4,random_state=42,contamination=.01,n_estimators=3,fit_subset=8,n_init=3,max_iter=11)
    with pytest.raises(ValueError,match="configuration"):
        loaded.validate_configuration(k=4,random_state=42,contamination=.01,n_estimators=3,fit_subset=8,n_init=7,max_iter=11)
    with pytest.raises(ValueError):
        load_production_kmeans(artifact.directory)


def test_invalid_consensus_table_rejected_before_fit(tmp_path,sample,monkeypatch):
    import main_v2 as entry
    processed=tmp_path/"processed";processed.mkdir()
    (processed/"venus_advanced_training_sample.csv").write_bytes(sample.read_bytes())
    clustering=tmp_path/"clustering";clustering.mkdir()
    (clustering/"v2_consensus_labels.csv").write_text("wrong_column\n1\n")
    monkeypatch.setattr(entry,"V2_PROCESSED",processed)
    monkeypatch.setattr(entry,"V2_CLUSTERING",clustering)
    monkeypatch.setattr(entry,"V2_ANALYSIS",tmp_path/"analysis")
    monkeypatch.setattr(entry,"_require_rasters",lambda:None)
    def forbidden(*a,**kw):raise AssertionError("fit reached")
    monkeypatch.setattr(entry,"fit_distance_uncertainty",forbidden)
    with pytest.raises(ValueError,match="consensus_agreement"):entry.stage3(V2Config())
