"""Routing/core-context regression tests: all writes and fits are synthetic."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform
from threadpoolctl import threadpool_limits

from test_integration_acceptance import synthetic_world, invoke, forbid_fitting
from src.v2.config import V2Config, ALL_FEATURES
from src.v2 import raster_features as rf


def test_standalone_help_and_paths_do_not_require_host(tmp_path):
    root=Path(__file__).resolve().parents[1]
    help_run=subprocess.run([sys.executable,"-B",str(root/"main_v2.py"),"--help"],
                             cwd=tmp_path,capture_output=True,text=True,check=False,stdin=subprocess.DEVNULL)
    assert help_run.returncode==0,help_run.stderr
    probe=subprocess.run([sys.executable,"-B","-c",
        "import src.paths; print(src.paths.__file__); print(src.paths.RADAR_FILE)"],
        cwd=root,capture_output=True,text=True,check=False,stdin=subprocess.DEVNULL)
    assert probe.returncode==0,probe.stderr
    assert str(root/"src/paths.py") in probe.stdout
    assert str(root/"data/raw/venus.tif") in probe.stdout


def test_config_relative_paths_from_json_directory(synthetic_world,tmp_path,monkeypatch):
    _,_,_,paths,_,_=synthetic_world
    directory=tmp_path/"configs";directory.mkdir()
    payload={"raster_inputs":{name:os.path.relpath(path,directory) for name,path in paths.items()},
             "output_root":"runs/pilot","core_window":{"row":4,"col":4,"height":8,"width":8},
             "halo_policy":"require_full"}
    config=directory/"config.json";config.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path.parent)
    cfg=V2Config.from_json(config)
    assert cfg.raster_inputs=={name:str(path.resolve()) for name,path in paths.items()}
    assert cfg.output_root==str((directory/"runs/pilot").resolve())
    assert cfg.core_window==payload["core_window"]


@pytest.mark.parametrize("extra",[
    {"raster_inputs":{"radar":"one.tif"}},{"raster_inputs":[]},
    {"raster_inputs":{name:3 for name in ["radar","height","emissivity","slope","reflectivity"]}},
    {"output_root":"."},{"output_root":".."},{"output_root":"/"},
    {"halo_policy":"nearest"},{"core_window":{"row":1,"col":2,"height":8}},
    {"core_window":{"row":1.5,"col":2,"height":8,"width":8}},
    {"core_window":{"row":-1,"col":2,"height":8,"width":8}},
])
def test_bad_routing_configuration(extra):
    with pytest.raises(ValueError):V2Config(**extra)


def test_input_inspection_hashes_and_nonrasters(synthetic_world,tmp_path):
    from src.v2.routing import inspect_rasters
    from src.v2.model_artifacts import sha256_file
    _,_,_,paths,_,_=synthetic_world
    metadata=inspect_rasters(paths)
    assert list(metadata)==["radar","height","emissivity","slope","reflectivity"]
    for name,item in metadata.items():
        assert item["path"]==str(paths[name].resolve())
        assert item["sha256"]==sha256_file(paths[name])
    missing=dict(paths,height=tmp_path/"missing.tif")
    with pytest.raises(FileNotFoundError,match="height"):inspect_rasters(missing)
    bad=tmp_path/"not_raster.tif";bad.write_text("not a raster")
    with pytest.raises(ValueError,match="height"):inspect_rasters(dict(paths,height=bad))


def test_exact_target_grid_plan_and_halo(synthetic_world):
    _,_,_,paths,_,_=synthetic_world
    ctx,radar,sources=rf.aligned_rasters(paths)
    with ctx:
        core={"row":4,"col":4,"height":8,"width":8}
        plan=rf.plan_core_window(radar,core,halo_policy="require_full")
        assert plan.core==Window(4,4,8,8)
        assert plan.context==Window(1,1,14,14)
        assert plan.core_row_in_context==3 and plan.core_col_in_context==3
        assert plan.context_transform==window_transform(plan.context,radar.transform)
        assert plan.core_transform==window_transform(plan.core,radar.transform)
        bundle=rf.read_core_context(radar,sources,plan,minimum_common_valid=20)
        assert all(a.shape==(14,14) for a in bundle["arrays"].values())
        for name in ["height","emissivity","slope","reflectivity"]:
            assert sources[name].dtypes==("float64",)
            assert sources[name].transform==radar.transform
            assert sources[name].resampling==rasterio.enums.Resampling.bilinear
        all_arrays,all_masks=rf._read_aligned_window(sources,Window(0,0,16,16))
        for name in sources:
            np.testing.assert_array_equal(bundle["arrays"][name],all_arrays[name][1:15,1:15])
            np.testing.assert_array_equal(bundle["masks"][name],all_masks[name][1:15,1:15])
        expected=np.logical_and.reduce(list(all_masks.values()))[4:12,4:12]
        assert bundle["coverage"]["common_valid_core"]==int(expected.sum())
        assert bundle["coverage"]["invalid_core"]==64-int(expected.sum())
        with pytest.raises(ValueError,match="coverage"):
            rf.read_core_context(radar,sources,plan,minimum_common_valid=65)
        smaller=rf.derive_local_features(bundle["arrays"],bundle["masks"],2,1,1)
        larger=rf.derive_local_features(all_arrays,all_masks,2,1,1)
        for name in smaller:
            np.testing.assert_allclose(smaller[name][3:11,3:11][expected],
                larger[name][4:12,4:12][expected],rtol=1e-10,atol=1e-8,err_msg=name)


def test_required_262_shape_without_reading_real_inputs():
    from types import SimpleNamespace
    grid=SimpleNamespace(width=1024,height=1024,transform=from_origin(0,1024,1,1),crs=None)
    plan=rf.plan_core_window(grid,{"row":100,"col":200,"height":256,"width":256})
    assert plan.context==Window(197,97,262,262)
    assert plan.halo_complete


def test_boundary_policy_is_explicit(synthetic_world):
    _,_,_,paths,_,_=synthetic_world
    ctx,radar,sources=rf.aligned_rasters(paths)
    with ctx:
        core={"row":0,"col":0,"height":8,"width":8}
        plan=rf.plan_core_window(radar,core,halo_policy="clip")
        assert plan.context==Window(0,0,11,11) and not plan.halo_complete
        assert plan.core_row_in_context==0 and plan.core_col_in_context==0
        with pytest.raises(ValueError,match="halo"):
            rf.plan_core_window(radar,core,halo_policy="require_full")
        with pytest.raises(ValueError,match="bounds"):
            rf.plan_core_window(radar,dict(core,row=14))


def test_core_sample_excludes_halo_and_keeps_global_coordinates(synthetic_world,tmp_path):
    _,_,_,paths,_,_=synthetic_world
    out=tmp_path/"sample.csv"
    result=rf.build_advanced_training_sample(out,sample_size=32,block_size=4,
        raster_inputs=paths,core_window={"row":4,"col":4,"height":8,"width":8},halo_policy="require_full")
    frame=pd.read_csv(out,float_precision="round_trip")
    assert len(frame)==32
    assert frame.row.between(4,11).all() and frame.col.between(4,11).all()
    assert result["window"]["core_row_in_context"]==3
    ctx,radar,sources=rf.aligned_rasters(paths)
    with ctx:
        plan=rf.plan_core_window(radar,{"row":4,"col":4,"height":8,"width":8})
        bundle=rf.read_core_context(radar,sources,plan)
        assert result["total_valid_aligned_pixels"]==bundle["coverage"]["common_valid_core"]
        features={**bundle["arrays"],**rf.derive_local_features(bundle["arrays"],bundle["masks"],2,1,1)}
        rr=frame.row.to_numpy()-1;cc=frame.col.to_numpy()-1
        for name in ALL_FEATURES:
            np.testing.assert_allclose(frame[name],features[name][rr,cc],rtol=1e-10,atol=1e-8)


def test_config_only_cli_contains_every_output_and_maps_only_core(synthetic_world,monkeypatch):
    entry,_,root,paths,base,_=synthetic_world
    isolated=root/"ISOLATED_ROOT"
    cfg=V2Config(**dict(base.to_dict(),raster_inputs={n:str(p) for n,p in paths.items()},
        output_root=str(isolated),core_window={"row":4,"col":4,"height":8,"width":8},
        halo_policy="require_full",sample_size=32,consensus_subset=24,anomaly_subset=16,output_prefix="corePilot"))
    config=root/"routed.json";cfg.save_json(config)
    # Restore actual module constants: this run must rely on config, not fixture routing.
    pack=Path(entry.__file__).resolve().parent
    monkeypatch.setattr(entry,"PROJECT_ROOT",pack)
    for name,value in {"V2_PROCESSED":pack/"data/processed/v2","V2_CLUSTERING":pack/"outputs/v2/clustering",
        "V2_ANALYSIS":pack/"outputs/v2/analysis","V2_SPATIAL":pack/"outputs/v2/spatial",
        "V2_FIGURES":pack/"figures/v2"}.items():monkeypatch.setattr(entry,name,value)
    monkeypatch.setattr(entry,"BASE_PATHS",{n:root/"wrong_default.tif" for n in paths})
    monkeypatch.setattr(rf,"BASE_PATHS",entry.BASE_PATHS)
    monkeypatch.setattr(rf,"RADAR_FILE",root/"wrong_default.tif")
    with threadpool_limits(limits=1):invoke(monkeypatch,entry,"all",config)
    processed,clustering,analysis,spatial,figures=entry._layout(cfg)
    for path in [processed,clustering,analysis,spatial,figures]:
        assert path.resolve().is_relative_to(isolated.resolve())
    summary=json.loads((analysis/"stage3_summary.json").read_text())
    stage1=json.loads((processed/"stage1_run.json").read_text())
    assert len(stage1["inputs"])==5
    from src.v2.model_artifacts import sha256_file
    for item in stage1["inputs"]:assert item["sha256"]==sha256_file(item["path"])
    assert stage1["configuration"]["core_window"]==cfg.core_window
    frame=pd.read_csv(processed/"venus_advanced_training_sample.csv",float_precision="round_trip")
    with rasterio.open(summary["full_map"]["cluster_raster"]) as dst:
        labels=dst.read(1)
        assert dst.shape==(8,8)
        assert dst.transform==from_origin(4,12,1,1)
        from src.v2.model_artifacts import load_production_kmeans
        model=load_production_kmeans(summary["full_map"]["model_artifact"])
        np.testing.assert_array_equal(labels[frame.row-4,frame.col-4],model.predict(frame[ALL_FEATURES].to_numpy(),ALL_FEATURES))
    # All actual generated stage output records, including files not named explicitly above.
    for manifest in isolated.rglob("stage*_run.json"):
        obj=json.loads(manifest.read_text())
        assert obj["status"]=="complete"
        assert all(Path(value).resolve().is_relative_to(isolated.resolve()) for value in obj["outputs"])
    from src.v2.routing import planned_outputs,validate_output_routes
    plan=planned_outputs(cfg,pack)
    validate_output_routes(cfg,plan,paths)
    assert all(Path(p).resolve().is_relative_to(isolated.resolve()) for p in plan.values())
    protected={p:p.read_bytes() for p in isolated.rglob("*") if p.is_file()}
    with monkeypatch.context() as guard:
        forbid_fitting(guard)
        with pytest.raises(FileExistsError):invoke(guard,entry,"all",config)
    assert all(p.read_bytes()==data for p,data in protected.items())


def test_output_root_cannot_collide_with_inputs(synthetic_world):
    from src.v2.routing import planned_outputs,validate_output_routes
    _,_,root,paths,base,_=synthetic_world
    cfg=V2Config(**dict(base.to_dict(),output_root=str(paths["radar"])))
    with pytest.raises((ValueError,NotADirectoryError)):
        validate_output_routes(cfg,planned_outputs(cfg,root),paths)


@pytest.mark.parametrize("change",["core","source","sample"])
def test_discovery_rejects_stale_sample_provenance(synthetic_world,monkeypatch,change):
    entry,_,root,paths,base,_=synthetic_world
    cfg=V2Config(**dict(base.to_dict(),raster_inputs={n:str(p) for n,p in paths.items()},
        output_root=str(root/"ISOLATED"),core_window={"row":4,"col":4,"height":8,"width":8},
        halo_policy="require_full",sample_size=32,consensus_subset=24,anomaly_subset=16))
    with threadpool_limits(limits=1):entry.stage1(cfg)
    altered=cfg
    if change=="core":
        altered=V2Config(**dict(cfg.to_dict(),core_window={"row":5,"col":4,"height":8,"width":8}))
    elif change=="source":
        clone=root/"different_height.tif"
        clone.write_bytes(paths["height"].read_bytes())
        with rasterio.open(clone,"r+") as dst:
            data=dst.read(1);data[2,2]+=1;dst.write(data,1)
        altered=V2Config(**dict(cfg.to_dict(),raster_inputs=dict(cfg.raster_inputs,height=str(clone))))
    else:
        sample=entry._layout(cfg)[0]/"venus_advanced_training_sample.csv"
        with sample.open("a") as stream:stream.write("\n")
    def forbidden(*a,**kw):raise AssertionError("Fitting reached stale feature sample")
    monkeypatch.setattr(entry,"run_consensus",forbidden)
    with pytest.raises(ValueError,match="provenance"):
        entry.stage2(altered)


def test_interpret_rejects_changed_core_before_fitting(synthetic_world,monkeypatch):
    entry,_,root,paths,base,_=synthetic_world
    cfg=V2Config(**dict(base.to_dict(),raster_inputs={n:str(p) for n,p in paths.items()},
        output_root=str(root/"ISOLATED"),core_window={"row":4,"col":4,"height":8,"width":8},
        halo_policy="require_full",sample_size=32,consensus_subset=24,anomaly_subset=16))
    with threadpool_limits(limits=1):
        entry.stage1(cfg);entry.stage2(cfg)
    changed=V2Config(**dict(cfg.to_dict(),core_window={"row":5,"col":4,"height":8,"width":8}))
    def forbidden(*a,**kw):raise AssertionError("Fitting reached stale feature sample")
    monkeypatch.setattr(entry,"fit_distance_uncertainty",forbidden)
    with pytest.raises(ValueError,match="provenance"):entry.stage3(changed)


def test_explicit_full_grid_does_not_read_unrelated_v1_atlas(synthetic_world,monkeypatch):
    entry,_,root,paths,base,_=synthetic_world
    sentinel=root/"unrelated_v1.tif";sentinel.write_text("must not be read")
    monkeypatch.setattr(entry,"K4_RASTER",sentinel)
    def forbidden(*a,**kw):raise AssertionError("Unrelated V1 atlas read")
    monkeypatch.setattr(entry,"extract_coarse_objects",forbidden)
    cfg=V2Config(**dict(base.to_dict(),raster_inputs={n:str(p) for n,p in paths.items()},
                         output_root=str(root/"ISOLATED")))
    config=root/"explicit_full_grid.json";cfg.save_json(config)
    invoke(monkeypatch,entry,"all",config)
    summary=json.loads((entry._layout(cfg)[2]/"stage3_summary.json").read_text())
    assert summary["coarse_objects"] is None
    assert sentinel.read_text()=="must not be read"


def test_standalone_features_cli_uses_config_without_host_adapter(synthetic_world,tmp_path):
    _,_,root,paths,base,_=synthetic_world
    cfg=V2Config(**dict(base.to_dict(),raster_inputs={n:str(p) for n,p in paths.items()},
        output_root=str(root/"STANDALONE_ONLY"),core_window={"row":4,"col":4,"height":8,"width":8},
        halo_policy="require_full",sample_size=32,consensus_subset=24,anomaly_subset=16,
        output_prefix="independent"))
    config=root/"standalone.json";cfg.save_json(config)
    pack=Path(__file__).resolve().parents[1]
    child=subprocess.run([sys.executable,"-B",str(pack/"main_v2.py"),"features","--config",str(config)],
        cwd=tmp_path,capture_output=True,text=True,stdin=subprocess.DEVNULL,check=False)
    assert child.returncode==0,child.stderr+"\n"+child.stdout
    processed=Path(cfg.output_root)/cfg.output_prefix/"processed"
    manifest=json.loads((processed/"stage1_run.json").read_text())
    assert manifest["status"]=="complete" and len(manifest["inputs"])==5
    assert manifest["configuration"]==cfg.to_dict()
    table=pd.read_csv(processed/"venus_advanced_training_sample.csv",float_precision="round_trip")
    assert len(table)==32 and table.row.between(4,11).all() and table.col.between(4,11).all()
    assert np.isfinite(table[ALL_FEATURES].to_numpy()).all()
