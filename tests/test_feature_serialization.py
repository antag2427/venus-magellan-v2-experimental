"""Float64 feature persistence tests; synthetic data only."""
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.v2.config import ALL_FEATURES, BASE_FEATURES
from src.v2.model_artifacts import matrix_sha256
from src.v2.anomaly_reference import fit_anomaly_reference, load_anomaly_reference
from src.v2.anomaly import run_anomaly_detection


@pytest.fixture
def precise_table(tmp_path):
    values=np.random.default_rng(19).normal(size=(32,17))
    values[0,:6]=[np.nextafter(1.,2.),1.2345678901234567,1e-16,
                   -12345.678901234567,1.000000000000003,1.2345678901234567e-12]
    table=pd.DataFrame(values,columns=ALL_FEATURES)
    table.insert(0,"col",np.arange(32,dtype=np.int64)*31)
    table.insert(0,"row",np.arange(32,dtype=np.int64)*37)
    path=tmp_path/"precise.csv"
    table.to_csv(path,index=False,float_format="%.17g")
    return table,path


def test_reference_fit_preserves_original_feature_matrix(precise_table,tmp_path):
    table,path=precise_table
    with threadpool_limits(limits=1):
        ref=fit_anomaly_reference(path,tmp_path/"reference",n_estimators=3,fit_subset=16)
    assert ref.metadata["input_sample"]["feature_matrix_sha256"]==matrix_sha256(table[ALL_FEATURES].to_numpy())


def test_sample_scoring_csv_preserves_exact_scores_and_coordinates(precise_table,tmp_path):
    table,path=precise_table
    out=tmp_path/"scored.csv"
    with threadpool_limits(limits=1):
        result=run_anomaly_detection(path,out,n_estimators=3,fit_subset=16)
    ref=load_anomaly_reference(result["reference_artifact"])
    expected=ref.score(table[ALL_FEATURES].to_numpy(),ALL_FEATURES)
    saved=pd.read_csv(out,float_precision="round_trip").sort_values("row")
    np.testing.assert_array_equal(saved[["row","col"]].to_numpy(),table[["row","col"]].to_numpy())
    for key,values in expected.items():
        np.testing.assert_array_equal(saved[key].to_numpy(),values)


def test_full_map_fit_reader_preserves_features(precise_table,tmp_path,monkeypatch):
    from src.v2 import full_map
    table,path=precise_table
    def capture(values,names,*a,**kw):
        assert names==ALL_FEATURES
        np.testing.assert_array_equal(values,table[ALL_FEATURES].to_numpy())
        return "captured without fitting"
    monkeypatch.setattr(full_map,"fit_production_kmeans",capture)
    assert full_map.fit_full_map_kmeans(path,tmp_path/"model")=="captured without fitting"


def test_actual_feature_writer_roundtrips_without_changing_header_or_coordinates(tmp_path,monkeypatch):
    from src.v2 import raster_features as rf
    y,x=np.mgrid[:4,:4]
    arrays={n:(0.12345678901234567+np.hypot(x,y)*.0000001+i*3.7) for i,n in enumerate(BASE_FEATURES)}
    arrays["height"][1,1]=np.nextafter(arrays["height"][1,1],np.inf)
    masks={n:np.ones((4,4),bool) for n in BASE_FEATURES}
    radar=SimpleNamespace(width=4,height=4,res=(2025.,2025.))
    monkeypatch.setattr(rf,"aligned_rasters",lambda:(ExitStack(),radar,{}))
    monkeypatch.setattr(rf,"_read_aligned_window",lambda *a:(arrays,masks))
    out=tmp_path/"features.csv"
    rf.build_advanced_training_sample(out,sample_size=16,block_size=4)
    loaded=pd.read_csv(out,float_precision="round_trip").sort_values(["row","col"])
    derived=rf.derive_local_features(arrays,masks,2,2025.,2025.)
    # Preserve the existing writer header; consumers explicitly select ALL_FEATURES.
    assert loaded.columns.tolist()==["row","col"]+BASE_FEATURES+list(derived)
    assert loaded[ALL_FEATURES].columns.tolist()==ALL_FEATURES
    np.testing.assert_array_equal(loaded.row,np.repeat(np.arange(4),4))
    np.testing.assert_array_equal(loaded.col,np.tile(np.arange(4),4))
    expected=np.column_stack([{**arrays,**derived}[n].ravel() for n in ALL_FEATURES])
    np.testing.assert_array_equal(loaded[ALL_FEATURES].to_numpy(),expected)
    assert np.issubdtype(loaded.row.dtype,np.integer)
    assert np.issubdtype(loaded.col.dtype,np.integer)


@pytest.mark.parametrize("invalid",[np.nan,np.inf,-np.inf])
def test_nonfinite_csv_features_remain_rejected(precise_table,tmp_path,invalid):
    table,path=precise_table
    table.loc[0,"radar"]=invalid
    bad=tmp_path/"invalid.csv"
    table.to_csv(bad,index=False,float_format="%.17g")
    with pytest.raises(ValueError,match="nonfinite"):
        fit_anomaly_reference(bad,tmp_path/"invalid_reference",n_estimators=3)
    assert not (tmp_path/"invalid_reference").exists()
