"""Small independent-reference regressions for masked population texture."""
import math
import numpy as np
import pytest
from src.v2.raster_features import _local_std

def reference(data, valid, radius):
    values=np.pad(data,radius,mode="edge")
    masks=np.pad(valid,radius,mode="edge")
    out=np.zeros(data.shape)
    for r,c in np.ndindex(data.shape):
        p=values[r:r+2*radius+1,c:c+2*radius+1]
        m=masks[r:r+2*radius+1,c:c+2*radius+1]
        x=p[m]
        if len(x):
            delta=[float(v)-float(x[0]) for v in x]
            mean=math.fsum(delta)/len(delta)
            out[r,c]=math.sqrt(math.fsum((v-mean)**2 for v in delta)/len(delta))
    return out

@pytest.mark.parametrize("offset,scale",[(0.,1.),(1e6,.01),(1e12,.001),(27.,0.),(27.,1e-12)])
@pytest.mark.parametrize("radius",[0,1,2,3])
def test_stable_reference_edges_and_missing(offset,scale,radius):
    rng=np.random.default_rng(42)
    a=offset+scale*rng.normal(size=(9,11))
    mask=rng.random(a.shape)>.3
    a[~mask]=np.nan
    actual=_local_std(a,mask,radius)
    np.testing.assert_allclose(actual,reference(a,mask,radius),rtol=1e-10,atol=1e-8)
    assert np.isfinite(actual).all()

@pytest.mark.parametrize("offset,scale",[(1e6,.01),(1500.,.1),(27.,0.)])
def test_identical_neighborhoods_different_contexts(offset,scale):
    rng=np.random.default_rng(42)
    a=offset+scale*rng.normal(size=(19,21))
    mask=rng.random(a.shape)>.15
    b=a[3:16,4:17].copy(); mb=mask[3:16,4:17].copy()
    u=_local_std(a,mask,2)[5:14,6:15]
    v=_local_std(b,mb,2)[2:11,2:11]
    np.testing.assert_allclose(u,v,rtol=1e-10,atol=1e-8)
    np.testing.assert_array_equal(u,v)

def test_empty_and_single_valid_neighborhoods():
    a=np.full((9,9),np.nan);m=np.zeros(a.shape,bool)
    np.testing.assert_array_equal(_local_std(a,m,2),np.zeros(a.shape))
    a[4,4]=12345.;m[4,4]=True
    np.testing.assert_array_equal(_local_std(a,m,2),np.zeros(a.shape))
