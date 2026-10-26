from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import label

from src.v2.consensus import align_labels, normalized_entropy
from src.v2.raster_features import derive_local_features
from src.v2.config import ALL_FEATURES
from src.v2.uncertainty import fit_distance_uncertainty


def test_label_alignment_recovers_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    candidate = np.array([2, 2, 0, 0, 3, 3, 1, 1])
    aligned, mapping = align_labels(reference, candidate, 4)
    assert np.array_equal(aligned, reference)
    assert mapping == {2: 0, 0: 1, 3: 2, 1: 3}


def test_derived_feature_shapes():
    rng = np.random.default_rng(42)
    arrays = {name: rng.normal(size=(11, 13)) for name in ["radar", "height", "emissivity", "slope", "reflectivity"]}
    masks = {name: np.ones((11, 13), dtype=bool) for name in arrays}
    features = derive_local_features(arrays, masks, radius=2, pixel_size_x=2025, pixel_size_y=2025)
    assert set(ALL_FEATURES) == set(arrays) | set(features)
    assert all(v.shape == (11, 13) for v in features.values())


def test_uncertainty_output(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(300, 5))
    data[:, 0] += np.repeat([-3, 0, 3], 100)
    df = pd.DataFrame(data, columns=["radar", "height", "emissivity", "slope", "reflectivity"])
    for feature in ALL_FEATURES[5:]:
        df[feature] = rng.normal(size=len(df))
    inp = tmp_path / "sample.csv"
    out = tmp_path / "uncertainty.csv"
    df.to_csv(inp, index=False)
    result = fit_distance_uncertainty(inp, out, k=4, random_state=42)
    assert out.exists()
    assert 0 <= result["mean_confidence"] <= 1


def test_normalized_entropy_eight_samples_four_models():
    labels = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [3, 3, 3, 3],
        [2, 2, 2, 3],
        [1, 1, 3, 3],
        [0, 1, 2, 2],
    ])
    original = labels.copy()
    result = normalized_entropy(labels, k=4)
    three_one = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(4)
    expected = [0, 1, 0.5, three_one, 0, three_one, 0.5, 0.75]
    assert result.shape == (8,)
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(labels, original)


def test_normalized_entropy_unanimous_labels():
    labels = np.repeat(np.arange(4)[:, None], 7, axis=1)
    np.testing.assert_allclose(normalized_entropy(labels, k=4), 0, atol=1e-12)


def test_normalized_entropy_balanced_labels():
    labels = np.tile(np.arange(4), (8, 2))
    np.testing.assert_allclose(normalized_entropy(labels, k=4), 1, atol=1e-12)


def test_normalized_entropy_mixed_sample_model_counts():
    labels = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2],
        [0, 0, 0, 1, 1, 2],
        [1, 1, 1, 2, 2, 2],
        [0, 1, 2, 2, 2, 2],
    ])
    probabilities = np.array([
        [1, 0, 0],
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 1 / 3, 1 / 6],
        [0, 1 / 2, 1 / 2],
        [1 / 6, 1 / 6, 2 / 3],
    ])
    terms = np.zeros_like(probabilities)
    nonzero = probabilities > 0
    terms[nonzero] = probabilities[nonzero] * np.log(probabilities[nonzero])
    expected = -terms.sum(axis=1) / np.log(3)
    result = normalized_entropy(labels, k=3)
    assert result.shape == (5,)
    assert np.isfinite(result).all()
    assert np.all((result >= -1e-12) & (result <= 1 + 1e-12))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_normalized_entropy_normalizes_by_clusters_not_models():
    labels = np.array([[0, 1], [1, 0], [2, 3]])
    np.testing.assert_allclose(normalized_entropy(labels, k=4), 0.5, atol=1e-12)


def test_normalized_entropy_single_model():
    labels = np.array([[0], [1], [3]])
    np.testing.assert_allclose(normalized_entropy(labels, k=4), 0, atol=1e-12)


def test_normalized_entropy_empty_samples():
    result = normalized_entropy(np.empty((0, 4), dtype=int), k=4)
    assert result.shape == (0,)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("labels,k", [
    (np.zeros(8, dtype=int), 4),
    (np.zeros((2, 2, 2), dtype=int), 4),
    (np.empty((8, 0), dtype=int), 4),
    (np.zeros((2, 4), dtype=int), 0),
    (np.zeros((2, 4), dtype=int), 1),
    (np.zeros((2, 4), dtype=int), 2.5),
    (np.array([[0, -1]]), 4),
    (np.array([[0, 4]]), 4),
    (np.array([[0, 0.5]]), 4),
    (np.array([[0, np.nan]]), 4),
    (np.array([[0, np.inf]]), 4),
    (np.array([["a", "b"]]), 4),
])
def test_normalized_entropy_invalid_input(labels, k):
    with pytest.raises(ValueError):
        normalized_entropy(labels, k)


@pytest.fixture
def memory_aligned_inputs(monkeypatch):
    """Five tiny TIFFs live exclusively in GDAL /vsimem/, not project files."""
    from contextlib import ExitStack
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from src.v2 import raster_features as rf

    crs = "+proj=eqc +lat_ts=0 +lon_0=0 +R=6051000 +units=m +no_defs"
    target_transform = from_origin(0, 16, 1, 1)
    source_transform = from_origin(0, 16, 2, 2)
    nodatas = {"height": -32768, "emissivity": -32768, "slope": 0, "reflectivity": 0}
    paths, expected = {}, {}
    with ExitStack() as memories:
        def make(data, transform, nodata):
            memory = memories.enter_context(MemoryFile())
            with memory.open(
                driver="GTiff", width=data.shape[1], height=data.shape[0],
                count=1, dtype=data.dtype.name, crs=crs,
                transform=transform, nodata=nodata,
            ) as dst:
                dst.write(data, 1)
            return memory.name

        radar = np.ones((16, 16), dtype=np.uint8)
        radar[0, 0] = 0
        paths["radar"] = make(radar, target_transform, 0)
        y, x = np.mgrid[:8, :8]
        for i, name in enumerate(nodatas):
            dtype = np.int16 if name in ("height", "emissivity") else np.uint8
            data = (20 + 7 * i + 10 * y + x).astype(dtype)
            data[6:, 6:] = nodatas[name]
            paths[name] = make(data, source_transform, nodatas[name])
            # Target (5,5) maps to source centre coordinates (2.25,2.25).
            expected[name] = 20 + 7 * i + 10 * 2.25 + 2.25
        monkeypatch.setattr(rf, "RADAR_FILE", paths["radar"])
        monkeypatch.setattr(rf, "BASE_PATHS", paths)
        yield rf, paths, expected, target_transform, nodatas


def test_raster_float64_interpolation_grid_and_masks(memory_aligned_inputs):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    rf, paths, expected, transform, nodatas = memory_aligned_inputs
    ctx, radar, sources = rf.aligned_rasters()
    with ctx:
        assert radar.shape == (16, 16)
        assert radar.transform == transform
        assert radar.dtypes == ("uint8",)
        arrays, masks = rf._read_aligned_window(sources, Window(0, 0, 16, 16))
        assert not masks["radar"][0, 0]
        for name in nodatas:
            view = sources[name]
            assert view.dtypes == ("float64",)
            assert view.crs == radar.crs
            assert view.transform == radar.transform
            assert view.shape == radar.shape
            assert view.bounds == radar.bounds
            assert view.res == radar.res
            assert view.nodata == -32768.0
            assert view.resampling == Resampling.bilinear
            assert arrays[name].dtype == np.float64
            assert not float(expected[name]).is_integer()
            np.testing.assert_allclose(arrays[name][5, 5], expected[name], atol=1e-12)
            assert not masks[name][13, 13]
            assert arrays[name][13, 13] == -32768.0
            with rasterio.open(paths[name]) as src:
                assert src.nodata == nodatas[name]
                # Precisely the original settings, intentionally without dtype.
                with WarpedVRT(
                    src, crs=radar.crs, transform=radar.transform,
                    width=radar.width, height=radar.height,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata, nodata=-32768.0,
                ) as original:
                    old = original.read(1, window=Window(0, 0, 16, 16))
                    np.testing.assert_array_equal(
                        rf._safe_nodata_mask(old, original.nodata), masks[name]
                    )


def _raster_feature_surface(shape=(23, 27), surface="quadratic", missing=True):
    from src.v2.config import BASE_FEATURES

    y, x = np.mgrid[:shape[0], :shape[1]]
    height = 2 * x + 3 * y + 10 if surface == "plane" else x*x + 2*y*y + 3*x - 4*y + 7
    arrays = {name: (height + 0.25 * i).astype(np.float64) for i, name in enumerate(BASE_FEATURES)}
    arrays["height"] = height.astype(np.float64)
    arrays["radar"] = (5 * x - 2 * y + 3).astype(np.float64)
    masks = {name: np.ones(shape, dtype=bool) for name in BASE_FEATURES}
    if missing:
        for name, row, col in [
            ("height", 12, 16), ("radar", 8, 7),
            ("height", 0, 3), ("radar", shape[0]-1, shape[1]-2),
        ]:
            masks[name][row, col] = False
            arrays[name][row, col] = -32768
        for name in BASE_FEATURES:
            masks[name][4, 4] = False
            arrays[name][4, 4] = -32768
    return arrays, masks


def _raster_core_features(arrays, masks, core, read, radius):
    r, c, h, w = map(int, (read.row_off, read.col_off, read.height, read.width))
    local = {name: a[r:r+h, c:c+w] for name, a in arrays.items()}
    valid = {name: a[r:r+h, c:c+w] for name, a in masks.items()}
    features = {**local, **derive_local_features(local, valid, radius, 2025, 2025)}
    r0, c0 = int(core.row_off)-r, int(core.col_off)-c
    h, w = int(core.height), int(core.width)
    result = np.stack([
        features[name][r0:r0+h, c0:c0+w] for name in ALL_FEATURES
    ], axis=-1)
    core_valid = np.logical_and.reduce(list(valid.values()))[r0:r0+h, c0:c0+w]
    assert result.shape == (h, w, 17)
    return result, core_valid


def test_raster_radius_two_reproduces_and_fixes_halo_discrepancy():
    from rasterio.windows import Window
    from src.v2 import raster_features as rf

    arrays, masks = _raster_feature_surface((25, 25))
    full = {**arrays, **derive_local_features(arrays, masks, 2, 2025, 2025)}
    whole = np.stack([full[name] for name in ALL_FEATURES], axis=-1)
    core = Window(8, 8, 8, 8)
    old, valid = _raster_core_features(arrays, masks, core, Window(6, 6, 12, 12), 2)
    expected = whole[8:16, 8:16]
    gradient = ALL_FEATURES.index("height_gradient")
    curvature = ALL_FEATURES.index("height_curvature")
    assert np.max(abs(old[..., gradient][valid]-expected[..., gradient][valid])) > 1e-6
    assert np.max(abs(old[..., curvature][valid]-expected[..., curvature][valid])) > 1e-10

    read = rf._feature_window(core, width=25, height=25, local_radius=2)
    assert read == Window(5, 5, 14, 14)
    corrected, valid = _raster_core_features(arrays, masks, core, read, 2)
    np.testing.assert_allclose(corrected[..., gradient][valid], expected[..., gradient][valid], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(corrected[..., curvature][valid], expected[..., curvature][valid], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("radius", [0, 1, 2, 3])
@pytest.mark.parametrize("block_size", [1, 4, 8, 32])
@pytest.mark.parametrize("surface", ["plane", "quadratic"])
def test_raster_tiled_features_match_whole(radius, block_size, surface):
    from src.v2 import raster_features as rf

    arrays, masks = _raster_feature_surface(surface=surface)
    height, width = arrays["radar"].shape
    full = {**arrays, **derive_local_features(arrays, masks, radius, 2025, 2025)}
    whole = np.stack([full[name] for name in ALL_FEATURES], axis=-1)
    expected_valid = np.logical_and.reduce(list(masks.values()))
    tiled = np.empty_like(whole)
    tiled_valid = np.empty_like(expected_valid)
    visits = np.zeros((height, width), dtype=np.uint8)
    for row, col, core in rf.iter_blocks(width, height, block_size):
        read = rf._feature_window(core, width, height, radius)
        assert read.col_off >= 0 and read.row_off >= 0
        assert read.col_off + read.width <= width
        assert read.row_off + read.height <= height
        values, valid = _raster_core_features(arrays, masks, core, read, radius)
        h, w = int(core.height), int(core.width)
        tiled[row:row+h, col:col+w] = values
        tiled_valid[row:row+h, col:col+w] = valid
        visits[row:row+h, col:col+w] += 1
    np.testing.assert_array_equal(visits, 1)  # No gaps, duplicate cores, or halo outputs.
    np.testing.assert_array_equal(tiled_valid, expected_valid)
    assert len(ALL_FEATURES) == 17 and tiled.shape == (height, width, 17)
    assert np.isfinite(tiled[expected_valid]).all()
    for i, name in enumerate(ALL_FEATURES):
        if name in arrays:
            np.testing.assert_array_equal(tiled[..., i], whole[..., i])
        else:
            # Float64 filters accumulate over different array lengths in each tile.
            # Texture tolerances are in stored-value units; scaled derivatives get
            # a tighter absolute tolerance so the known halo discrepancy fails.
            atol = 1e-12 if name in ("height_gradient", "radar_gradient", "height_curvature") else 1e-8
            np.testing.assert_allclose(
                tiled[..., i][expected_valid], whole[..., i][expected_valid],
                rtol=1e-10, atol=atol, err_msg=name,
            )


@pytest.mark.parametrize("surface", ["plane", "quadratic"])
def test_raster_known_surface_derivatives(surface):
    arrays, masks = _raster_feature_surface(surface=surface, missing=False)
    derived = derive_local_features(arrays, masks, 2, 2025, 2025)
    row, col = 7, 9
    gx, gy = (2, 3) if surface == "plane" else (2*col+3, 4*row-4)
    np.testing.assert_allclose(derived["height_gradient"][row, col], np.hypot(gx, gy)/2025, atol=1e-12)
    np.testing.assert_allclose(derived["radar_gradient"][row, col], np.hypot(5, -2)/2025, atol=1e-12)
    expected = 0 if surface == "plane" else 6/2025**2
    np.testing.assert_allclose(derived["height_curvature"][row, col], expected, atol=1e-12)
    np.testing.assert_allclose(
        derived["height_local_std"][row, col],
        arrays["height"][row-2:row+3, col-2:col+3].std(), atol=1e-10,
    )


def test_raster_training_and_mapping_use_shared_feature_window():
    import ast
    from pathlib import Path
    from src.v2 import raster_features as rf

    paths = [Path(rf.__file__), Path(rf.__file__).with_name("full_map.py")]
    functions = ["build_advanced_training_sample", "predict_full_v2_maps"]
    for path, name in zip(paths, functions):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = [
            n for n in ast.walk(function) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_feature_window"
        ]
        assert len(calls) == 1


def test_raster_validity_rules_and_feature_order_unchanged():
    from src.v2 import raster_features as rf

    np.testing.assert_array_equal(
        rf._safe_nodata_mask(np.array([0., 1., np.nan, np.inf, -32768.]), -32768),
        [True, True, False, False, False],
    )
    assert rf._safe_nodata_mask(np.array([0.]), None).item()
    assert not rf._safe_nodata_mask(np.array([0.]), 0).item()
    assert ALL_FEATURES == [
        "radar", "height", "emissivity", "slope", "reflectivity",
        "radar_local_std", "height_local_std", "emissivity_local_std",
        "slope_local_std", "reflectivity_local_std", "radar_local_range",
        "height_local_range", "emissivity_local_range", "reflectivity_local_range",
        "height_curvature", "radar_gradient", "height_gradient",
    ]
