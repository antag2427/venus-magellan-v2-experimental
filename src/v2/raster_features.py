from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform, bounds as window_bounds
from affine import Affine

from src.paths import (
    EMISSIVITY_FILE,
    HEIGHT_FILE,
    RADAR_FILE,
    REFLECTIVITY_FILE,
    SLOPE_FILE,
)

from .config import ALL_FEATURES, BASE_FEATURES, validate_sampling_factors, validate_core_window
from .report import ensure_new, ensure_outputs_new, write_csv


BASE_PATHS = {
    "radar": RADAR_FILE,
    "height": HEIGHT_FILE,
    "emissivity": EMISSIVITY_FILE,
    "slope": SLOPE_FILE,
    "reflectivity": REFLECTIVITY_FILE,
}


def _safe_nodata_mask(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def _fill_invalid_with_local_mean(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    kernel = 2 * radius + 1
    weight = valid.astype(np.float64)
    safe = np.where(valid, data, 0.0)
    numerator = ndimage.uniform_filter(safe, size=kernel, mode="nearest")
    denominator = ndimage.uniform_filter(weight, size=kernel, mode="nearest")
    local_mean = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    fallback = np.nanmedian(data[valid]) if np.any(valid) else 0.0
    filled = np.where(valid, data, local_mean)
    filled = np.where(np.isfinite(filled), filled, fallback)
    return filled


def _local_std(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Masked population std, with nearest edges and zero for empty neighborhoods.

    Accumulate centered moments in a fixed neighborhood order, independently for
    each pixel. A valid neighborhood member anchors the subtraction, avoiding
    large DC offsets and context-dependent running sums from uniform filtering.
    The loops visit offsets, never individual pixels.
    """
    data = np.asarray(data, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    height, width = data.shape
    padded = np.pad(data, radius, mode="edge")
    padded_valid = np.pad(valid, radius, mode="edge")
    anchor = np.where(valid, data, 0.0)
    found = valid.copy()
    count = np.zeros_like(data)
    total = np.zeros_like(data)
    second = np.zeros_like(data)
    work = np.zeros_like(data)
    for row in range(2 * radius + 1):
        for col in range(2 * radius + 1):
            values = padded[row:row + height, col:col + width]
            good = padded_valid[row:row + height, col:col + width]
            np.copyto(anchor, values, where=good & ~found)
            found |= good
            work.fill(0.0)
            np.subtract(values, anchor, out=work, where=good)
            count += good
            total += work
            np.square(work, out=work)
            second += work
    np.divide(total, count, out=total, where=count > 0)
    np.divide(second, count, out=second, where=count > 0)
    np.square(total, out=total)
    second -= total
    np.maximum(second, 0.0, out=second)
    return np.sqrt(second, out=second)


def _local_range(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    high = np.where(valid, data, -np.inf)
    low = np.where(valid, data, np.inf)
    high = ndimage.maximum_filter(high, size=size, mode="nearest")
    low = ndimage.minimum_filter(low, size=size, mode="nearest")
    out = high - low
    out[~np.isfinite(out)] = 0.0
    return out


def derive_local_features(
    arrays: Dict[str, np.ndarray],
    valid_masks: Dict[str, np.ndarray],
    radius: int,
    pixel_size_x: float,
    pixel_size_y: float,
) -> Dict[str, np.ndarray]:
    """Compute local derivatives while respecting per-raster validity masks."""
    out: Dict[str, np.ndarray] = {}
    for name in BASE_FEATURES:
        data = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(valid_masks[name], dtype=bool)
        out[f"{name}_local_std"] = _local_std(data, valid, radius)

    for name in ("radar", "height", "emissivity", "reflectivity"):
        data = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(valid_masks[name], dtype=bool)
        out[f"{name}_local_range"] = _local_range(data, valid, radius)

    height = _fill_invalid_with_local_mean(
        np.asarray(arrays["height"], dtype=np.float64),
        np.asarray(valid_masks["height"], dtype=bool),
        radius,
    )
    radar = _fill_invalid_with_local_mean(
        np.asarray(arrays["radar"], dtype=np.float64),
        np.asarray(valid_masks["radar"], dtype=bool),
        radius,
    )

    gy_h, gx_h = np.gradient(height, pixel_size_y, pixel_size_x)
    gy_r, gx_r = np.gradient(radar, pixel_size_y, pixel_size_x)
    out["height_gradient"] = np.hypot(gx_h, gy_h)
    out["radar_gradient"] = np.hypot(gx_r, gy_r)

    # A normalized discrete Laplacian is used as a compact curvature proxy.
    lap = ndimage.laplace(height, mode="nearest")
    spacing2 = max((abs(pixel_size_x) + abs(pixel_size_y)) * 0.5, 1.0) ** 2
    out["height_curvature"] = lap / spacing2

    missing = set(ALL_FEATURES) - set(BASE_FEATURES) - set(out)
    if missing:
        raise RuntimeError(f"Feature generation incomplete: {sorted(missing)}")
    return out


def _read_aligned_window(
    stack: dict[str, WarpedVRT | rasterio.io.DatasetReader],
    window: Window,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    arrays: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for name, src in stack.items():
        arr = src.read(1, window=window).astype(np.float64, copy=False)
        valid = _safe_nodata_mask(arr, src.nodata)
        arrays[name] = arr
        masks[name] = valid
    return arrays, masks


def aligned_rasters(raster_inputs=None):
    """One authoritative alignment: float64 bilinear companions on radar's grid."""
    if raster_inputs is not None and (not isinstance(raster_inputs, dict) or set(raster_inputs) != set(BASE_FEATURES)):
        raise ValueError(f"raster_inputs must contain exactly {BASE_FEATURES}.")
    paths = dict(BASE_PATHS, radar=RADAR_FILE) if raster_inputs is None else raster_inputs
    stack = ExitStack()
    try:
        radar = stack.enter_context(rasterio.open(paths["radar"]))
        sources = {"radar":radar}
        for name in ("height", "emissivity", "slope", "reflectivity"):
            src = stack.enter_context(rasterio.open(paths[name]))
            sources[name] = stack.enter_context(WarpedVRT(
                src, crs=radar.crs, transform=radar.transform,
                width=radar.width, height=radar.height,
                resampling=Resampling.bilinear, src_nodata=src.nodata,
                nodata=-32768.0, dtype="float64",
            ))
        return stack, radar, sources
    except Exception:
        stack.close()
        raise


def _feature_window(core_window: Window, width: int, height: int, local_radius: int) -> Window:
    """Expand a core to cover every feature dependency, clipped to the raster.

    Base values need no halo; local standard deviation and range need radius r.
    Height/radar imputation needs r, then gradients and the height Laplacian
    need one adjacent imputed pixel: their composed dependency is r + 1.
    Clipping preserves the existing one-sided/nearest physical-edge rules.
    """
    halo = int(local_radius) + 1
    col0 = max(0, int(core_window.col_off) - halo)
    row0 = max(0, int(core_window.row_off) - halo)
    col1 = min(width, int(core_window.col_off + core_window.width) + halo)
    row1 = min(height, int(core_window.row_off + core_window.height) + halo)
    return Window(col0, row0, col1 - col0, row1 - row0)


def iter_blocks(width: int, height: int, block_size: int) -> Iterator[tuple[int, int, Window]]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            h = min(block_size, height - row)
            w = min(block_size, width - col)
            yield row, col, Window(col, row, w, h)


def _valid_count_pass(radar: rasterio.io.DatasetReader, sources: dict, block_size: int, region=None) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    keys = []
    region = region if region is not None else Window(0,0,radar.width,radar.height)
    for row, col, local in iter_blocks(int(region.width), int(region.height), block_size):
        row += int(region.row_off)
        col += int(region.col_off)
        window = Window(col,row,local.width,local.height)
        arrays, masks = _read_aligned_window(sources, window)
        valid = np.logical_and.reduce([masks[name] for name in BASE_FEATURES])
        counts.append(int(valid.sum()))
        keys.append((row, col, window))
    return np.asarray(counts, dtype=np.int64), np.asarray(keys, dtype=object)


def _allocate_stratified_sample(counts: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    total = int(counts.sum())
    if total < sample_size:
        raise ValueError(f"Only {total:,} valid aligned pixels are available; requested {sample_size:,}.")
    expected = counts.astype(np.float64) * sample_size / total
    allocation = np.floor(expected).astype(np.int64)
    remainder = sample_size - int(allocation.sum())
    if remainder:
        probs = expected - allocation
        order = np.argsort(probs)[::-1]
        allocation[order[:remainder]] += 1
    return allocation


def build_advanced_training_sample(
    output_csv: str | Path,
    sample_size: int = 100_000,
    random_state: int = 42,
    block_size: int = 512,
    local_radius: int = 2,
    *, sample_min_candidate_factor: float = 1.10, sample_batch_factor: float = 1.5,
    raster_inputs=None, core_window=None, halo_policy="clip",
) -> dict[str, int | float | str]:
    """Build a reproducible enriched training table from aligned Venus rasters.

    Sampling is stratified across raster blocks proportional to their valid-pixel count,
    then local feature windows are evaluated only for sampled blocks.
    """
    import pandas as pd

    ensure_outputs_new(output_csv)
    validate_sampling_factors(sample_min_candidate_factor, sample_batch_factor)
    rng = np.random.default_rng(random_state)
    ctx, radar, sources = aligned_rasters() if raster_inputs is None else aligned_rasters(raster_inputs)
    with ctx:
        plan = plan_core_window(radar,core_window,local_radius=local_radius,halo_policy=halo_policy) if core_window is not None else None
        counts, keys = (_valid_count_pass(radar, sources, block_size) if plan is None else
                        _valid_count_pass(radar, sources, block_size, plan.core))
        total_valid = int(counts.sum())
        allocation = _allocate_stratified_sample(counts, sample_size, rng)
        records: list[dict[str, float | int]] = []
        radius = int(local_radius)
        px_x, px_y = abs(radar.res[0]), abs(radar.res[1])

        for block_idx, (_, _, base_window) in enumerate(keys):
            take = int(allocation[block_idx])
            if take == 0:
                continue
            halo_window = _feature_window(base_window, radar.width, radar.height, radius)
            arrays, masks = _read_aligned_window(sources, halo_window)
            valid = np.logical_and.reduce([masks[name] for name in BASE_FEATURES])
            center_row0 = int(base_window.row_off) - int(halo_window.row_off)
            center_col0 = int(base_window.col_off) - int(halo_window.col_off)
            center = valid[center_row0:center_row0 + int(base_window.height), center_col0:center_col0 + int(base_window.width)]
            valid_positions = np.argwhere(center)
            if take > len(valid_positions):
                raise RuntimeError("Block allocation exceeded available valid pixels.")
            chosen_idx = rng.choice(len(valid_positions), size=take, replace=False)
            chosen = valid_positions[chosen_idx]

            derived = derive_local_features(arrays, masks, radius, px_x, px_y)
            for r, c in chosen:
                rr = center_row0 + int(r)
                cc = center_col0 + int(c)
                global_row = int(base_window.row_off) + int(r)
                global_col = int(base_window.col_off) + int(c)
                record: dict[str, float | int] = {"row": global_row, "col": global_col}
                for name in BASE_FEATURES:
                    record[name] = float(arrays[name][rr, cc])
                for name in derived:
                    record[name] = float(derived[name][rr, cc])
                records.append(record)

    df = pd.DataFrame.from_records(records)
    if len(df) != sample_size:
        raise RuntimeError(f"Expected {sample_size:,} samples but built {len(df):,}.")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, output, index=False, float_format="%.17g")
    return {
        "samples": int(len(df)),
        "total_valid_aligned_pixels": total_valid,
        "feature_count": len([c for c in df.columns if c not in {"row", "col"}]),
        "output": str(output),
        "window": plan.to_dict() if plan is not None else None,
    }


@dataclass(frozen=True)
class CoreWindowPlan:
    core: Window
    context: Window
    core_transform: object
    context_transform: object
    core_row_in_context: int
    core_col_in_context: int
    halo_complete: bool
    halo: dict
    target_grid: dict
    halo_policy: str

    def to_dict(self):
        def window(w):
            return {"row": int(w.row_off), "col": int(w.col_off),
                    "height": int(w.height), "width": int(w.width)}
        return {"core": window(self.core), "context": window(self.context),
                "core_transform": list(self.core_transform),
                "context_transform": list(self.context_transform),
                "core_bounds": list(window_bounds(self.core, Affine(*self.target_grid["transform"][:6]))),
                "context_bounds": list(window_bounds(self.context, Affine(*self.target_grid["transform"][:6]))),
                "core_row_in_context": self.core_row_in_context,
                "core_col_in_context": self.core_col_in_context,
                "halo_complete": self.halo_complete, "halo": self.halo,
                "halo_policy": self.halo_policy, "target_grid": self.target_grid,
                "coordinate_policy": "CSV row/col are global radar-grid indices; output raster indices subtract core row/col."}


def plan_core_window(radar, core_window, *, local_radius=2, halo_policy="clip"):
    """Plan on the original radar grid; no raster reads or new resampling."""
    validate_core_window(core_window)
    if core_window is None:
        raise ValueError("An explicit core_window is required for window planning.")
    if local_radius != 2:
        raise ValueError("Core/halo policy requires local_radius=2.")
    if halo_policy not in ("clip", "require_full"):
        raise ValueError("Unknown halo_policy.")
    core = Window(core_window["col"], core_window["row"], core_window["width"], core_window["height"])
    if core.col_off+core.width > radar.width or core.row_off+core.height > radar.height:
        raise ValueError("core_window exceeds radar target-grid bounds.")
    context = _feature_window(core, radar.width, radar.height, local_radius)
    halo = {"top": int(core.row_off-context.row_off),
            "left": int(core.col_off-context.col_off),
            "bottom": int(context.row_off+context.height-core.row_off-core.height),
            "right": int(context.col_off+context.width-core.col_off-core.width)}
    complete = all(value == local_radius+1 for value in halo.values())
    if halo_policy == "require_full" and not complete:
        raise ValueError(f"Incomplete context halo at target-grid boundary: {halo}")
    return CoreWindowPlan(core, context, window_transform(core, radar.transform),
        window_transform(context, radar.transform), halo["top"], halo["left"], complete, halo,
        {"width":radar.width,"height":radar.height,"transform":list(radar.transform),
         "crs":radar.crs.to_wkt() if radar.crs else None}, halo_policy)


def read_core_context(radar, sources, plan, *, minimum_common_valid=0):
    """Read the same target-grid context from all authoritative aligned sources.

    Missing native coverage and native NoData remain masked, not extrapolated.
    clip applies only at radar-grid edges; per-source invalid coverage is always
    reported independently. No input or extracted raster is written.
    """
    if not isinstance(minimum_common_valid, int) or minimum_common_valid < 0:
        raise ValueError("minimum_common_valid must be a nonnegative integer.")
    arrays, masks = _read_aligned_window(sources, plan.context)
    r,c = plan.core_row_in_context,plan.core_col_in_context
    h,w = int(plan.core.height),int(plan.core.width)
    core_masks = {name: masks[name][r:r+h,c:c+w] for name in BASE_FEATURES}
    common = np.logical_and.reduce(list(core_masks.values()))
    coverage = {"common_valid_core":int(common.sum()), "invalid_core":int(common.size-common.sum()),
                "core_pixels":int(common.size),
                "valid_core_by_feature":{name:int(mask.sum()) for name,mask in core_masks.items()},
                "valid_context_by_feature":{name:int(masks[name].sum()) for name in BASE_FEATURES}}
    if coverage["common_valid_core"] < minimum_common_valid:
        raise ValueError(f"Insufficient common-valid core coverage: {coverage['common_valid_core']} < {minimum_common_valid}")
    return {"plan":plan, "arrays":arrays, "masks":masks,
            "core_valid":common,"coverage":coverage}

