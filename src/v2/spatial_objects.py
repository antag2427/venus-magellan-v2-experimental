from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage


def extract_coarse_objects(
    cluster_raster: str | Path,
    output_csv: str | Path,
    max_dim: int = 2048,
    nodata: int = 255,
    min_pixels: int = 8,
) -> dict[str, int | float | str]:
    """Build a coarse object/candidate atlas from a categorical cluster raster.

    This intentionally downsamples the raster before connected components. It is a
    candidate-region atlas, not a replacement for exact full-resolution topology.
    """
    with rasterio.open(cluster_raster) as src:
        scale = max(src.width, src.height) / max_dim
        scale = max(scale, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.nearest,
        )
        pixel_x = abs(src.res[0]) * (src.width / out_width)
        pixel_y = abs(src.res[1]) * (src.height / out_height)

    records: list[dict[str, float | int]] = []
    structure = np.ones((3, 3), dtype=np.int8)
    object_id = 0
    for cluster in sorted(int(x) for x in np.unique(data) if 0 <= x <= 3):
        mask = data == cluster
        labels, count = ndimage.label(mask, structure=structure)
        for local_id in range(1, count + 1):
            coords = np.argwhere(labels == local_id)
            n = len(coords)
            if n < min_pixels:
                continue
            object_id += 1
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            boundary = ndimage.binary_dilation(labels == local_id, structure=structure) & ~(labels == local_id)
            records.append({
                "object_id": object_id,
                "cluster": cluster,
                "coarse_pixels": int(n),
                "approx_area_m2": float(n * pixel_x * pixel_y),
                "approx_perimeter_m": float(boundary.sum() * 0.5 * (pixel_x + pixel_y)),
                "row_min": int(r0),
                "row_max": int(r1),
                "col_min": int(c0),
                "col_max": int(c1),
                "centroid_row": float(coords[:, 0].mean()),
                "centroid_col": float(coords[:, 1].mean()),
            })

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return {
        "objects": int(len(records)),
        "clusters_present": int(pd.DataFrame(records)["cluster"].nunique()) if records else 0,
        "output": str(output),
        "coarse_width": int(out_width),
        "coarse_height": int(out_height),
    }
