"""One directory routing policy plus read-only raster input inspection."""
from pathlib import Path
import math
import rasterio

from .config import BASE_FEATURES
from .model_artifacts import sha256_file


def resolved_inputs(cfg, defaults):
    return {name: Path((cfg.raster_inputs or defaults)[name]).expanduser().resolve()
            for name in BASE_FEATURES}


def inspect_rasters(paths):
    if not isinstance(paths, dict) or set(paths) != set(BASE_FEATURES):
        raise ValueError(f"raster_inputs must contain exactly {BASE_FEATURES}.")
    result = {}
    for name in BASE_FEATURES:
        path = Path(paths[name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing raster input '{name}': {path}")
        before = path.stat()
        try:
            with rasterio.open(path) as src:
                if src.count < 1 or src.width < 1 or src.height < 1 or src.crs is None:
                    raise ValueError("A georeferenced nonempty raster band is required.")
                if src.transform.determinant == 0 or not all(math.isfinite(x) for x in src.transform):
                    raise ValueError("Raster transform must be finite and invertible.")
                nodata = src.nodata
                if nodata is not None and not math.isfinite(nodata):
                    nodata = str(nodata)
                item = {"path": str(path), "width": src.width, "height": src.height,
                        "crs": src.crs.to_wkt(), "transform": list(src.transform),
                        "bounds": list(src.bounds), "resolution": list(src.res),
                        "dtype": src.dtypes[0], "nodata": nodata, "bands": src.count}
        except (rasterio.errors.RasterioError, ValueError) as exc:
            raise ValueError(f"Invalid raster input '{name}': {path}: {exc}") from exc
        item["sha256"] = sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError(f"Raster input changed during inspection: {path}")
        result[name] = item
    return result


def planned_outputs(cfg, project_root, legacy=None):
    project = Path(project_root).resolve()
    if cfg.output_root is not None:
        base = Path(cfg.output_root) / cfg.output_prefix
        return {name: (base / name).resolve() for name in
                ("processed", "clustering", "analysis", "spatial", "figures", "rasters", "models", "validation")}
    if legacy is None:
        legacy = (project/"data/processed/v2", project/"outputs/v2/clustering",
                  project/"outputs/v2/analysis", project/"outputs/v2/spatial", project/"figures/v2")
    processed, clustering, analysis, spatial, figures = map(Path, legacy)
    if cfg.output_prefix != "v2":
        prefix = cfg.output_prefix
        processed = processed.parent / prefix
        clustering = clustering.parent.parent / prefix / "clustering"
        analysis = analysis.parent.parent / prefix / "analysis"
        spatial = spatial.parent.parent / prefix / "spatial"
        figures = figures.parent / prefix
    return {name: value.resolve() for name,value in {
        "processed":processed,"clustering":clustering,"analysis":analysis,
        "spatial":spatial,"figures":figures,"rasters":analysis.parent/"rasters",
        "models":analysis.parent/"models","validation":project/"outputs/validation"}.items()}


def validate_output_routes(cfg, outputs, inputs):
    protected = [Path(path).resolve() for path in inputs.values()]
    authorized = Path(cfg.output_root).resolve() if cfg.output_root is not None else None
    for name,value in outputs.items():
        path = Path(value).resolve()
        if authorized is not None and not path.is_relative_to(authorized):
            raise ValueError(f"Output escapes output_root: {name}: {path}")
        for ancestor in (path, *path.parents):
            if ancestor.exists() and not ancestor.is_dir():
                raise NotADirectoryError(f"Output parent is not a directory: {ancestor}")
        if any(path == source or path.is_relative_to(source) or source.is_relative_to(path)
               for source in protected):
            raise ValueError(f"Output/input collision: {name}: {path}")


def planned_stage_files(cfg, directories):
    """Known final destinations; model UUID children are reserved exclusively at fit."""
    p = cfg.output_prefix
    d = directories
    result = {
        "sample": d["processed"]/"venus_advanced_training_sample.csv",
        "diagnostics": d["processed"]/"diagnostics",
        "uncertainty": d["analysis"]/f"{p}_cluster_uncertainty.csv",
        "anomaly_candidates": d["analysis"]/f"{p}_anomaly_candidates.csv",
        "anomaly_reference": d["analysis"]/f"{p}_anomaly_candidates_reference",
        "geology_associations": d["analysis"]/"geology_associations",
        "coarse_objects": d["spatial"]/f"{p}_coarse_object_atlas.csv",
        "classifier_container": d["models"],
        "cluster_map": d["rasters"]/f"venus_{p}_clusters_k{cfg.k}.tif",
        "confidence_map": d["rasters"]/f"venus_{p}_confidence.tif",
        "anomaly_map": d["rasters"]/f"venus_{p}_anomaly.tif",
        "map_association": d["rasters"]/f"{p}_production_model.json",
        "centroids": d["rasters"]/f"{p}_kmeans_centroids_standardized.csv",
        "scaler": d["rasters"]/f"{p}_scaler_parameters.csv",
    }
    for name in ("feature_quality.csv", "feature_correlation.csv", "pca_explained_variance.csv"):
        result[name] = d["processed"]/"diagnostics"/f"{p}_{name}"
    for name in ("consensus_pairwise_ari.csv","consensus_labels.csv","advanced_scaler.csv",
                 "consensus_summary.csv","label_mappings.json","kmeans_stability_pairs.csv"):
        result[name] = d["clustering"]/f"{p}_{name}"
    for name in ("confidence_distribution","anomaly_distribution","consensus_agreement","top_anomalies"):
        result[name+"_figure"] = d["figures"]/f"{p}_{name}.png"
    for stage, directory in ((1,d["processed"]),(2,d["clustering"]),(3,d["analysis"])):
        for suffix in ("summary.json","report.md","run.json"):
            result[f"stage{stage}_{suffix}"] = directory/f"stage{stage}_{suffix}"
    return result
