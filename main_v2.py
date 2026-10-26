from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.paths import K4_RASTER
from src.v2.anomaly import run_anomaly_detection
from src.v2.config import V2Config, ALL_FEATURES
from src.v2.routing import resolved_inputs, inspect_rasters, planned_outputs, validate_output_routes, planned_stage_files
from src.v2.consensus import run_consensus
from src.v2.discovery import kmeans_stability
from src.v2.feature_diagnostics import run_feature_diagnostics
from src.v2.geology_association import discover_and_run_associations
from src.v2.full_map import build_full_v2_maps
from src.v2.report import write_json, write_stage_report, stage_run, require_inputs, ensure_outputs_new
from src.v2.raster_features import build_advanced_training_sample, BASE_PATHS, aligned_rasters, plan_core_window
from src.v2.model_artifacts import validate_features, sha256_file, preprocessing_config
from src.v2.spatial_objects import extract_coarse_objects
from src.v2.uncertainty import fit_distance_uncertainty
from src.v2.visualization import (
    save_anomaly_distribution, save_confidence_histogram,
    save_consensus_agreement, save_top_anomalies,
)

PROJECT_ROOT = Path(__file__).resolve().parent
V2_PROCESSED = PROJECT_ROOT / "data" / "processed" / "v2"
V2_CLUSTERING = PROJECT_ROOT / "outputs" / "v2" / "clustering"
V2_ANALYSIS = PROJECT_ROOT / "outputs" / "v2" / "analysis"
V2_SPATIAL = PROJECT_ROOT / "outputs" / "v2" / "spatial"
V2_FIGURES = PROJECT_ROOT / "figures" / "v2"
CONFIG_PATH = PROJECT_ROOT / "configs" / "v2_default.json"


def load_config() -> V2Config:
    return V2Config.from_json(CONFIG_PATH) if CONFIG_PATH.exists() else V2Config()


def _directories(cfg):
    return planned_outputs(cfg,PROJECT_ROOT,
        legacy=(V2_PROCESSED,V2_CLUSTERING,V2_ANALYSIS,V2_SPATIAL,V2_FIGURES))


def _layout(cfg):
    paths = _directories(cfg)
    return tuple(paths[name] for name in ("processed","clustering","analysis","spatial","figures"))


def _raster_options(cfg):
    result = {}
    if cfg.raster_inputs is not None:
        result["raster_inputs"] = cfg.raster_inputs
    if cfg.core_window is not None:
        result.update(core_window=cfg.core_window,halo_policy=cfg.halo_policy)
    return result


def _source_context(cfg):
    paths = resolved_inputs(cfg,BASE_PATHS)
    metadata = _require_rasters() if cfg.raster_inputs is None else inspect_rasters(paths)
    validate_output_routes(cfg,_directories(cfg),paths)
    window = None
    if cfg.core_window is not None:
        ctx, radar, sources = aligned_rasters(paths)
        with ctx:
            window = plan_core_window(radar,cfg.core_window,local_radius=cfg.local_radius,
                                      halo_policy=cfg.halo_policy).to_dict()
    return {"paths":paths,"metadata":metadata,"window":window}


def _source_files(context):
    return list(context["paths"].values()) if context is not None and context["metadata"] is not None else []


def _record_context(run,context):
    if context is not None:
        run["raster_inputs"] = context["metadata"]
        run["window"] = context["window"]
        run["feature_preprocessing"] = preprocessing_config(run["configuration"]["local_radius"])



def _configured_routing(cfg):
    return cfg.raster_inputs is not None or cfg.output_root is not None or cfg.core_window is not None


def _completed_provenance(directory, stage):
    path = directory/f"{stage}_run.json"
    if path.with_name(path.name+".pending").exists():
        raise ValueError(f"Incomplete {stage} provenance: {path}")
    require_inputs(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["stage"] != stage or record["status"] != "complete":
            raise ValueError("not complete")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid {stage} provenance: {path}") from exc
    return path,record


def _sample_provenance(cfg, sample, processed, context):
    if not _configured_routing(cfg):
        return []
    path, record = _completed_provenance(processed,"stage1")
    try:
        same_context = (record["raster_inputs"] == context["metadata"] and
                        record["window"] == context["window"] and
                        record["feature_preprocessing"] == preprocessing_config(cfg.local_radius) and
                        all(record["configuration"][name] == getattr(cfg,name)
                            for name in ("local_radius","block_size","core_window","halo_policy")))
        same_sample = (record["results"]["sample_sha256"] == sha256_file(sample) and
                       Path(record["results"]["feature_build"]["output"]).resolve() == sample.resolve())
        if not same_context or not same_sample:
            raise ValueError("source, core, preprocessing or sample changed")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Feature sample provenance does not match this run: {path}") from exc
    return [path]


def _discovery_provenance(cfg, sample, clustering, feature_provenance):
    if not _configured_routing(cfg):
        return []
    path,record = _completed_provenance(clustering,"stage2")
    try:
        hashes = {Path(item["path"]).resolve():item["sha256"] for item in record["inputs"]}
        if hashes.get(sample.resolve()) != sha256_file(sample):
            raise ValueError("discovery sample changed")
        if any(hashes.get(p.resolve()) != sha256_file(p) for p in feature_provenance):
            raise ValueError("upstream feature run changed")
        if record["configuration"]["k"] != cfg.k:
            raise ValueError("discovery cluster count changed")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Discovery provenance does not match this run: {path}") from exc
    return [path]


def _sample_input(path, k):
    import pandas as pd
    import numpy as np
    require_inputs(path)
    table = pd.read_csv(path, float_precision="round_trip")
    missing = [name for name in ALL_FEATURES if name not in table]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = validate_features(table[ALL_FEATURES].to_numpy(dtype=np.float64), ALL_FEATURES)
    if len(X) < k:
        raise ValueError(f"Advanced sample needs at least {k} rows: {path}")


def _require_rasters():
    return inspect_rasters(BASE_PATHS)


def stage1(cfg: V2Config) -> None:
    processed, _, _, _, _ = _layout(cfg)
    sample = processed / "venus_advanced_training_sample.csv"
    diagnostics_dir = processed / "diagnostics"
    summary = processed / "stage1_summary.json"
    report = processed / "stage1_report.md"
    context = _source_context(cfg)
    with stage_run(processed, "stage1", cfg, _source_files(context), [sample, diagnostics_dir, summary, report]) as run:
        _record_context(run,context)
        result = build_advanced_training_sample(
            sample, sample_size=cfg.sample_size, random_state=cfg.random_state,
            block_size=cfg.block_size, local_radius=cfg.local_radius,
            sample_min_candidate_factor=cfg.sample_min_candidate_factor,
            sample_batch_factor=cfg.sample_batch_factor, **_raster_options(cfg),
        )
        diagnostics = run_feature_diagnostics(sample, diagnostics_dir, output_prefix=cfg.output_prefix)
        payload = {"run_id": run["run_id"], "feature_build": result, "diagnostics": diagnostics,
                   "source_paths": {name: str(path) for name, path in context["paths"].items()}}
        payload["sample_sha256"] = sha256_file(sample)
        write_json(summary, payload)
        write_stage_report(processed, "Stage 1 — Planetary Feature Expansion",
                           [("Feature build", result), ("Feature diagnostics", diagnostics)], report.name)
        run["results"] = payload
    print(json.dumps(payload, indent=2))


def stage2(cfg: V2Config) -> None:
    processed, clustering, _, _, _ = _layout(cfg)
    sample = processed / "venus_advanced_training_sample.csv"
    _sample_input(sample, cfg.k)
    context = _source_context(cfg) if cfg.raster_inputs is not None or cfg.core_window is not None or cfg.output_root is not None else None
    upstream = _sample_provenance(cfg,sample,processed,context)
    prefix = cfg.output_prefix
    names = [f"{prefix}_{name}" for name in ("consensus_pairwise_ari.csv", "consensus_labels.csv",
             "advanced_scaler.csv", "consensus_summary.csv", "label_mappings.json",
             "kmeans_stability_pairs.csv")] + ["stage2_summary.json", "stage2_report.md"]
    outputs = [clustering / name for name in names]
    with stage_run(clustering, "stage2", cfg, [sample]+_source_files(context)+upstream, outputs) as run:
        _record_context(run,context)
        result = run_consensus(
            sample, clustering, k=cfg.k, subset_size=cfg.consensus_subset,
            random_state=cfg.random_state, n_init=cfg.kmeans_n_init,
            max_iter=cfg.kmeans_max_iter, gmm_n_init=cfg.gmm_n_init,
            gmm_max_iter=cfg.gmm_max_iter, birch_threshold=cfg.birch_threshold,
            minibatch_batch_size=cfg.minibatch_batch_size, output_prefix=prefix,
        )
        stability = kmeans_stability(
            sample, clustering / f"{prefix}_kmeans_stability_pairs.csv",
            k=cfg.k, n_init=cfg.kmeans_n_init, max_iter=cfg.kmeans_max_iter,
        )
        payload = {"run_id": run["run_id"], "consensus": result, "stability": stability}
        write_json(clustering / "stage2_summary.json", payload)
        write_stage_report(clustering, "Stage 2 — Multi-Model Surface Regime Discovery",
                           [("Consensus", result), ("K-Means stability", stability)], "stage2_report.md")
        run["results"] = payload
    print(json.dumps(payload, indent=2))


def stage3(cfg: V2Config) -> None:
    processed, clustering, analysis, spatial, figures = _layout(cfg)
    sample = processed / "venus_advanced_training_sample.csv"
    prefix = cfg.output_prefix
    consensus_csv = clustering / f"{prefix}_consensus_labels.csv"
    require_inputs(sample, consensus_csv)
    if (clustering / "stage2_run.json.pending").exists():
        raise ValueError(f"Discovery stage is incomplete: {clustering / 'stage2_run.json.pending'}")
    import pandas as pd
    import numpy as np
    try:
        agreement = pd.read_csv(consensus_csv, usecols=["consensus_agreement"],
                                float_precision="round_trip")["consensus_agreement"].to_numpy(dtype=float)
        if not len(agreement) or not np.isfinite(agreement).all() or np.any((agreement < 0) | (agreement > 1)):
            raise ValueError("consensus_agreement must be finite and in [0, 1].")
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Invalid consensus_agreement input: {consensus_csv}") from exc
    _sample_input(sample, cfg.k)
    context = _source_context(cfg)
    upstream = _sample_provenance(cfg,sample,processed,context)
    upstream += _discovery_provenance(cfg,sample,clustering,upstream)
    use_legacy_atlas = not _configured_routing(cfg) and Path(K4_RASTER).exists()
    uncertainty_csv = analysis / f"{prefix}_cluster_uncertainty.csv"
    anomaly_csv = analysis / f"{prefix}_anomaly_candidates.csv"
    anomaly_reference = analysis / f"{prefix}_anomaly_candidates_reference"
    rasters = _directories(cfg)["rasters"]
    plot_paths = [figures / f"{prefix}_{name}.png" for name in
                  ("confidence_distribution", "anomaly_distribution", "consensus_agreement", "top_anomalies")]
    object_path = spatial / f"{prefix}_coarse_object_atlas.csv"
    outputs = [uncertainty_csv, anomaly_csv, anomaly_reference, rasters,
               analysis / "geology_associations", analysis / "stage3_summary.json",
               analysis / "stage3_report.md", *plot_paths]
    if use_legacy_atlas:
        outputs.append(object_path)
    with stage_run(analysis, "stage3", cfg, [sample, consensus_csv]+_source_files(context)+upstream+([Path(K4_RASTER)] if use_legacy_atlas else []), outputs) as run:
        _record_context(run,context)
        uncertainty = fit_distance_uncertainty(
            sample, uncertainty_csv, k=cfg.k, random_state=cfg.random_state,
            n_init=cfg.kmeans_n_init, max_iter=cfg.kmeans_max_iter,
        )
        anomaly = run_anomaly_detection(
            sample, anomaly_csv, k=cfg.k, random_state=cfg.random_state,
            contamination=cfg.anomaly_contamination, n_estimators=cfg.isolation_estimators,
            fit_subset=cfg.anomaly_subset, local_radius=cfg.local_radius,
            n_init=cfg.kmeans_n_init, max_iter=cfg.kmeans_max_iter,
        )
        save_confidence_histogram(uncertainty_csv, plot_paths[0])
        save_anomaly_distribution(anomaly_csv, plot_paths[1])
        save_consensus_agreement(consensus_csv, plot_paths[2])
        save_top_anomalies(anomaly_csv, plot_paths[3])
        associations = discover_and_run_associations(
            _directories(cfg)["validation"], analysis / "geology_associations")
        objects = None
        if use_legacy_atlas:
            objects = extract_coarse_objects(K4_RASTER, object_path,
                                             max_dim=cfg.object_downsample_max_dim)
        # This exploratory orchestration explicitly fits a NEW production artifact.
        # Inference-only callers use predict_full_v2_maps with existing artifacts.
        full_map = build_full_v2_maps(
            sample, rasters, k=cfg.k, random_state=cfg.random_state,
            block_size=cfg.block_size, local_radius=cfg.local_radius,
            isolation_contamination=cfg.anomaly_contamination,
            isolation_estimators=cfg.isolation_estimators,
            isolation_fit_subset=cfg.anomaly_subset,
            anomaly_reference=anomaly["reference_artifact"],
            output_prefix=prefix, n_init=cfg.kmeans_n_init, max_iter=cfg.kmeans_max_iter,
            **_raster_options(cfg),
        )
        payload = {"run_id": run["run_id"], "uncertainty": uncertainty, "anomaly": anomaly,
                   "geology_associations": associations, "coarse_objects": objects, "full_map": full_map}
        write_json(analysis / "stage3_summary.json", payload)
        write_stage_report(analysis, "Stage 3 — Uncertainty, Anomaly, and Spatial Candidate Discovery",
                           [("Uncertainty", uncertainty), ("Anomaly detection", anomaly),
                            ("Geology association tables", {"compatible_tables": len(associations)}),
                            ("Coarse spatial objects", objects or {"status": "skipped", "reason": "Legacy V1 atlas absent or not applicable to explicitly routed inputs"}),
                            ("Full-map V2 products", full_map)], "stage3_report.md")
        run["results"] = payload
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Venus / Magellan V2 expansion pipeline")
    parser.add_argument("stage", choices=["features", "discover", "interpret", "all"])
    parser.add_argument("--config", type=Path, help="Explicit configuration JSON; unknown fields are errors.")
    args = parser.parse_args()
    cfg = V2Config.from_json(args.config) if args.config else load_config()
    if cfg.raster_inputs is not None or cfg.output_root is not None or cfg.core_window is not None:
        _source_context(cfg)
        if args.stage == "all":
            ensure_outputs_new(*planned_stage_files(cfg,_directories(cfg)).values())
    if args.stage in ("features", "all"):
        stage1(cfg)
    if args.stage in ("discover", "all"):
        stage2(cfg)
    if args.stage in ("interpret", "all"):
        stage3(cfg)


if __name__ == "__main__":
    main()
