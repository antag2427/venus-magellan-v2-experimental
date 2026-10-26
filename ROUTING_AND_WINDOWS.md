# V2 standalone input routing and target-grid core windows

This upgrade pack supplies its own minimal src/paths.py. Importing the package does
not read or modify the original project. With no new options, defaults remain
relative to the project containing src/paths.py: data/raw/venus*.tif, the existing
V2 stage directories, and the optional legacy K4 classification.

## Configuration

The existing strict V2Config accepts four optional fields:

- raster_inputs: either null or a mapping containing exactly radar, height,
  emissivity, slope, reflectivity. Every value must be a nonempty path string.
- output_root: null for legacy directories, otherwise an explicit directory.
  Empty strings, '.', '..', filesystem roots and drive-relative paths are rejected.
- core_window: null for full-grid behavior, otherwise exactly row, col, height,
  width. Values are integers; offsets are nonnegative and dimensions positive.
  Core mode requires local_radius=2.
- halo_policy: 'clip' (default) or 'require_full'.

Relative input and output paths in a JSON file resolve from that JSON file's
directory, independently of the launch working directory. Programmatic V2Config
construction resolves relative paths from the working directory at construction.
Tilde expansion uses the current user's home. Config values are normalized to
absolute paths before manifests are written. Unknown top-level or nested keys,
missing required inputs, non-raster files and unusable georeferencing are errors.

Example (paths are illustrative and must exist before execution):

    {
      "raster_inputs": {
        "radar": "inputs/venus.tif",
        "height": "inputs/venus_height.tif",
        "emissivity": "inputs/venus_emissivity.tif",
        "slope": "inputs/venus_slope.tif",
        "reflectivity": "inputs/venus_reflectivity.tif"
      },
      "output_root": "runs/isolated_pilot",
      "output_prefix": "pilot_001",
      "core_window": {"row": 1000, "col": 2000, "height": 256, "width": 256},
      "halo_policy": "require_full",
      "local_radius": 2,
      "sample_size": 10000,
      "consensus_subset": 5000,
      "anomaly_subset": 5000
    }

The command remains: python main_v2.py all --config path/to/config.json.
It executes work. Loading a configuration and calling the planning/inspection
APIs below does not execute the pipeline.

## One output layout

routing.planned_outputs returns the authoritative stage directories. With an
output_root, ALL generated stage files are below:

    output_root/output_prefix/
        processed/      # sample, diagnostics, stage1 reports and manifest
        clustering/     # consensus, stability, stage2 reports and manifest
        analysis/       # uncertainty, anomaly/reference, associations, stage3 reports
        figures/
        models/         # exclusive UUID-named classifier artifacts
        rasters/        # class, confidence, anomaly, centroid/scaler exports, association
        spatial/        # reserved optional atlas outputs
        validation/     # optional existing geology count-table discovery input

planned_stage_files expands known filenames; classifier UUID child names are
reserved exclusively when fitting. validate_output_routes normalizes paths,
checks containment (including resolved symlinks), rejects blocked parents and
overlap with input rasters. Pending and existing final destinations are rejected
before execution. Stage completion is published only after successful return.

Without output_root the historical data/processed, outputs and figures layout is
retained. output_prefix continues selecting names/namespaces in either mode.
Direct module APIs still accept explicit output paths; callers outside root
orchestration must themselves select their authorized destination.

Explicitly routed runs do not automatically read a default V1 classification
for coarse-object extraction. That atlas is unbound to their configured inputs.
The legacy default-input behavior is retained, with its input hash recorded if
the atlas is used. Core-window runs skip it.

## Authoritative alignment and window policy

The radar TIFF defines the target CRS, transform, dimensions and pixel grid.
aligned_rasters(raster_inputs=None) still uses the established companion
WarpedVRT settings: radar grid, bilinear resampling, float64 destination,
native source NoData, destination NoData=-32768. Radar values are not resampled.
Never apply radar row/column offsets directly to each companion's native grid.

plan_core_window(radar, core_window, halo_policy=...) expands the core with the
existing _feature_window dependency radius: local_radius + 1 = 3 target pixels.
An interior 256x256 core therefore has a 262x262 context. The returned plan records
global core/context offsets, transforms, bounds, actual four halo widths,
whether the halo is complete, and the canonical grid metadata.

'clip' clips context at the actual radar-grid edge and reports which halo sides
are shortened. Existing one-sided/nearest physical-edge feature handling is
unchanged. 'require_full' rejects any shortened halo. Neither policy changes
interpolation or invents neighboring data. Per-source missing coverage/NoData is
distinct from canonical-grid clipping.

read_core_context(radar, aligned_sources, plan, minimum_common_valid=...) reads
the SAME target-grid context from all aligned sources and returns arrays, masks,
core validity, and per-feature/common coverage counts. Fewer common-valid core
pixels than requested raises a coverage error. NoData holes remain masked.
Arrays use the existing float64 feature-reader representation; byte radar values
are unchanged. No context TIFF writer or new resampler is introduced.

Core slices start at plan.core_row_in_context and plan.core_col_in_context
(normally 3,3). Core features match their counterparts in a larger aligned read
within floating filter accumulation tolerance. The feature formulas and ALL_FEATURES
ordering are unchanged.

## Sampling, mapping, and coordinates

In configured core mode, the valid-count pass and stratified sampling traverse
only the core. Halo pixels supply feature dependencies and are never extra sample
observations. Source reads for each core block use the same r+1 halo helper.
Saved CSV row and col remain GLOBAL radar-grid indices.

Mapping likewise visits only core blocks but computes their features with
original-grid context. Its outputs have core dimensions and the core transform;
raster row/col = CSV global row/col minus the core's global offsets. Halo pixels
are not classified as extra output pixels. Cluster output remains uint8 with
NoData=255; confidence/anomaly remain float32 with NoData=-9999.

With core_window=null the previous whole-grid traversal, profiles, labels, masks
and feature calculations are retained. No new reference grid is selected.

## Provenance and compatibility

Configured downstream stages require compatible completed upstream provenance.
Stage1 records source absolute paths/content hashes, the target/core/context plan,
feature preprocessing fingerprint and the saved CSV hash. Stage2 refuses changed
sources, core/context, local radius, block size, preprocessing or sample bytes.
Stage3 additionally verifies the completed discovery stage's sample and upstream
manifest hashes and cluster count. Models remain distinct experiments; their IDs
are associated through stage results and existing artifact manifests.

Model/artifact schemas and anomaly formulas are unchanged. Existing artifacts
are not rewritten or migrated. Strict source/preprocessing fingerprint checks can
reject older artifacts after source changes; there is no silent fallback parser,
migration, refitting or reference substitution during inference.

## Safe preflight

Use inspect_rasters for metadata and SHA-256, planned_outputs/planned_stage_files
for destinations, validate_output_routes for isolation, and aligned_rasters plus
plan_core_window/read_core_context for bounded coverage. These APIs do not fit
models or write extracted rasters. A future 256x256 pilot should require at least
20,000 common-valid core pixels and use the first qualifying candidate from a
documented deterministic list, independent of clusters or geological results.

A successful read-only preflight is a plan approval, not an executed pilot or
evidence of geological validity.
