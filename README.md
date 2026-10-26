# Venus Magellan V2 — exploratory source pack

Standalone Python scripts for radar-grid raster alignment, 17-feature surface-property clustering, model consensus, uncertainty and statistical anomaly candidates. V1 source code is not required. Users supply the five original rasters locally; datasets and fitted models are not included in the source release.

Status: exploratory research software. A single approved 256x256 real-data pilot and the lightweight regression suite have passed. This does not establish geological validity, calibrated physical units, full-resolution correctness or general cross-platform support.

## Environment and dependencies

The reviewed local environment is Windows 11, CPython 3.14.3. Runtime versions recorded during review: NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0, Rasterio 1.5.0, affine 2.4.0, Matplotlib 3.10.8. This records a tested environment, not a minimum-version or support matrix. Other platforms and fresh installations have not been verified.

From the pack root, in an environment you manage:

```text
python -m pip install -r requirements-v2.txt
python -m pip install pytest
python main_v2.py --help
python -m pytest tests -q -p no:cacheprovider
```

These are setup instructions, not operations performed by the release review. requirements-v2.txt is unpinned and includes the broader geospatial stack; it is not a lock file. Rasterio supplies affine and scikit-learn supplies threadpoolctl transitively. Tests use pytest and threadpoolctl. No package installation mechanism or binary distribution is supplied.

Tests generate tiny synthetic rasters and fitted states in temporary directories; they do not need the real Venus inputs. In the reviewed environment pytest plugin autoload was disabled and thread counts limited to one. If the default system temporary directory is inaccessible, select an owned temporary test directory with --basetemp; do not point this option at existing data, because pytest manages that directory.

## Configure inputs and an isolated bounded run

Start from [configs/v2_pilot.example.json](configs/v2_pilot.example.json). Supply exactly these raster_inputs keys:

radar, height, emissivity, slope, reflectivity.

The example expects inputs under data/raw and writes under outputs/pilot_example/magellan_pilot_example. It uses the previously tested core row915/col1854, size256x256, which is appropriate only for the corresponding original radar grid. Preflight coverage must be checked for your supplied data. Keep sample counts within valid coverage.

- JSON-relative input paths and output_root resolve from the configuration file's directory, not the launch directory.
- Programmatically constructed V2Config resolves relative paths from the working directory at construction.
- output_root/output_prefix contains all supported orchestration outputs; prefix is a safe filename component.
- core_window=null selects the entire radar grid. The default configuration has no bounded core: do not treat it as a smoke-test configuration.
- core_window requires local_radius=2. The feature dependency halo is three target-grid pixels; a256x256 interior core reads262x262.
- halo_policy=require_full rejects clipped halos; clip permits truncation at the actual radar boundary. It does not fill missing companion coverage.
- Unknown configuration fields are errors.
- sample_min_candidate_factor=1.1 and sample_batch_factor=1.5 are reserved legacy sampling controls. Only these defaults are supported; non-default values are rejected because the exact stratified sampler defines no candidate-batch semantics.

[ROUTING_AND_WINDOWS.md](ROUTING_AND_WINDOWS.md) documents inspect_rasters, planned_outputs, planned_stage_files, validate_output_routes, plan_core_window and read_core_context for preflight. Loading configuration or displaying CLI help does not run processing. There is no CLI dry-run subcommand.

## CLI execution

The following commands perform work and write new outputs. Run only after checking configuration and destinations:

```text
python main_v2.py features --config configs/v2_pilot.example.json
python main_v2.py discover --config configs/v2_pilot.example.json
python main_v2.py interpret --config configs/v2_pilot.example.json
```

Alternatively, use all instead of the three separate commands on a fresh output namespace. Existing or incomplete stage destinations are rejected; do not run all again against a completed run.

- features: bounded/block sampling, local features and diagnostics.
- discover: consensus models and KMeans seed stability.
- interpret: separate uncertainty/anomaly experiments, saved classifier fitting and mapping, plots and optional associations.
- all: those stages in order.

Interpret/all explicitly fit new models; they are not inference-only commands. scripts/run_v2.ps1 is a historical default-layout wrapper with no --config routing; use the explicit Python CLI above for isolated runs. It is not a verified failure-propagating release launcher.

## Outputs, coordinates and inference

With output_root set, all stage files are routed beneath output_root/output_prefix:
processed, clustering, analysis, figures, models, rasters, spatial and validation.

The radar defines CRS/transform/dimensions. Companion rasters use bilinear float64 WarpedVRT alignment, source NoData and destination NoData=-32768. Valid centers require all five base values finite and different from each layer's NoData. Local texture uses valid neighbors, population standard deviation, nearest-edge extension, float64 fixed-order centered accumulation and zero for empty neighborhoods.

CSV row/col are global radar-grid indices. Halo pixels supply context but are not sampled/classified as extra core observations. Bounded output indices subtract core offsets; their affine transform is the core transform. Classification is uint8 with NoData255; confidence/anomaly are float32 with NoData-9999. Feature matrices select the exact ALL_FEATURES order declared in src/v2/config.py; use names, not incidental CSV column positions.

For saved-artifact inference, call src.v2.full_map.predict_full_v2_maps with an explicit saved classifier, the unchanged reference sample, explicit saved anomaly_reference, and the same configuration/raster routing. It loads state and never fits. build_full_v2_maps is a fit-then-map wrapper when no classifier artifact is supplied. There is currently no inference-only CLI subcommand.

Classifier/scaler state and anomaly scaler/KMeans/IsolationForest/ECDF references are separate experiments with distinct identities. Manifests, content hashes, feature order, sample identity, estimator parameters and scikit-learn version are checked. Map inference additionally requires the current preprocessing fingerprint, including the raster_features.py byte hash and Rasterio/GDAL versions. Anomaly loading checks that fingerprint directly. The classifier's generic loader accepts an expected_preprocessing argument; direct callers should supply it when validating the current raster pipeline.

Older artifacts can fail stricter fingerprint checks; none have been migrated. Even source-byte-only changes (including line endings) may invalidate saved preprocessing fingerprints. Preserve the exact source/environment associated with an experiment. Only load trusted local pickle artifacts; hashes detect corruption, not malicious pickle code.

## Scientific and release limitations

Clusters are mathematical surface-property regimes; anomaly scores identify statistical candidates. Neither is a validated geological unit or calibrated probability. Slope calibration and physical interpretation remain unresolved. Correlated/derived features, spatial dependence, resampling and limited geographic coverage can influence results.

The tested real window had no NoData holes; synthetic tests exercise missing-data/boundary paths. No full-resolution Venus map is validated by that pilot. The stable texture kernel measured about2.5–3.2x the former kernel's runtime on262/518-pixel windows, with lower traced allocation; full-pipeline performance is a separate measurement.

Geology association is unassessed without approved count tables. Optional geology discovery scans namespace validation CSVs without establishing their scientific provenance; do not populate that directory with unrelated experiments. Count tables must have unique filename stems: the current optional helper derives output names from stems. Optional enrichment, plotting and coarse-object helpers have weaker direct-call overwrite protection than stage orchestration; use new destinations. These helpers need further hardening before broader production claims.

Source release should contain source, tests, portable configs and documentation only. Keep inputs, local configs, fitted states, maps, figures and all experimental evidence local. .gitignore helps avoid accidental inclusion but cannot establish an actual tracked/staged release inventory; no Git operations were performed. See RELEASE_READINESS_REVIEW.md for the historical review and its technical limitations.

## Licensing

The approved V2 source-only release, including project source, tests, documentation and portable configuration examples, is licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0). The complete licence text is in the root [LICENSE](LICENSE) file.

Existing third-party notices and licences must be respected. Licensing the V2 source does not automatically grant permission to redistribute external Venus datasets or separately licensed material.

This remains experimental software; geological validity has not been established.
