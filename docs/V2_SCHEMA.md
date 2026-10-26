# V2 Output Schema

## Advanced training sample

`data/processed/v2/venus_advanced_training_sample.csv`

Coordinates:

- `row`
- `col`

Base features:

- `radar`
- `height`
- `emissivity`
- `slope`
- `reflectivity`

Derived features:

- `radar_local_std`
- `height_local_std`
- `emissivity_local_std`
- `slope_local_std`
- `reflectivity_local_std`
- `radar_local_range`
- `height_local_range`
- `emissivity_local_range`
- `reflectivity_local_range`
- `height_curvature`
- `radar_gradient`
- `height_gradient`

## Consensus labels

`outputs/v2/clustering/v2_consensus_labels.csv` stores raw labels, aligned labels, consensus label, agreement, and entropy for the fixed discovery subset.

## Uncertainty table

`outputs/v2/analysis/v2_cluster_uncertainty.csv` contains cluster, nearest/second-centroid distances, distance margin, normalized entropy, confidence, and soft cluster probabilities.

## Anomaly table

`outputs/v2/analysis/v2_anomaly_candidates.csv` contains centroid-distance anomaly, Isolation Forest anomaly, percentile-normalized signals, and the combined candidate score.

## Full-map GeoTIFFs

When Stage 3 is run:

- `outputs/v2/rasters/venus_v2_clusters_k4.tif`
- `outputs/v2/rasters/venus_v2_confidence.tif`
- `outputs/v2/rasters/venus_v2_anomaly.tif`

All three use the V1 radar reference grid. The categorical raster uses `255` as NoData; the continuous products use `-9999.0`.
