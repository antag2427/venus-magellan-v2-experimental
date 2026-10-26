# Venus / Magellan V2 Design

## Objective

Extend the V1 project from a five-variable K-Means experiment into a reproducible planetary surface-regime discovery system.

## Stage 1 — Planetary Feature Representation

### Base variables

- radar
- height
- emissivity
- slope
- reflectivity

### Derived variables

For each base raster, V2 computes local standard deviation. For selected variables it also computes local range. From height and radar it derives gradient magnitude; height also receives a discrete Laplacian curvature proxy.

This produces a compact 17-dimensional representation without writing a separate full-size derived raster for every feature.

Sampling is performed in two passes over aligned blocks. First, valid-pixel counts are measured; second, a reproducible proportional allocation selects exactly the requested number of valid samples. Local features are computed only for selected blocks.

## Stage 2 — Multi-Model Discovery

Four K=4 model families are compared on the same standardized subset:

1. K-Means
2. MiniBatch K-Means
3. Gaussian Mixture
4. BIRCH

Because cluster IDs are arbitrary across algorithms, labels are aligned against K-Means using a maximum-overlap Hungarian assignment. Consensus labels are the majority vote after alignment.

Outputs include:

- pairwise ARI
- consensus label
- model agreement
- normalized model disagreement entropy
- K-Means seed stability

Consensus is deliberately calculated on a fixed subset to avoid quadratic-memory methods across the full 100k rows.

## Stage 3 — Scientific Interpretation

### Uncertainty

For each sample:

- nearest centroid distance
- second-nearest centroid distance
- distance margin
- soft cluster probabilities
- normalized entropy
- confidence = 1 - normalized entropy

These quantify ambiguity rather than pretending every K-Means label is equally certain.

### Anomaly detection

Two complementary signals are combined:

- distance from the nearest K-Means centroid
- Isolation Forest outlier score

Each signal is percentile-normalized and averaged. The result is a **candidate anomaly score**.

An anomaly is a statistical candidate for follow-up, not proof of an unmapped geological unit.

### Spatial candidate objects

When the V1 K=4 GeoTIFF exists, V2 creates a coarse connected-component atlas after downsampling. This is explicitly a candidate-object layer rather than an exact full-resolution topology product.

## Scientific guardrails

- V2 does not overwrite V1.
- ML regimes are not renamed as geological classes.
- Geological maps remain independent evidence.
- Anomaly scores are candidate-generation tools.
- Coarse object areas are approximate.
- Full-resolution map processing remains available through the existing V1 pipeline.
