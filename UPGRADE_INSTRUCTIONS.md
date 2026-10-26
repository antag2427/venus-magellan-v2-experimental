# Venus / Magellan V2 Upgrade Pack

For standalone setup and the current configuration interface, read README.md and ROUTING_AND_WINDOWS.md first. The overlay workflow below is historical context, not a complete standalone release inventory.

This pack adds an expansion layer without replacing the V1 baseline.

## What it adds

- **Feature engine:** 5 base variables + 12 local/derivative variables = 17-dimensional surface representation.
- **Multi-model discovery:** K-Means, MiniBatch K-Means, Gaussian Mixture, and BIRCH on a common K=4 consensus subset.
- **Consensus alignment:** Hungarian label matching, pairwise ARI, consensus label, agreement, and entropy.
- **Uncertainty:** centroid distance, distance margin, soft probabilities, normalized entropy, confidence.
- **Anomaly detection:** K-Means centroid anomaly + Isolation Forest anomaly + combined candidate score.
- **Spatial objects:** coarse connected-component candidate atlas from the existing K=4 GeoTIFF when available.
- **Automatic figures and Markdown reports.**
- **Unit tests** for the new core functions.

## Drop-in layout

Copy these paths into the root of the existing `venus_magallen` repository:

```text
main_v2.py
configs/v2_default.json
docs/V2_DESIGN.md
scripts/
src/v2/
tests/test_v2_core.py
UPGRADE_INSTRUCTIONS.md
```

No raw Venus datasets are included.

## Run order on the Windows repository

From:

```text
<your-existing-project-root>
```

run:

```powershell
python main_v2.py features
python main_v2.py discover
python main_v2.py interpret
```

Or run the full pipeline:

```powershell
python main_v2.py all
```

## Important scientific interpretation

The V2 outputs remain **surface-property regimes and candidate anomalies**. They are not asserted to be new geological units. Independent geology remains a validation/interpretation layer.

The V1 K=4 pipeline remains intact, so V2 results can be compared against the existing baseline rather than silently replacing it.
