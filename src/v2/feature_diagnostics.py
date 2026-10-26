from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES, validate_prefix
from .report import ensure_new, ensure_outputs_new, require_inputs, write_csv


def run_feature_diagnostics(sample_csv: str | Path, output_dir: str | Path, *, output_prefix: str = "v2") -> dict[str, str | int | float]:
    output = Path(output_dir)
    validate_prefix(output_prefix)
    ensure_outputs_new(*(output / f"{output_prefix}_{name}" for name in
                 ("feature_quality.csv", "feature_correlation.csv", "pca_explained_variance.csv")))
    require_inputs(sample_csv)
    df = pd.read_csv(sample_csv, float_precision="round_trip")
    output.mkdir(parents=True, exist_ok=True)
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = df[features].to_numpy(dtype=np.float64)

    quality = pd.DataFrame({
        "feature": features,
        "missing": [int(df[c].isna().sum()) for c in features],
        "unique": [int(df[c].nunique(dropna=True)) for c in features],
        "mean": [float(df[c].mean()) for c in features],
        "std": [float(df[c].std(ddof=0)) for c in features],
        "min": [float(df[c].min()) for c in features],
        "max": [float(df[c].max()) for c in features],
    })
    quality["missing_fraction"] = quality["missing"] / len(df)
    write_csv(quality, output / f"{output_prefix}_feature_quality.csv", index=False)

    corr = df[features].corr()
    write_csv(corr, output / f"{output_prefix}_feature_correlation.csv")

    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    pca_df = pd.DataFrame({
        "component": np.arange(1, len(features) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    write_csv(pca_df, output / f"{output_prefix}_pca_explained_variance.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "feature_count": int(len(features)),
        "features_with_missing": int((quality["missing"] > 0).sum()),
        "constant_features": int((quality["std"] == 0).sum()),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc1_pc2_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "output": str(output),
    }
    return summary
