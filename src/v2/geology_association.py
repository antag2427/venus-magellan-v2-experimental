from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"None of these columns were found: {candidates}")


def compute_enrichment(
    counts_csv: str | Path,
    output_csv: str | Path,
    cluster_column_candidates: list[str] | None = None,
    geology_column_candidates: list[str] | None = None,
    count_column_candidates: list[str] | None = None,
) -> dict[str, int | float | str]:
    """Compute geology-by-cluster purity and enrichment from a long-form count table.

    Required conceptual columns: cluster, geological unit/category, count.
    """
    cluster_column_candidates = cluster_column_candidates or ["cluster", "cluster_id", "ml_cluster"]
    geology_column_candidates = geology_column_candidates or ["unit", "geology", "category", "geologic_unit"]
    count_column_candidates = count_column_candidates or ["count", "pixels", "n", "pixel_count"]

    df = pd.read_csv(counts_csv)
    ccol = _find_column(df, cluster_column_candidates)
    gcol = _find_column(df, geology_column_candidates)
    ncol = _find_column(df, count_column_candidates)
    df = df[[ccol, gcol, ncol]].copy()
    df.columns = ["cluster", "geology", "count"]
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    global_cluster = df.groupby("cluster")["count"].sum()
    global_geology = df.groupby("geology")["count"].sum()
    total = float(df["count"].sum())

    rows = []
    for row in df.itertuples(index=False):
        cluster = row.cluster
        geology = row.geology
        count = float(row.count)
        cluster_total = float(global_cluster.loc[cluster])
        geology_total = float(global_geology.loc[geology])
        p_geology_given_cluster = count / cluster_total if cluster_total else np.nan
        p_cluster_given_geology = count / geology_total if geology_total else np.nan
        expected = cluster_total * geology_total / total if total else np.nan
        enrichment = count / expected if expected and expected > 0 else np.nan
        rows.append({
            "cluster": cluster,
            "geology": geology,
            "count": count,
            "cluster_total": cluster_total,
            "geology_total": geology_total,
            "purity": p_geology_given_cluster,
            "cluster_fraction_within_geology": p_cluster_given_geology,
            "enrichment": enrichment,
        })

    out = pd.DataFrame(rows).sort_values(["cluster", "enrichment"], ascending=[True, False])
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return {
        "rows": int(len(out)),
        "clusters": int(out["cluster"].nunique()) if len(out) else 0,
        "geology_units": int(out["geology"].nunique()) if len(out) else 0,
        "output": str(output),
    }


def discover_and_run_associations(validation_root: str | Path, output_dir: str | Path) -> list[dict[str, str | int | float]]:
    """Search existing validation CSVs for compatible long-form cluster/geology/count tables."""
    root = Path(validation_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    if not root.exists():
        return results
    for csv_path in root.rglob("*.csv"):
        try:
            result = compute_enrichment(
                csv_path,
                out / f"{csv_path.stem}_enrichment.csv",
            )
        except (ValueError, KeyError, pd.errors.ParserError, TypeError):
            continue
        result["source"] = str(csv_path)
        results.append(result)
    return results
