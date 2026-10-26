"""Minimal standalone defaults; explicit V2 config can override the five inputs.

When overlaid on a host project, __file__ keeps defaults relative to that host.
No code in this module creates directories or accesses the original project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RADAR_FILE = RAW_DATA_DIR / "venus.tif"
HEIGHT_FILE = RAW_DATA_DIR / "venus_height.tif"
EMISSIVITY_FILE = RAW_DATA_DIR / "venus_emissivity.tif"
SLOPE_FILE = RAW_DATA_DIR / "venus_slope.tif"
REFLECTIVITY_FILE = RAW_DATA_DIR / "venus_reflectivity.tif"
K4_RASTER = PROJECT_ROOT / "outputs" / "rasters" / "venus_clusters_k4_full.tif"
