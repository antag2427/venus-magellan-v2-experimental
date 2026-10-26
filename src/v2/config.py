from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any

BASE_FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

DERIVED_FEATURES = [
    "radar_local_std",
    "height_local_std",
    "emissivity_local_std",
    "slope_local_std",
    "reflectivity_local_std",
    "radar_local_range",
    "height_local_range",
    "emissivity_local_range",
    "reflectivity_local_range",
    "height_curvature",
    "radar_gradient",
    "height_gradient",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


@dataclass(frozen=True)
class V2Config:
    sample_size: int = 100_000
    sample_min_candidate_factor: float = 1.10
    sample_batch_factor: float = 1.5
    random_state: int = 42
    block_size: int = 512
    local_radius: int = 2
    k: int = 4
    consensus_subset: int = 20_000
    anomaly_subset: int = 50_000
    anomaly_contamination: float = 0.01
    isolation_estimators: int = 300
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 500
    gmm_n_init: int = 3
    gmm_max_iter: int = 300
    birch_threshold: float = 0.5
    minibatch_batch_size: int = 2048
    object_downsample_max_dim: int = 2048
    output_prefix: str = "v2"
    raster_inputs: dict[str, str] | None = None
    output_root: str | None = None
    core_window: dict[str, int] | None = None
    halo_policy: str = "clip"

    def __post_init__(self) -> None:
        positive = ("sample_size", "block_size", "consensus_subset", "anomaly_subset",
                    "isolation_estimators", "kmeans_n_init", "kmeans_max_iter",
                    "gmm_n_init", "gmm_max_iter", "minibatch_batch_size",
                    "object_downsample_max_dim")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.k, bool) or not isinstance(self.k, Integral) or not 2 <= self.k <= 255:
            raise ValueError("k must be an integer in [2, 255].")
        if not isinstance(self.random_state, Integral) or isinstance(self.random_state, bool) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if not isinstance(self.local_radius, Integral) or isinstance(self.local_radius, bool) or self.local_radius < 0:
            raise ValueError("local_radius must be a nonnegative integer.")
        for name in ("sample_size", "consensus_subset"):
            if getattr(self, name) < self.k:
                raise ValueError(f"{name} must be at least k.")
        for name in ("birch_threshold", "anomaly_contamination"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.anomaly_contamination > 0.5:
            raise ValueError("anomaly_contamination must be at most 0.5.")
        if self.object_downsample_max_dim < 2:
            raise ValueError("object_downsample_max_dim must be at least 2.")
        if self.raster_inputs is not None:
            validate_input_mapping(self.raster_inputs)
            object.__setattr__(self, "raster_inputs",
                {name: str(resolve_path(self.raster_inputs[name], Path.cwd())) for name in BASE_FEATURES})
        if self.output_root is not None:
            validate_output_root(self.output_root)
            object.__setattr__(self, "output_root", str(resolve_path(self.output_root, Path.cwd())))
        validate_core_window(self.core_window)
        if self.halo_policy not in ("clip", "require_full"):
            raise ValueError("halo_policy must be 'clip' or 'require_full'.")
        if self.core_window is not None and self.local_radius != 2:
            raise ValueError("core_window policy requires local_radius=2 (three-pixel halo).")
        validate_prefix(self.output_prefix)
        validate_sampling_factors(self.sample_min_candidate_factor, self.sample_batch_factor)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "V2Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object.")
        unknown = set(payload) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        base = Path(path).expanduser().resolve().parent
        if payload.get("raster_inputs") is not None:
            validate_input_mapping(payload["raster_inputs"])
            payload["raster_inputs"] = {name: str(resolve_path(value, base))
                                        for name,value in payload["raster_inputs"].items()}
        if payload.get("output_root") is not None:
            validate_output_root(payload["output_root"])
            payload["output_root"] = str(resolve_path(payload["output_root"], base))
        return cls(**payload)

    def save_json(self, path: str | Path) -> None:
        from .report import write_json
        write_json(path, self.to_dict())


def validate_prefix(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("output_prefix must be a safe nonempty file-name component.")
    return value


def validate_sampling_factors(min_candidate_factor=1.10, batch_factor=1.5) -> None:
    # Reserved legacy controls: the current sampler allocates exact counts, with
    # no candidate/rejection batch operation to which these factors could apply.
    for name, value, default in (("sample_min_candidate_factor", min_candidate_factor, 1.10),
                                 ("sample_batch_factor", batch_factor, 1.5)):
        if isinstance(value, bool) or not isinstance(value, Real) or value != default:
            raise ValueError(f"{name} is unsupported by the exact stratified sampler; retain the legacy default {default}.")


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be nonempty strings.")
    path = Path(value).expanduser()
    if path.drive and not path.is_absolute():
        raise ValueError(f"Drive-relative paths are ambiguous: {value}")
    return (path if path.is_absolute() else Path(base)/path).resolve()


def validate_input_mapping(mapping):
    if not isinstance(mapping, dict) or set(mapping) != set(BASE_FEATURES):
        raise ValueError(f"raster_inputs must contain exactly {BASE_FEATURES}.")
    if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
        raise ValueError("raster_inputs values must be nonempty path strings.")


def validate_output_root(value):
    if not isinstance(value, str) or not value.strip() or value.strip() in (".", ".."):
        raise ValueError("output_root must be an explicit directory path, not '.' or '..'.")
    path = resolve_path(value, Path.cwd())
    if path == path.parent:
        raise ValueError("output_root cannot be a filesystem root.")


def validate_core_window(core):
    if core is None:
        return
    if not isinstance(core, dict) or set(core) != {"row", "col", "height", "width"}:
        raise ValueError("core_window requires exactly row, col, height, width.")
    for name,value in core.items():
        minimum = 0 if name in ("row", "col") else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"core_window {name} must be an integer >= {minimum}.")
