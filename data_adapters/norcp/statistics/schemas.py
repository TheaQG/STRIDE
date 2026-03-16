

"""
Schemas for split-aware statistics in the NorCP STRIDE adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CropConfig:
    """
    Spatial crop definition used when computing statistics.

    Coordinates follow the NorCP [H, W] convention.
    """

    top: int
    left: int
    height: int
    width: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StatsRequest:
    """
    Declarative request describing which NorCP statistics should be computed.
    """

    scenario_name: str
    root_dir: str
    split_name: str
    split_manifest_path: str
    variable: str
    source: str
    transform_name: str
    domain_tag: str
    spatial_tag: str
    temporal_tag: str | None = None
    variable_time_offset_hours: float = 0.0
    crop: CropConfig | None = None
    static_file_path: str | None = None
    backend: str = "python"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.crop is None:
            data["crop"] = None
        return data


@dataclass(frozen=True)
class StatsSummary:
    """
    Basic descriptive summary for the pooled physical-space values.
    """

    n_samples: int
    n_values: int
    min: float
    max: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class StatsResult:
    """
    Final statistics object saved to JSON and later consumed by transforms.
    """

    scenario_name: str
    variable: str
    source: str
    transform_name: str
    split_name: str
    split_manifest_path: str
    sample_count: int
    timestamps_used: list[str]
    domain_tag: str
    spatial_tag: str
    temporal_tag: str | None
    crop: dict[str, int] | None
    transform_stats: dict[str, float]
    physical_summary: StatsSummary
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "variable": self.variable,
            "source": self.source,
            "transform_name": self.transform_name,
            "split_name": self.split_name,
            "split_manifest_path": self.split_manifest_path,
            "sample_count": self.sample_count,
            "timestamps_used": self.timestamps_used,
            "domain_tag": self.domain_tag,
            "spatial_tag": self.spatial_tag,
            "temporal_tag": self.temporal_tag,
            "crop": self.crop,
            "transform_stats": self.transform_stats,
            "physical_summary": self.physical_summary.to_dict(),
            "metadata": self.metadata,
        }