"""
Schemas for split-aware statistics in the small DANRA/ERA5 STRIDE adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CropConfig:
    """
    Spatial crop definition used when computing statistics.
    """

    anchor_y: int
    anchor_x: int
    height: int
    width: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StatsRequest:
    """
    Declarative request describing which statistics should be computed.
    """

    split_name: str
    split_manifest_path: str
    variable: str
    source: str
    transform_name: str
    domain_tag: str
    crop: CropConfig | None = None
    conditioning_variable_order: tuple[str, ...] = ("prcp", "temp")
    static_variable_order: tuple[str, ...] = ("lsm", "topo")

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

    n_dates: int
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

    variable: str
    source: str
    transform_name: str
    split_name: str
    split_manifest_path: str
    date_count: int
    dates_used: list[str]
    domain_tag: str
    crop: dict[str, int] | None
    transform_stats: dict[str, float]
    physical_summary: StatsSummary
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "source": self.source,
            "transform_name": self.transform_name,
            "split_name": self.split_name,
            "split_manifest_path": self.split_manifest_path,
            "date_count": self.date_count,
            "dates_used": self.dates_used,
            "domain_tag": self.domain_tag,
            "crop": self.crop,
            "transform_stats": self.transform_stats,
            "physical_summary": self.physical_summary.to_dict(),
            "metadata": self.metadata,
        }
