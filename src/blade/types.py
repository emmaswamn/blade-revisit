from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    RETAKE = "retake"
    REJECT = "reject"
    WATCH = "watch"
    ESCALATE = "escalate"
    SKIP = "skip"  # query too weak — did not search gallery


@dataclass
class Thresholds:
    blur_laplacian_min: float = 80.0
    inlier_ratio_min: float = 0.15
    area_delta_min: float = 0.25
    max_side: int = 1280
    detector: str = "akaze"
    ratio_test: float = 0.75
    ransac_reproj_threshold: float = 5.0
    min_matches: int = 12
    min_query_keypoints: int = 40  # below this → skip gallery search

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "blur_laplacian_min": self.blur_laplacian_min,
            "inlier_ratio_min": self.inlier_ratio_min,
            "area_delta_min": self.area_delta_min,
            "min_query_keypoints": self.min_query_keypoints,
        }


@dataclass
class MatchStats:
    inlier_ratio: float | None
    num_matches: int
    num_inliers: int
    H: Any | None = None  # numpy 3x3 or None


@dataclass
class PairResult:
    query_id: str  # 2018
    ref_id: str  # 2017 (or "" if skipped before search)
    sharpness_ok: bool
    inlier_ratio: float | None
    blade_width_px: float | None
    area_ratio_2017: float | None
    area_ratio_2018: float | None
    area_delta: float | None
    tier: Tier
    reason: str
    thresholds_used: dict[str, float | int | str] = field(default_factory=dict)
    overlay_uri: str | None = None
    num_matches: int = 0
    num_inliers: int = 0
    # Per-year sharpness (query=2018, ref=2017)
    lap_2018: float | None = None
    lap_2017: float | None = None
    sharp_2018_ok: bool | None = None
    sharp_2017_ok: bool | None = None
    pipeline_furthest: str = ""
    pipeline_steps: list[dict[str, str]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["thresholds_used"] = dict(self.thresholds_used)
        d["pipeline_steps"] = list(self.pipeline_steps)
        return d
