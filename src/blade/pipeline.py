"""Pipeline stage labels — print / store how far we got.

Week-1 implemented stages are marked; later stages stay skipped/stub.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    LOAD = "1_load"
    GRAY_RESIZE = "2_gray_resize"
    SHARPNESS = "3_sharpness"
    BLADE_MASK = "4_blade_mask"  # stub
    FEATURES = "5_features"
    MATCH_RATIO = "6_match_ratio"
    HOMOGRAPHY = "7_homography"
    DECIDE = "8_decide"
    DIFF = "9_diff"  # stub


# Human-readable order for the debug UI
STAGE_LABELS: dict[Stage, str] = {
    Stage.LOAD: "读图",
    Stage.GRAY_RESIZE: "转灰度 + 缩边",
    Stage.SHARPNESS: "清晰度（拉普拉斯方差）",
    Stage.BLADE_MASK: "整机 mask（1.1 edge_hub）",
    Stage.FEATURES: "机上特征点（1.2）",
    Stage.MATCH_RATIO: "描述子匹配 + Lowe ratio",
    Stage.HOMOGRAPHY: "RANSAC findHomography",
    Stage.DECIDE: "三档判定",
    Stage.DIFF: "对齐差分（未实现）",
}


@dataclass
class StepEvent:
    stage: str
    status: str  # ok | fail | skip | stub
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PipelineLog:
    events: list[StepEvent] = field(default_factory=list)
    furthest: str = Stage.LOAD.value

    def add(self, stage: Stage, status: str, detail: str = "") -> None:
        self.events.append(StepEvent(stage=stage.value, status=status, detail=detail))
        if status in ("ok", "stub"):
            self.furthest = stage.value

    def to_list(self) -> list[dict[str, str]]:
        return [e.to_dict() for e in self.events]
