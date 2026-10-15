from __future__ import annotations

from pathlib import Path

import yaml

from blade.types import Thresholds


def load_thresholds(path: str | Path) -> Thresholds:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    match = raw.get("match") or {}
    return Thresholds(
        blur_laplacian_min=float(raw.get("blur_laplacian_min", 80.0)),
        inlier_ratio_min=float(raw.get("inlier_ratio_min", 0.15)),
        area_delta_min=float(raw.get("area_delta_min", 0.25)),
        max_side=int(match.get("max_side", 1280)),
        detector=str(match.get("detector", "akaze")).lower(),
        ratio_test=float(match.get("ratio_test", 0.75)),
        ransac_reproj_threshold=float(match.get("ransac_reproj_threshold", 5.0)),
        min_matches=int(match.get("min_matches", 12)),
        min_query_keypoints=int(match.get("min_query_keypoints", 40)),
    )
