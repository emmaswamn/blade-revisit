from __future__ import annotations

import cv2
import numpy as np

from blade.diff import estimate_change_stub
from blade.match import MatchStats
from blade.pipeline import PipelineLog, Stage
from blade.sharpness import is_sharp, laplacian_variance
from blade.types import PairResult, Thresholds, Tier


def _blur_reason(
    *,
    lap_2018: float,
    lap_2017: float | None,
    ok_2018: bool,
    ok_2017: bool | None,
    thr: float,
) -> str:
    parts = []
    if not ok_2018:
        parts.append(f"2018清晰度不足(lap={lap_2018:.1f}<{thr})")
    if ok_2017 is False and lap_2017 is not None:
        parts.append(f"2017清晰度不足(lap={lap_2017:.1f}<{thr})")
    return "；".join(parts) if parts else f"清晰度不足(thr={thr})"


def decide_pair(
    query_id: str,
    ref_id: str,
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    thresholds: Thresholds,
    stats: MatchStats | None = None,
    log: PipelineLog | None = None,
) -> PairResult:
    """Map match stats onto a tier. query=2018, ref=2017."""
    log = log or PipelineLog()
    snap = thresholds.snapshot()
    thr = thresholds.blur_laplacian_min

    q_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    log.add(Stage.GRAY_RESIZE, "ok", f"query={q_gray.shape} ref={r_gray.shape}")

    lap_2018 = laplacian_variance(q_gray)
    lap_2017 = laplacian_variance(r_gray)
    ok_2018 = lap_2018 >= thr
    ok_2017 = lap_2017 >= thr
    sharpness_ok = ok_2018 and ok_2017
    log.add(
        Stage.SHARPNESS,
        "ok" if sharpness_ok else "fail",
        f"2018 lap={lap_2018:.1f} ({'ok' if ok_2018 else 'FAIL'}) | "
        f"2017 lap={lap_2017:.1f} ({'ok' if ok_2017 else 'FAIL'}) | thr={thr}",
    )
    log.add(Stage.BLADE_MASK, "stub", "整图匹配，尚未抠叶片")

    if stats is None:
        raise ValueError("decide_pair expects precomputed MatchStats from retrieve")

    if stats.inlier_ratio is not None:
        log.add(
            Stage.HOMOGRAPHY,
            "ok",
            f"inlier_ratio={stats.inlier_ratio:.3f} "
            f"matches={stats.num_matches} inliers={stats.num_inliers}",
        )
    else:
        log.add(
            Stage.HOMOGRAPHY,
            "fail",
            f"no H; matches={stats.num_matches} inliers={stats.num_inliers}",
        )

    diff = estimate_change_stub()
    log.add(Stage.DIFF, "stub", "面积差分未实现")

    base = dict(
        query_id=query_id,
        ref_id=ref_id,
        sharpness_ok=sharpness_ok,
        inlier_ratio=stats.inlier_ratio,
        blade_width_px=diff.blade_width_px,
        area_ratio_2017=diff.area_ratio_2017,
        area_ratio_2018=diff.area_ratio_2018,
        area_delta=diff.area_delta,
        thresholds_used=snap,
        num_matches=stats.num_matches,
        num_inliers=stats.num_inliers,
        lap_2018=lap_2018,
        lap_2017=lap_2017,
        sharp_2018_ok=ok_2018,
        sharp_2017_ok=ok_2017,
        pipeline_furthest=log.furthest,
        pipeline_steps=log.to_list(),
    )

    if not sharpness_ok:
        reason = _blur_reason(
            lap_2018=lap_2018,
            lap_2017=lap_2017,
            ok_2018=ok_2018,
            ok_2017=ok_2017,
            thr=thr,
        )
        log.add(Stage.DECIDE, "ok", f"tier=retake · {reason}")
        base["pipeline_furthest"] = log.furthest
        base["pipeline_steps"] = log.to_list()
        return PairResult(tier=Tier.RETAKE, reason=reason, **base)

    if stats.inlier_ratio is None or stats.inlier_ratio < thresholds.inlier_ratio_min:
        reason = (
            f"内点比例不足(inlier={stats.inlier_ratio}, "
            f"thr={thresholds.inlier_ratio_min}) 或匹配太少"
        )
        log.add(Stage.DECIDE, "ok", f"tier=reject · {reason}")
        base["pipeline_furthest"] = log.furthest
        base["pipeline_steps"] = log.to_list()
        return PairResult(tier=Tier.REJECT, reason=reason, **base)

    reason = "已匹配；面积差分未实现 → 暂标 watch"
    log.add(Stage.DECIDE, "ok", f"tier=watch · {reason}")
    base["pipeline_furthest"] = log.furthest
    base["pipeline_steps"] = log.to_list()
    return PairResult(tier=Tier.WATCH, reason=reason, **base)
