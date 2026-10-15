"""Stage-1 debug views: render + disk cache for API / prewarm CLI.

Algorithms still run on full-res; only the saved/served JPEG is downscaled.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from blade.blade_mask import (
    apply_mask_bgr,
    debug_expand_close_pass,
    debug_expand_lines_pass,
    debug_expand_trap_pass,
    debug_geo2_pass,
    debug_seed_aux_pass,
    debug_sky_water_pass,
    debug_sunny_water_pass,
    debug_tip_edge_pass,
    debug_tip_need_pass,
    debug_white_refine,
    estimate_blade_mask,
    overlay_edge_seed_debug,
    overlay_expand_close_debug,
    overlay_expand_lines_debug,
    overlay_expand_trap_debug,
    overlay_geo2_debug,
    overlay_mask_with_hub,
    overlay_seed_aux_debug,
    overlay_seed_only,
    overlay_sky_water_debug,
    overlay_sunny_water_debug,
    overlay_tip_edge_debug,
    overlay_tip_need_debug,
    overlay_white_refine_debug,
)
from blade.config import Thresholds
from blade.lighting import estimate_lighting, overlay_lighting_debug
from blade.match import draw_keypoints_bgr, extract_features

STAGE1_VIEWS = frozenset(
    {
        "original",
        "lighting",
        "sky_water",
        "sunny_water",
        "edge_seed",
        "seed_aux",
        "overlay",
        "blade",
        "seed",
        "refine_delta",
        "geo2",
        "tip_need",
        "tip_edge",
        "expand_lines",
        "expand_trap",
        "expand_delta",
        "kps_masked",
        "kps_full",
    }
)

# Views the /review step bar actually shows (plus original for fallbacks).
REVIEW_PREWARM_VIEWS = ("original", "lighting", "overlay", "kps_masked")


def stage1_cache_name(year: str, name: str, view: str) -> str:
    stem = Path(name).stem
    return f"stage1_{year}_{stem}_{view}.jpg"


def stage1_cache_path(out_dir: Path, year: str, name: str, view: str) -> Path:
    return Path(out_dir) / stage1_cache_name(year, name, view)


def downscale_bgr(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink for browser display; INTER_AREA keeps edges readable after shrink."""
    if max_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def encode_jpeg(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    quality: int = 80,
) -> bytes:
    out = downscale_bgr(bgr, max_side)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def render_stage1_view(
    bgr: np.ndarray,
    *,
    year: str,
    name: str,
    view: str,
    thresholds: Thresholds,
) -> np.ndarray:
    """Return full-res BGR debug frame for one stage1 view name."""
    if view not in STAGE1_VIEWS:
        raise ValueError(f"unknown view: {view}")

    safe = Path(name).name
    if view == "original":
        return bgr
    if view == "lighting":
        return overlay_lighting_debug(estimate_lighting(bgr, year=year))
    if view == "sky_water":
        return overlay_sky_water_debug(debug_sky_water_pass(bgr))
    if view == "sunny_water":
        return overlay_sunny_water_debug(debug_sunny_water_pass(bgr))
    if view == "edge_seed":
        return overlay_edge_seed_debug(debug_sky_water_pass(bgr))
    if view == "seed_aux":
        return overlay_seed_aux_debug(debug_seed_aux_pass(bgr, year=year))
    if view in ("seed", "refine_delta"):
        wr = debug_white_refine(bgr, year=year)
        if view == "seed":
            return overlay_seed_only(wr)
        return overlay_white_refine_debug(wr)
    if view == "geo2":
        return overlay_geo2_debug(debug_geo2_pass(bgr, year=year))
    if view == "tip_need":
        return overlay_tip_need_debug(debug_tip_need_pass(bgr, year=year))
    if view == "tip_edge":
        return overlay_tip_edge_debug(debug_tip_edge_pass(bgr, year=year))
    if view == "expand_lines":
        return overlay_expand_lines_debug(debug_expand_lines_pass(bgr, year=year))
    if view == "expand_trap":
        return overlay_expand_trap_debug(debug_expand_trap_pass(bgr, year=year))
    if view == "expand_delta":
        return overlay_expand_close_debug(debug_expand_close_pass(bgr, year=year))

    mr = estimate_blade_mask(bgr, year=year)
    if view == "overlay":
        return overlay_mask_with_hub(bgr, mr.mask, mr.hub_xy)
    if view == "blade":
        return apply_mask_bgr(bgr, mr.mask)
    if view == "kps_full":
        pack = extract_features(safe, bgr, thresholds, mask=None)
        return draw_keypoints_bgr(bgr, pack, color=(255, 0, 255))
    if view == "kps_masked":
        pack = extract_features(safe, bgr, thresholds, mask=mr.mask)
        return draw_keypoints_bgr(bgr, pack, color=(0, 255, 255))
    raise ValueError(f"unknown view: {view}")


def write_stage1_view(
    src: Path,
    *,
    year: str,
    view: str,
    out_dir: Path,
    thresholds: Thresholds,
    max_side: int = 1280,
    quality: int = 80,
    force: bool = False,
) -> Path:
    """Render one view to disk (skip if fresh). Returns cache path."""
    if view not in STAGE1_VIEWS:
        raise ValueError(f"unknown view: {view}")
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = stage1_cache_path(out_dir, year, src.name, view)
    if (
        not force
        and out.is_file()
        and out.stat().st_mtime >= src.stat().st_mtime
    ):
        return out

    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"unreadable: {src}")
    vis = render_stage1_view(
        bgr, year=year, name=src.name, view=view, thresholds=thresholds
    )
    out.write_bytes(encode_jpeg(vis, max_side=max_side, quality=quality))
    return out
