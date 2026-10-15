"""Stage 1.0 — estimate sunny vs overcast from a single frame.

Does NOT change the mask yet. Debug overlay first; overcast seed tightening
comes after human checks on 2017 cloudy vs 2018 sunny samples.

See docs/pipeline-stages.md §1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LightingResult:
    work_bgr: np.ndarray
    lighting: str  # "sunny" | "overcast"
    sky_band: tuple[int, int, int, int]  # x0, y0, x1, y1
    water_band: tuple[int, int, int, int]
    sky_v: float
    sky_std: float
    sky_s: float
    contrast: float
    lap: float
    veg_frac: float
    ground_v: float
    sky_ground_gap: float
    score_sunny: float
    score_overcast: float
    year_bias_used: bool
    detail: str
    veg_mask: np.ndarray
    scale: float


def _resize_max_side_bgr(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return bgr, 1.0
    scale = long_side / max_side
    small = cv2.resize(
        bgr,
        (int(round(w / scale)), int(round(h / scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def _local_std(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.GaussianBlur(g, (ksize, ksize), 0)
    mean_sq = cv2.GaussianBlur(g * g, (ksize, ksize), 0)
    var = np.maximum(mean_sq - mean * mean, 0)
    return np.sqrt(var)


def _veg_mask(bgr: np.ndarray) -> np.ndarray:
    """High-sat green/brown field — same spirit as blade_mask reject."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv)
    green = (h >= 30) & (h <= 95) & (s >= 35)
    brown = ((h <= 30) | (h >= 170)) & (s >= 40)
    return green | brown


def estimate_lighting(
    bgr: np.ndarray,
    *,
    year: str | None = None,
    max_side: int = 1280,
    sky_frac: float = 0.14,
    water_frac: float = 0.12,
) -> LightingResult:
    """0a–0e: sky/water bands → features → sunny|overcast.

    ``year`` is only a weak tie-breaker (2017 → lean overcast when close).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")

    small, scale = _resize_max_side_bgr(bgr, max_side)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _hh, ss, vv = cv2.split(hsv)

    # --- 0a bands ---
    sky_h = max(8, int(round(h * sky_frac)))
    water_h = max(8, int(round(h * water_frac)))
    sky_band = (0, 0, w - 1, sky_h - 1)
    water_band = (0, h - water_h, w - 1, h - 1)

    sky_roi_v = vv[:sky_h, :]
    sky_roi_s = ss[:sky_h, :]
    sky_roi_g = gray[:sky_h, :]
    # mid-lower ground strip (avoid sky + very bottom water)
    g0 = int(h * 0.35)
    g1 = int(h * 0.85)
    ground_v = float(np.mean(vv[g0:g1, :])) if g1 > g0 else float(np.mean(vv))

    # --- 0b sky features ---
    sky_v = float(np.mean(sky_roi_v))
    sky_s = float(np.mean(sky_roi_s))
    sky_std = float(np.mean(_local_std(sky_roi_g, 11)))
    sky_ground_gap = float(sky_v - ground_v)

    # --- 0c global contrast / laplacian ---
    contrast = float(np.std(gray.astype(np.float32)))
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # --- 0d vegetation fraction (lower 70% — fields live there) ---
    veg = _veg_mask(small)
    lower = veg[int(h * 0.30) :, :]
    veg_frac = float(lower.mean()) if lower.size else 0.0

    # --- 0e scores (display-first; thresholds will be tuned on samples) ---
    # Classic overcast sky: bright + nearly flat + low sat (2017 grey ceiling).
    # This must outweigh field color / global contrast — white sky vs dark blade
    # still has high contrast on cloudy days.
    oc = 0.0
    flat_grey_sky = sky_v >= 150 and sky_std <= 6 and sky_s <= 30
    soft_grey_sky = sky_v >= 130 and sky_std <= 14 and sky_s <= 45
    if flat_grey_sky:
        oc += 3.5
    elif soft_grey_sky:
        oc += 2.2
    elif sky_v >= 120 and sky_std <= 22 and sky_s <= 55:
        oc += 1.0
    if lap < 100:
        oc += 1.2
    elif lap < 200:
        oc += 0.6
    # Only trust low global contrast when sky is already soft; otherwise blade
    # edges inflate contrast on overcast frames.
    if not flat_grey_sky:
        if contrast < 42:
            oc += 1.0
        elif contrast < 52:
            oc += 0.5
    if veg_frac < 0.08:
        oc += 0.8
    elif veg_frac < 0.18 and flat_grey_sky:
        oc += 0.3

    # Sunny: textured / blue-ish sky, or strong field color with non-flat sky.
    su = 0.0
    if sky_s >= 55 or sky_std >= 18:
        su += 1.5
    if sky_s >= 80:
        su += 1.0  # clear blue
    if lap >= 400:
        su += 1.2
    elif lap >= 250:
        su += 0.6
    if veg_frac >= 0.18 and not flat_grey_sky:
        su += 1.2
    elif veg_frac >= 0.10 and not flat_grey_sky:
        su += 0.5
    # Bright sky vs darker ground only counts if sky isn't a flat grey slab
    if sky_ground_gap >= 35 and (sky_s >= 40 or sky_std >= 12):
        su += 0.8
    if contrast >= 55 and not flat_grey_sky and not soft_grey_sky:
        su += 0.8

    year_bias_used = False
    # Close call: |su - oc| < 0.8 → year tie-break
    if abs(su - oc) < 0.8 and year == "2017":
        oc += 0.6
        year_bias_used = True
    elif abs(su - oc) < 0.8 and year == "2018":
        su += 0.4
        year_bias_used = True

    lighting = "overcast" if oc > su else "sunny"
    detail = (
        f"lighting={lighting} su={su:.1f} oc={oc:.1f}"
        f" sky_v={sky_v:.0f} sky_std={sky_std:.1f} sky_s={sky_s:.0f}"
        f" contrast={contrast:.1f} lap={lap:.0f} veg={veg_frac:.2f}"
        f" gap={sky_ground_gap:.0f}"
        + (" year_bias" if year_bias_used else "")
    )

    return LightingResult(
        work_bgr=small,
        lighting=lighting,
        sky_band=sky_band,
        water_band=water_band,
        sky_v=sky_v,
        sky_std=sky_std,
        sky_s=sky_s,
        contrast=contrast,
        lap=lap,
        veg_frac=veg_frac,
        ground_v=ground_v,
        sky_ground_gap=sky_ground_gap,
        score_sunny=su,
        score_overcast=oc,
        year_bias_used=year_bias_used,
        detail=detail,
        veg_mask=veg.astype(np.uint8) * 255,
        scale=scale,
    )


def overlay_lighting_debug(dbg: LightingResult) -> np.ndarray:
    """1.0 view: yellow sky box, cyan water box, magenta veg hint, banner."""
    out = dbg.work_bgr.copy()
    h, w = out.shape[:2]

    # magenta veg (lower-field reference), light
    veg = dbg.veg_mask > 0
    if veg.any():
        tint = out.copy()
        tint[veg] = (255, 0, 255)
        out = cv2.addWeighted(tint, 0.22, out, 0.78, 0)

    # sky band = yellow
    x0, y0, x1, y1 = dbg.sky_band
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        out,
        "SKY",
        (x0 + 8, min(y1 - 6, y0 + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # water band = cyan
    wx0, wy0, wx1, wy1 = dbg.water_band
    cv2.rectangle(out, (wx0, wy0), (wx1, wy1), (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        out,
        "WATER?",
        (wx0 + 8, wy0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # color bar under top edge
    bar_h = max(10, h // 48)
    if dbg.lighting == "sunny":
        bar = (40, 180, 40)  # green-ish BGR
        verdict = "LIGHTING = SUNNY"
        vcol = (40, 220, 40)
    else:
        bar = (180, 140, 80)  # grey-blue
        verdict = "LIGHTING = OVERCAST"
        vcol = (220, 180, 120)

    overlay_bar = out.copy()
    cv2.rectangle(overlay_bar, (0, 0), (w - 1, bar_h), bar, -1)
    out = cv2.addWeighted(overlay_bar, 0.55, out, 0.45, 0)

    lines = [
        verdict,
        (
            f"sky_v={dbg.sky_v:.0f}  sky_std={dbg.sky_std:.1f}  sky_s={dbg.sky_s:.0f}"
        ),
        (
            f"contrast={dbg.contrast:.1f}  lap={dbg.lap:.0f}  veg_frac={dbg.veg_frac:.2f}"
        ),
        (
            f"su={dbg.score_sunny:.1f}  oc={dbg.score_overcast:.1f}"
            f"  gap={dbg.sky_ground_gap:.0f}"
            + ("  YEAR_BIAS" if dbg.year_bias_used else "")
        ),
        "YELLOW=sky  CYAN=water?  MAGENTA=veg ref  (mask untouched)",
    ]
    y_text = bar_h + 28
    for i, text in enumerate(lines):
        col = vcol if i == 0 else (255, 255, 255)
        scale = 0.72 if i == 0 else 0.55
        thick = 2 if i == 0 else 1
        cv2.putText(
            out, text, (12, y_text + i * 26), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (12, y_text + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            col,
            thick,
            cv2.LINE_AA,
        )
    return out
