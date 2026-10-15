"""Near-shot filter: drop whole-turbine far frames from the debug picker.

See docs/pipeline-stages.md 「构图范围」.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from blade.blade_mask import (
    _elongation,
    _hub_from_distance,
    _keep_hub_connected,
    _resize_max_side_bgr,
    _smooth_candidates,
)

FILTER_VERSION = 1


def _longest_arm_frac(mask: np.ndarray) -> tuple[float, float, float]:
    """Return (arm_frac, span, thick_frac) on the geometric seed."""
    h, w = mask.shape[:2]
    if mask.max() == 0:
        return 0.0, 0.0, 0.0
    hub, dt = _hub_from_distance(mask)
    if hub is None:
        return 0.0, 0.0, 0.0
    hx, hy = hub
    arm_len = 0
    for i in range(48):
        ang = 2.0 * np.pi * i / 48.0
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        length = 0
        for step in range(1, max(h, w)):
            x = int(round(hx + dx * step))
            y = int(round(hy + dy * step))
            if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
                break
            length = step
        arm_len = max(arm_len, length)
    ys, xs = np.where(mask > 0)
    span = float(max((xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h))
    thick = float(dt.max()) / float(max(h, w))
    return float(arm_len / np.hypot(h, w)), span, thick


def classify_near_shot(
    bgr: np.ndarray,
    *,
    max_side: int = 800,
) -> tuple[bool, dict]:
    """True when the frame looks like a near/mid blade-dominant shot."""
    small, _ = _resize_max_side_bgr(bgr, max_side)
    kept, _hub, _n, geo = _keep_hub_connected(_smooth_candidates(small))
    cov = float((kept > 0).mean())
    elong = float(_elongation(kept)) if kept.max() else 0.0
    arm, span, thick = _longest_arm_frac(kept)
    # Tuned on 2018: keep 0190/0058/0004; drop 0109-class whole-turbine far shots.
    near = bool(
        (arm >= 0.20 and cov >= 0.008)
        or (span >= 0.50 and elong >= 3.0)
        or (arm >= 0.28)
        or (thick >= 0.05 and span >= 0.35)
    )
    meta = {
        "near": near,
        "cov": round(cov, 4),
        "arm": round(arm, 4),
        "span": round(span, 4),
        "thick": round(thick, 4),
        "elong": round(elong, 2),
        "geo": geo,
    }
    return near, meta


def _folder_fingerprint(folder: Path) -> str:
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return f"n={len(names)}|first={names[0] if names else ''}|last={names[-1] if names else ''}"


def list_near_images(
    folder: Path,
    *,
    cache_path: Path | None = None,
    limit: int = 500,
) -> tuple[list[Path], dict]:
    """List near-shot images; cache JSON next to results so the picker stays fast."""
    folder = folder.resolve()
    if cache_path is None:
        cache_path = Path(__file__).resolve().parents[2] / "data" / "near_shots_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp = _folder_fingerprint(folder)
    year_key = folder.name
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    entry = cache.get(year_key)
    if (
        isinstance(entry, dict)
        and entry.get("version") == FILTER_VERSION
        and entry.get("fingerprint") == fp
        and isinstance(entry.get("near"), list)
    ):
        names = entry["near"][:limit]
        return [folder / n for n in names if (folder / n).is_file()], {
            "cached": True,
            "near": len(entry["near"]),
            "far": int(entry.get("far", 0)),
            "total": int(entry.get("total", 0)),
        }

    all_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    near_names: list[str] = []
    far_names: list[str] = []
    t0 = time.time()
    for p in all_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            far_names.append(p.name)
            continue
        ok, _meta = classify_near_shot(bgr)
        (near_names if ok else far_names).append(p.name)
    cache[year_key] = {
        "version": FILTER_VERSION,
        "fingerprint": fp,
        "near": near_names,
        "far": len(far_names),
        "total": len(all_paths),
        "secs": round(time.time() - t0, 1),
    }
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return [folder / n for n in near_names[:limit]], {
        "cached": False,
        "near": len(near_names),
        "far": len(far_names),
        "total": len(all_paths),
        "secs": cache[year_key]["secs"],
    }
