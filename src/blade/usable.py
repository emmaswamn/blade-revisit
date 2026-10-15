"""Stage-1 usable filter: drop frames that should not enter matching.

See docs/pipeline-stages.md 「边不可读 → reject」/「多叶/轮毂」.
Near-shot filter keeps composition scale; this gate drops:
  - 0687-class flat leaf/sky mush (edge unreadable)
  - 0600-class multi-blade / hub closeups (题材边界，不硬抠整机)
  - 0782/0783/0787/0788 complex structure (近景，但不是这期要抠的叶)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from blade.blade_mask import (
    _elongation,
    _geometry_seed,
    _hub_from_distance,
    _keep_hub_connected,
    _resize_max_side_bgr,
    _smooth_candidates,
    _sobel_mag,
    estimate_blade_mask,
)
from blade.lighting import estimate_lighting
from blade.near_shot import list_near_images

FILTER_VERSION = 6  # 0685: reject overcast seed-on-water

# Near shots, but not a single blade this phase can segment.
# Marked by name — not a far-shot reject.
COMPLEX_STRUCTURE_NAMES = frozenset(
    {
        "DJI_0782.JPG",
        "DJI_0783.JPG",
        "DJI_0787.JPG",
        "DJI_0788.JPG",
    }
)


def is_complex_structure(name: str) -> bool:
    return Path(name).name in COMPLEX_STRUCTURE_NAMES


def _polar_arm_peaks(
    mask: np.ndarray,
    hub: tuple[int, int] | None,
    *,
    n_rays: int = 90,
    min_len_frac: float = 0.12,
    peak_ratio: float = 0.55,
    merge_deg: float = 40.0,
) -> list[tuple[float, float]]:
    """Distinct polar length peaks around hub → candidate blade arms."""
    if hub is None or mask.max() == 0:
        return []
    h, w = mask.shape[:2]
    hx, hy = hub
    lengths: list[float] = []
    for i in range(n_rays):
        ang = 2.0 * np.pi * i / n_rays
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        length = 0
        for step in range(1, max(h, w)):
            x = int(round(hx + dx * step))
            y = int(round(hy + dy * step))
            if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
                break
            length = step
        lengths.append(float(length))
    L = np.asarray(lengths, dtype=np.float32)
    ker = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    ker /= float(ker.sum())
    Ls = np.convolve(np.r_[L[-2:], L, L[:2]], ker, mode="valid")
    thr = max(
        min_len_frac * max(h, w),
        peak_ratio * float(Ls.max()) if float(Ls.max()) > 0 else 1e9,
    )
    raw: list[tuple[float, float]] = []
    for i in range(n_rays):
        if Ls[i] >= thr and Ls[i] >= Ls[(i - 1) % n_rays] and Ls[i] >= Ls[(i + 1) % n_rays]:
            raw.append((i * 360.0 / n_rays, float(Ls[i])))
    if not raw:
        return []
    raw.sort()
    merged: list[list[float]] = [list(raw[0])]
    for ang, val in raw[1:]:
        if ang - merged[-1][0] < merge_deg:
            if val > merged[-1][1]:
                merged[-1] = [ang, val]
        else:
            merged.append([ang, val])
    if len(merged) > 1 and (merged[0][0] + 360.0 - merged[-1][0]) < merge_deg:
        if merged[-1][1] > merged[0][1]:
            merged[0] = merged[-1]
        merged.pop()
    return [(float(a), float(v)) for a, v in merged]


def assess_multi_blade_hub(bgr: np.ndarray) -> tuple[bool, dict]:
    """True when frame looks like hub + multi blade roots (reject; not stage-1).

    Uses hub-connected smooth mass (not the often-wrong overcast seed):
    ≥3 polar arms with stubby elongation, or 2 well-separated stubby arms.
    Tuned so DJI_0600–0603 reject; 0190/0752/0713/0624 keep.
    """
    meta: dict = {
        "multi_blade_hub": False,
        "n_arms": 0,
        "elong": 0.0,
        "thick": 0.0,
        "min_sep": 0.0,
    }
    if bgr is None or bgr.size == 0:
        return False, meta
    small, _ = _resize_max_side_bgr(bgr, 1280)
    kept, _hub, _n, _geo = _keep_hub_connected(_smooth_candidates(small))
    if kept.max() == 0:
        return False, meta
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, k)
    hub, dt = _hub_from_distance(closed)
    h, w = closed.shape[:2]
    if hub is None:
        return False, meta
    # tip-of-frame hubs are usually sky weld — prefer thickest below top band
    if hub[1] < 0.08 * h:
        dt2 = dt.copy()
        dt2[: int(0.10 * h)] = 0.0
        _mn, mx, _mnl, max_l = cv2.minMaxLoc(dt2)
        if mx > 3.0:
            hub = (int(max_l[0]), int(max_l[1]))
    peaks = _polar_arm_peaks(closed, hub)
    elong = float(_elongation(closed)) if closed.max() else 0.0
    thick = float(dt.max()) / float(max(h, w))
    n_arms = len(peaks)
    min_sep = 0.0
    if n_arms >= 2:
        angs = sorted(a for a, _ in peaks)
        seps = [angs[i + 1] - angs[i] for i in range(n_arms - 1)]
        seps.append(angs[0] + 360.0 - angs[-1])
        min_sep = float(min(seps))
    meta.update(
        n_arms=n_arms,
        elong=round(elong, 2),
        thick=round(thick, 3),
        min_sep=round(min_sep, 1),
        peaks=[round(a, 1) for a, _ in peaks],
        hub_xy=[int(hub[0]), int(hub[1])],
    )
    # 3+ radiating arms on a stubby hub mass (0600–0603)
    if n_arms >= 3 and elong < 3.5:
        meta["multi_blade_hub"] = True
        return True, meta
    # 2 arms, widely separated, stubby+thick (0604/0605 neighborhood)
    if (
        n_arms == 2
        and elong < 1.6
        and thick > 0.08
        and 90.0 <= min_sep <= 140.0
    ):
        meta["multi_blade_hub"] = True
        return True, meta
    return False, meta


def _rim_edge_stats(
    bgr: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, int]:
    """Return (coverage, rim_sobel_mean, rim_canny_frac, ring_px)."""
    h, w = mask.shape[:2]
    cov = float((mask > 0).mean())
    if mask.max() == 0:
        return cov, 0.0, 0.0, 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = _sobel_mag(gray)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    u8 = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ring = (cv2.dilate(u8, k) - cv2.erode(u8, k)) > 0
    n = int(ring.sum())
    if n < 50:
        return cov, 0.0, 0.0, n
    rim_sobel = float(edge[ring].mean())
    rim_canny = float((canny > 0)[ring].mean())
    return cov, rim_sobel, rim_canny, n


def assess_mask_edge_usable(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    lighting: str = "sunny",
) -> tuple[bool, str, dict]:
    """True when mask boundary looks edge-readable enough for stage 1.2.

    Reject signatures (0687): tiny seed + weak rim, or almost no rim energy.
    Keep thin-but-sharp blades (0059) via rim_canny / rim_sobel.
    """
    cov, rim_sobel, rim_canny, ring_px = _rim_edge_stats(bgr, mask)
    meta = {
        "usable": True,
        "reason": "ok",
        "lighting": lighting,
        "cov": round(cov, 4),
        "rim_sobel": round(rim_sobel, 2),
        "rim_canny": round(rim_canny, 4),
        "ring_px": ring_px,
    }
    if mask.max() == 0:
        meta.update(usable=False, reason="empty_mask")
        return False, "empty_mask", meta
    # 0687: sky remnant seed — small + soft boundary
    if cov < 0.035 and rim_canny < 0.06:
        meta.update(usable=False, reason="small_weak_rim")
        return False, "small_weak_rim", meta
    # edge vanishing into grey sky/water (0684-class soft weld)
    if rim_sobel < 22.0 and rim_canny < 0.04:
        meta.update(usable=False, reason="edge_unreadable")
        return False, "edge_unreadable", meta
    # overcast: tiny seed with middling rim still unreliable for matching
    if lighting == "overcast" and cov < 0.04 and rim_sobel < 35.0:
        meta.update(usable=False, reason="overcast_weak_seed")
        return False, "overcast_weak_seed", meta
    # 0685: geometric seed landed on water/marsh — small cov, mid/lower blob, flat
    if lighting == "overcast" and cov < 0.08 and len(np.where(mask > 0)[0]) > 40:
        ys, xs = np.where(mask > 0)
        h, w = mask.shape[:2]
        cy = float(ys.mean()) / float(max(h, 1))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        std = cv2.blur(
            (gray.astype(np.float32) - cv2.blur(gray.astype(np.float32), (15, 15))) ** 2,
            (15, 15),
        )
        std = np.sqrt(np.maximum(std, 0.0))
        mean_std = float(std[mask > 0].mean())
        meta["seed_cy"] = round(cy, 3)
        meta["seed_mean_std"] = round(mean_std, 2)
        if cy > 0.40 and mean_std < 10.0 and cov < 0.06:
            meta.update(usable=False, reason="seed_is_water")
            return False, "seed_is_water", meta
    # 0687-class: peel may raise rim a bit, still soft mush at small cov
    if lighting == "overcast" and cov < 0.10 and rim_canny < 0.075 and rim_sobel < 38.0:
        meta.update(usable=False, reason="overcast_soft_rim")
        return False, "overcast_soft_rim", meta
    # 0578-class: stubby sky-shelf with hub glued to L/R border.
    # Tip-from-corner blades (0713) after expand can also peak on the frame edge —
    # require stubby elong so real leaf arms are not rejected.
    ys, xs = np.where(mask > 0)
    if lighting == "overcast" and len(xs) > 80:
        # rough hub = distance peak
        dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _mn, _mx, _mn_l, max_l = cv2.minMaxLoc(dt)
        hx, hy = int(max_l[0]), int(max_l[1])
        h, w = mask.shape[:2]
        meta["hub_xy"] = [hx, hy]
        elong = float(_elongation(mask)) if mask.max() else 0.0
        meta["elong"] = round(elong, 2)
        if (hx <= 6 or hx >= w - 7) and cov > 0.05 and elong < 3.0:
            meta.update(usable=False, reason="border_hub_weld")
            return False, "border_hub_weld", meta
    # overcast: fat low-texture plate touching top = sky/ground weld (0735-class residual)
    if lighting == "overcast" and cov > 0.28:
        h, w = mask.shape[:2]
        top = mask[: max(1, int(0.12 * h))] > 0
        top_frac = float(top.sum()) / float(max(int((mask > 0).sum()), 1))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        u8 = (mask > 0).astype(np.uint8) * 255
        # hub-ish dens proxy: mean local std inside mask (flat paint/sky weld → low)
        std = cv2.blur(
            (gray.astype(np.float32) - cv2.blur(gray.astype(np.float32), (15, 15))) ** 2,
            (15, 15),
        )
        std = np.sqrt(np.maximum(std, 0.0))
        mean_std = float(std[mask > 0].mean()) if (mask > 0).any() else 0.0
        meta["top_frac"] = round(top_frac, 4)
        meta["mean_std"] = round(mean_std, 2)
        if top_frac > 0.02 and mean_std < 5.5 and rim_canny < 0.08:
            meta.update(usable=False, reason="sky_ground_weld")
            return False, "sky_ground_weld", meta
        # lower rim heavily on textured ground → house/tree weld
        y0 = int(0.55 * h)
        lower = mask[y0:] > 0
        if lower.any():
            canny_full = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
            lower_canny = float((canny_full[y0:] > 0)[lower].mean())
            meta["lower_canny"] = round(lower_canny, 4)
            if lower_canny > 0.12 and rim_sobel < 28.0:
                meta.update(usable=False, reason="ground_rim_weld")
                return False, "ground_rim_weld", meta
    return True, "ok", meta


def classify_blade_usable(
    bgr: np.ndarray,
    *,
    year: str | None = None,
    max_side: int = 1280,
    use_full_mask: bool = False,
) -> tuple[bool, dict]:
    """Fast usable gate on geometry seed (list filter); optional full mask."""
    if bgr is None or bgr.size == 0:
        return False, {"usable": False, "reason": "empty_image"}
    # Composition first: multi-blade/hub closeups — don't bother with seed edges
    is_multi, multi_meta = assess_multi_blade_hub(bgr)
    if is_multi:
        meta = {
            "usable": False,
            "reason": "multi_blade_hub",
            "lighting": "?",
            **{f"mb_{k}": v for k, v in multi_meta.items() if k != "multi_blade_hub"},
        }
        return False, meta
    small, _ = _resize_max_side_bgr(bgr, max_side)
    lit = estimate_lighting(small, year=year).lighting
    if use_full_mask:
        mr = estimate_blade_mask(bgr, year=year, lighting=lit)
        mask = cv2.resize(
            mr.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        geo_note = ""
    else:
        seed, _hub, _n, geo = _geometry_seed(small, lit)
        mask = seed
        geo_note = geo
    ok, reason, meta = assess_mask_edge_usable(small, mask, lighting=lit)
    meta["reason"] = reason
    meta["usable"] = ok
    if not use_full_mask:
        meta["seed_geo"] = geo_note[:120]
    return ok, meta


def _folder_fingerprint(folder: Path) -> str:
    names = sorted(
        p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return f"n={len(names)}|first={names[0] if names else ''}|last={names[-1] if names else ''}"


def list_usable_near_images(
    folder: Path,
    *,
    cache_path: Path | None = None,
    limit: int = 500,
    year: str | None = None,
) -> tuple[list[Path], dict]:
    """Near shots minus edge-unreadable rejects; cached like near_shot."""
    folder = folder.resolve()
    year_key = year or folder.name
    if cache_path is None:
        cache_path = (
            Path(__file__).resolve().parents[2] / "data" / "usable_shots_cache.json"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp = _folder_fingerprint(folder)

    near_paths, near_filt = list_near_images(folder, limit=5000)
    near_names = [p.name for p in near_paths]

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
        and entry.get("near_n") == len(near_names)
        and isinstance(entry.get("usable"), list)
    ):
        names = entry["usable"][:limit]
        return [folder / n for n in names if (folder / n).is_file()], {
            "cached": True,
            "usable": len(entry["usable"]),
            "rejected": int(entry.get("rejected", 0)),
            "near": len(near_names),
            "total": int(near_filt.get("total", 0)),
            "far": int(near_filt.get("far", 0)),
            "reasons": entry.get("reasons", {}),
        }

    usable_names: list[str] = []
    rejected_names: list[str] = []
    reasons: dict[str, int] = {}
    t0 = time.time()
    for p in near_paths:
        if is_complex_structure(p.name):
            rejected_names.append(p.name)
            reasons["complex_structure"] = reasons.get("complex_structure", 0) + 1
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            rejected_names.append(p.name)
            reasons["unreadable"] = reasons.get("unreadable", 0) + 1
            continue
        ok, meta = classify_blade_usable(bgr, year=year_key)
        if ok:
            usable_names.append(p.name)
        else:
            rejected_names.append(p.name)
            r = str(meta.get("reason", "reject"))
            reasons[r] = reasons.get(r, 0) + 1
    cache[year_key] = {
        "version": FILTER_VERSION,
        "fingerprint": fp,
        "near_n": len(near_names),
        "usable": usable_names,
        "rejected": len(rejected_names),
        "rejected_names": rejected_names,
        "reasons": reasons,
        "total": int(near_filt.get("total", 0)),
        "secs": round(time.time() - t0, 1),
    }
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return [folder / n for n in usable_names[:limit]], {
        "cached": False,
        "usable": len(usable_names),
        "rejected": len(rejected_names),
        "near": len(near_names),
        "total": int(near_filt.get("total", 0)),
        "far": int(near_filt.get("far", 0)),
        "reasons": reasons,
        "secs": cache[year_key]["secs"],
    }
