"""Background / station prior: sunny 2018 query → 2017 usable gallery.

v1: ~dilated blade mask + spatial HSV histograms (fast, cacheable).
v2: color+edge blocks → coarse class → multi-anchor vote (see docs).
Not a rust/damage matcher — only weak viewpoint prior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from blade.blade_mask import estimate_blade_mask
from blade.usable import FILTER_VERSION, list_usable_near_images

CACHE_TAG = "bg_sig_v1"
_DEFAULT_CACHE_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "gallery_features" / CACHE_TAG
)

# Spatial HSV signature knobs (bump CACHE_TAG if changed)
_MAX_SIDE = 640
_GRID_ROWS = 3
_GRID_COLS = 4
_H_BINS, _S_BINS, _V_BINS = 16, 8, 8
_BLADE_DILATE = 15  # px at full-res-ish; applied after resize too via scaled k

# v2 block knobs (bump CACHE_TAG_V2 when geometry/merge changes)
_BLOCK_MAX_SIDE = 720
_BLOCK_K = 6
_BLOCK_MIN_AREA_FRAC = 0.012
_BLOCK_TOP_N = 8
# Same-label fragments closer than this (norm) merge into one super-anchor
_MERGE_DIST = {
    "veg": 0.36,
    "sky": 0.32,
    "water": 0.28,
    "structure": 0.22,
    "other": 0.28,
}
_MERGE_EDGE_MAX_DELTA = {
    "veg": 16.0,  # open field ≠ tree belt
    "sky": 24.0,
    "water": 18.0,
    "structure": 28.0,
    "other": 22.0,
}
_CLASS_COLORS = {
    "sky": (220, 180, 60),  # BGR cyan-ish
    "water": (200, 120, 40),  # blue-orange
    "veg": (60, 200, 60),  # green
    "structure": (40, 80, 220),  # red-ish
    "other": (160, 160, 160),
}


@dataclass
class BgSig:
    name: str
    hist: np.ndarray  # float32 L2-normalized
    bg_frac: float
    mask_detail: str


@dataclass
class BgBlock:
    """One color+edge region on background (v2 anchor candidate)."""

    idx: int
    label: str  # sky|water|veg|structure|other
    area: int
    cx: float  # normalized 0..1
    cy: float
    edge_mean: float
    mean_h: float
    mean_s: float
    mean_v: float
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray = field(repr=False)  # bool HxW on working size


@dataclass
class BgBlocksResult:
    image_id: str
    small_bgr: np.ndarray
    bg: np.ndarray  # uint8 0/255
    blocks: list[BgBlock]
    anchors: list[BgBlock]  # top-N by area
    bg_frac: float
    mask_detail: str


def _resize_max_side(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    scale = m / max_side
    small = cv2.resize(
        bgr,
        (int(round(w / scale)), int(round(h / scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def background_mask_from_blade(
    blade_mask: np.ndarray, *, dilate_px: int = _BLADE_DILATE
) -> np.ndarray:
    """bg=255 where not blade; dilate blade so rim does not leak into bg."""
    if blade_mask.max() == 0:
        return np.ones_like(blade_mask) * 255
    k = max(1, int(dilate_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
    fat = cv2.dilate(blade_mask, ker)
    return np.where(fat > 0, 0, 255).astype(np.uint8)


def _zone_name(cx: float, cy: float) -> str:
    col = "L" if cx < 0.33 else ("R" if cx > 0.66 else "M")
    row = "U" if cy < 0.33 else ("D" if cy > 0.66 else "M")
    return f"{row}{col}"


_LABEL_CODE = {
    "sky": "S",
    "water": "W",
    "veg": "V",
    "structure": "T",  # T≠S so tags don't collide with sky
    "other": "O",
}


def _label_block(
    *,
    mean_h: float,
    mean_s: float,
    mean_v: float,
    edge_mean: float,
    cy: float,
    veg_frac: float,
    aspect: float = 1.0,
    fill: float = 0.5,
) -> str:
    """Coarse class — wrong is ok; voting uses multi-block."""
    weak_edge = edge_mean < 18.0
    strong_edge = edge_mean > 32.0
    # Tree belts / roads: strong edge or flat horizontal strip — before veg
    if strong_edge and (aspect >= 2.2 or mean_v < 70):
        return "structure"
    if aspect >= 3.0 and fill > 0.25 and edge_mean > 22 and cy < 0.55:
        return "structure"
    # sky: bright, flat, upper-ish
    if mean_v >= 155 and mean_s < 45 and weak_edge and cy < 0.55:
        return "sky"
    # water: mid-grey, flat, mid/lower
    if (
        mean_s < 50
        and weak_edge
        and 70 <= mean_v <= 175
        and cy > 0.35
        and veg_frac < 0.20
    ):
        return "water"
    # vegetation / field
    if veg_frac > 0.35 or (mean_s >= 40 and 25 <= mean_h <= 100):
        return "veg"
    # structure: hedges / houses / roads (fallback)
    if strong_edge or (mean_s < 55 and mean_v >= 100 and edge_mean > 24):
        return "structure"
    if mean_v >= 170 and mean_s < 40 and cy < 0.45:
        return "sky"
    return "other"


def _masks_near(a: np.ndarray, b: np.ndarray, *, dilate: int = 7) -> bool:
    """True if masks touch / nearly touch after light dilate."""
    if not a.any() or not b.any():
        return False
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
    )
    ad = cv2.dilate(a.astype(np.uint8), k)
    return bool((ad.astype(bool) & b).any())


def _merge_same_label_blocks(blocks: list[BgBlock]) -> list[BgBlock]:
    """Collapse nearby same-class fragments so one field ≠ three V votes."""
    if len(blocks) <= 1:
        return blocks

    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, a in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            b = blocks[j]
            if a.label != b.label:
                continue
            if abs(a.edge_mean - b.edge_mean) > _MERGE_EDGE_MAX_DELTA.get(
                a.label, 22.0
            ):
                continue
            dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
            lim = _MERGE_DIST.get(a.label, 0.28)
            # veg/sky: also merge when fragments nearly touch (one field, far centroids)
            dil = 22 if a.label == "veg" else 9
            near = dist <= lim or (
                a.label in ("veg", "sky") and _masks_near(a.mask, b.mask, dilate=dil)
            )
            if not near:
                continue
            union(i, j)

    groups: dict[int, list[BgBlock]] = {}
    for i, b in enumerate(blocks):
        groups.setdefault(find(i), []).append(b)

    merged: list[BgBlock] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        mask = np.zeros_like(members[0].mask, dtype=bool)
        for m in members:
            mask |= m.mask
        area = int(mask.sum())
        if area < 1:
            continue
        ys, xs = np.where(mask)
        tot = float(sum(m.area for m in members)) or 1.0
        w_edge = sum(m.edge_mean * m.area for m in members) / tot
        w_h = sum(m.mean_h * m.area for m in members) / tot
        w_s = sum(m.mean_s * m.area for m in members) / tot
        w_v = sum(m.mean_v * m.area for m in members) / tot
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        merged.append(
            BgBlock(
                idx=0,
                label=members[0].label,
                area=area,
                cx=float(xs.mean()) / float(max(mask.shape[1], 1)),
                cy=float(ys.mean()) / float(max(mask.shape[0], 1)),
                edge_mean=float(w_edge),
                mean_h=float(w_h),
                mean_s=float(w_s),
                mean_v=float(w_v),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                mask=mask,
            )
        )
    return merged


def extract_bg_blocks(
    image_id: str,
    bgr: np.ndarray,
    *,
    year: str | None = None,
    k_colors: int = _BLOCK_K,
    top_n: int = _BLOCK_TOP_N,
) -> BgBlocksResult:
    """Color+edge regions on background; coarse sky/water/veg/structure labels."""
    mr = estimate_blade_mask(bgr, year=year)
    small, _ = _resize_max_side(bgr, _BLOCK_MAX_SIDE)
    blade_s = cv2.resize(
        mr.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    dil = max(3, int(round(_BLADE_DILATE * small.shape[1] / max(bgr.shape[1], 1))))
    bg = background_mask_from_blade(blade_s, dilate_px=dil)
    bg_frac = float((bg > 0).mean())
    h, w = small.shape[:2]
    frame = float(h * w)

    # Smooth then Lab k-means on bg pixels (edge barrier lightly splits merges)
    blur = cv2.bilateralFilter(small, d=7, sigmaColor=40, sigmaSpace=7)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge = cv2.magnitude(edge, cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_u8 = np.clip(edge, 0, 255).astype(np.uint8)
    barrier = cv2.dilate(
        (edge_u8 > 40).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # veg proxy (same spirit as blade_mask._vegetation_mask)
    hh, ss, vv = cv2.split(hsv)
    veg = ((hh >= 30) & (hh <= 95) & (ss >= 35)) | (
        ((hh <= 30) | (hh >= 170)) & (ss >= 40)
    )

    bg_bool = bg > 0
    ys, xs = np.where(bg_bool & (barrier == 0))
    if len(xs) < 200:
        ys, xs = np.where(bg_bool)
    blocks: list[BgBlock] = []
    if len(xs) >= 80:
        samples = lab[ys, xs].astype(np.float32)
        # subsample for speed
        if len(samples) > 12000:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(samples), 12000, replace=False)
            train = samples[pick]
        else:
            train = samples
        k = int(max(2, min(k_colors, len(train) // 40)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        _compact, _label_train, centers = cv2.kmeans(
            train, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        # assign all bg pixels (incl. barrier) to nearest center
        all_ys, all_xs = np.where(bg_bool)
        pts = lab[all_ys, all_xs].astype(np.float32)
        # distances to centers
        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_flat = np.argmin(d2, axis=1).astype(np.int32)
        label_map = np.full((h, w), -1, dtype=np.int32)
        label_map[all_ys, all_xs] = labels_flat

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        min_area = int(_BLOCK_MIN_AREA_FRAC * frame)
        bid = 0
        for ci in range(k):
            comp_u8 = ((label_map == ci) & bg_bool).astype(np.uint8) * 255
            comp_u8 = cv2.morphologyEx(comp_u8, cv2.MORPH_OPEN, k3)
            num, cc, stats, _cent = cv2.connectedComponentsWithStats(comp_u8, 8)
            for j in range(1, num):
                area = int(stats[j, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                m = cc == j
                my, mx = np.where(m)
                cx = float(mx.mean()) / float(max(w, 1))
                cy = float(my.mean()) / float(max(h, 1))
                mean_h = float(hh[m].mean())
                mean_s = float(ss[m].mean())
                mean_v = float(vv[m].mean())
                edge_mean = float(edge_u8[m].mean())
                veg_frac = float(veg[m].mean())
                x0 = int(stats[j, cv2.CC_STAT_LEFT])
                y0 = int(stats[j, cv2.CC_STAT_TOP])
                bw = int(stats[j, cv2.CC_STAT_WIDTH])
                bh = int(stats[j, cv2.CC_STAT_HEIGHT])
                aspect = float(bw) / float(max(bh, 1))
                fill = float(area) / float(max(bw * bh, 1))
                lab_name = _label_block(
                    mean_h=mean_h,
                    mean_s=mean_s,
                    mean_v=mean_v,
                    edge_mean=edge_mean,
                    cy=cy,
                    veg_frac=veg_frac,
                    aspect=aspect,
                    fill=fill,
                )
                bid += 1
                blocks.append(
                    BgBlock(
                        idx=bid,
                        label=lab_name,
                        area=area,
                        cx=cx,
                        cy=cy,
                        edge_mean=edge_mean,
                        mean_h=mean_h,
                        mean_s=mean_s,
                        mean_v=mean_v,
                        x0=x0,
                        y0=y0,
                        x1=x0 + bw - 1,
                        y1=y0 + bh - 1,
                        mask=m,
                    )
                )

    blocks = _merge_same_label_blocks(blocks)
    blocks.sort(key=lambda b: b.area, reverse=True)
    # re-index by area rank for stable anchor numbers
    for i, b in enumerate(blocks, 1):
        b.idx = i
    anchors = blocks[:top_n]
    return BgBlocksResult(
        image_id=image_id,
        small_bgr=small,
        bg=bg,
        blocks=blocks,
        anchors=anchors,
        bg_frac=bg_frac,
        mask_detail=mr.detail[:160],
    )


def draw_bg_blocks_debug(res: BgBlocksResult, *, show_all: bool = False) -> np.ndarray:
    """Overlay class colors + numbered anchors for eyeball check."""
    out = res.small_bgr.copy()
    blade_dim = res.bg == 0
    out[blade_dim] = (out[blade_dim] * 0.22).astype(np.uint8)

    show = res.blocks if show_all else res.anchors
    tint = np.zeros_like(out)
    for b in show:
        color = _CLASS_COLORS.get(b.label, _CLASS_COLORS["other"])
        tint[b.mask] = color
    mix = out.copy()
    on = np.zeros(out.shape[:2], dtype=bool)
    for b in show:
        on |= b.mask
    mix[on] = cv2.addWeighted(out, 0.45, tint, 0.55, 0)[on]
    out = mix

    for b in show:
        color = _CLASS_COLORS.get(b.label, _CLASS_COLORS["other"])
        cnts, _ = cv2.findContours(
            b.mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
        tx = int(b.cx * out.shape[1])
        ty = int(b.cy * out.shape[0])
        tag = f"#{b.idx}{_LABEL_CODE.get(b.label, 'O')}{_zone_name(b.cx, b.cy)}"
        cv2.circle(out, (tx, ty), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            tag,
            (tx + 8, ty - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            tag,
            (tx + 8, ty - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    legend = "S=sky W=water V=veg T=struct  tag=#id+code+zone"
    cv2.putText(
        out,
        f"blocks={len(res.blocks)} anchors={len(res.anchors)} bg={res.bg_frac:.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        legend,
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_bg_block_sig_fast(bgr: np.ndarray, sig: "BgBlockSig") -> np.ndarray:
    """Fast preview from cached anchors only (no mask re-extract).

    npz has no region masks — draw class-colored dots + tags at (cx,cy).
    Good enough for /bg layout; use extract_bg_blocks for filled contours.
    """
    small, _ = _resize_max_side(bgr, _BLOCK_MAX_SIDE)
    out = small.copy()
    h, w = out.shape[:2]
    for a in sig.anchors:
        color = _CLASS_COLORS.get(a.label, _CLASS_COLORS["other"])
        tx = int(float(a.cx) * w)
        ty = int(float(a.cy) * h)
        # area → rough radius hint (norm frame)
        r = int(max(10, min(48, np.sqrt(max(a.area, 1)) * 0.12)))
        cv2.circle(out, (tx, ty), r, color, 2, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
        tag = f"#{a.idx}{_LABEL_CODE.get(a.label, 'O')}{_zone_name(a.cx, a.cy)}"
        cv2.putText(
            out, tag, (tx + 8, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            out,
            tag,
            (tx + 8, ty - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"anchors={len(sig.anchors)} bg={sig.bg_frac:.2f}  (npz fast)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "S=sky W=water V=veg T=struct  tag=#id+code+zone",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


# Distinct colors for matched pair ids (BGR)
_PAIR_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 220, 0),
    (0, 128, 255),
    (255, 180, 0),
    (180, 0, 255),
    (255, 255, 0),
    (128, 255, 128),
]


def _anchor_by_idx(res: BgBlocksResult, idx: int) -> BgBlock | None:
    for b in res.anchors:
        if b.idx == idx:
            return b
    for b in res.blocks:
        if b.idx == idx:
            return b
    return None


def _crop_block_texture(
    res: BgBlocksResult, block: BgBlock, *, out_h: int = 88, out_w: int = 120
) -> np.ndarray:
    """Tight crop of block interior for texture eyeballing."""
    ys, xs = np.where(block.mask)
    if len(xs) < 8:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = 2
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(res.small_bgr.shape[0] - 1, y1 + pad)
    x1 = min(res.small_bgr.shape[1] - 1, x1 + pad)
    crop = res.small_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
    # mute non-mask pixels inside bbox
    local = block.mask[y0 : y1 + 1, x0 : x1 + 1]
    crop[~local] = (crop[~local] * 0.15).astype(np.uint8)
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _draw_match_overlay(
    res: BgBlocksResult,
    pair_ids: dict[int, int],
    *,
    title: str,
) -> np.ndarray:
    """Dim blade; highlight matched anchors with shared pair color + M#."""
    out = res.small_bgr.copy()
    out[res.bg == 0] = (out[res.bg == 0] * 0.18).astype(np.uint8)
    # mute unmatched bg a bit
    matched_mask = np.zeros(out.shape[:2], dtype=bool)
    for b in res.anchors:
        if b.idx in pair_ids:
            matched_mask |= b.mask
    unmatched = (res.bg > 0) & (~matched_mask)
    out[unmatched] = (out[unmatched] * 0.45).astype(np.uint8)

    for b in res.anchors:
        if b.idx not in pair_ids:
            continue
        mid = pair_ids[b.idx]
        color = _PAIR_COLORS[(mid - 1) % len(_PAIR_COLORS)]
        tint = np.zeros_like(out)
        tint[b.mask] = color
        out[b.mask] = cv2.addWeighted(out, 0.4, tint, 0.6, 0)[b.mask]
        cnts, _ = cv2.findContours(
            b.mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cnts, -1, color, 3, cv2.LINE_AA)
        tx = int(b.cx * out.shape[1])
        ty = int(b.cy * out.shape[0])
        tag = f"M{mid} #{b.idx}{_LABEL_CODE.get(b.label, 'O')}"
        cv2.putText(
            out, tag, (tx + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            out, tag, (tx + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "same M# = algorithm thinks similar (texture crop below)",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def compose_bg_match_texture_debug(
    q_res: BgBlocksResult,
    r_res: BgBlocksResult,
    vote: dict,
    *,
    crop_h: int = 88,
) -> np.ndarray:
    """Debug panel: matched blocks share M# colors + side-by-side texture crops.

    Lets you eyeball whether 'similar' pairs actually look alike.
    """
    pairs = list(vote.get("pairs") or [])
    q_map: dict[int, int] = {}
    r_map: dict[int, int] = {}
    for i, p in enumerate(pairs, 1):
        q_map[int(p["q"])] = i
        r_map[int(p["r"])] = i

    left = _draw_match_overlay(
        q_res, q_map, title=f"Q {q_res.image_id}  hits={vote.get('hits')} adj={vote.get('adj_score', 0)}"
    )
    right = _draw_match_overlay(
        r_res,
        r_map,
        title=f"R {r_res.image_id}  score={float(vote.get('score', 0)):.2f}",
    )
    h = max(left.shape[0], right.shape[0])

    def _pad(im: np.ndarray) -> np.ndarray:
        if im.shape[0] == h:
            return im
        canvas = np.zeros((h, im.shape[1], 3), dtype=np.uint8)
        canvas[: im.shape[0]] = im
        return canvas

    left, right = _pad(left), _pad(right)
    gap = np.zeros((h, 10, 3), dtype=np.uint8)
    top = np.hstack([left, gap, right])

    # texture strip: for each match, Q crop | R crop
    strips: list[np.ndarray] = []
    for i, p in enumerate(pairs, 1):
        qb = _anchor_by_idx(q_res, int(p["q"]))
        rb = _anchor_by_idx(r_res, int(p["r"]))
        color = _PAIR_COLORS[(i - 1) % len(_PAIR_COLORS)]
        if qb is None or rb is None:
            continue
        qc = _crop_block_texture(q_res, qb, out_h=crop_h)
        rc = _crop_block_texture(r_res, rb, out_h=crop_h)
        cell_w = qc.shape[1] + 8 + rc.shape[1] + 16
        cell = np.zeros((crop_h + 28, cell_w, 3), dtype=np.uint8)
        cell[24 : 24 + crop_h, 8 : 8 + qc.shape[1]] = qc
        cell[24 : 24 + crop_h, 16 + qc.shape[1] : 16 + qc.shape[1] + rc.shape[1]] = rc
        cv2.rectangle(cell, (0, 0), (cell_w - 1, crop_h + 27), color, 2)
        lab = (
            f"M{i} Q#{p['q']}{_LABEL_CODE.get(p['q_label'], 'O')} | "
            f"R#{p['r']}{_LABEL_CODE.get(p['r_label'], 'O')} aff={p.get('aff', 0)}"
        )
        cv2.putText(
            cell, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
        )
        strips.append(cell)

    if not strips:
        note = np.zeros((40, top.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            note,
            "no matched pairs — nothing to compare",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([top, note])

    # pack strips in rows
    max_w = top.shape[1]
    rows: list[np.ndarray] = []
    row: list[np.ndarray] = []
    row_w = 0
    for cell in strips:
        if row and row_w + cell.shape[1] + 6 > max_w:
            rows.append(np.hstack(row + [np.zeros((row[0].shape[0], 4, 3), dtype=np.uint8)]))
            row, row_w = [], 0
        if row:
            pad = np.zeros((cell.shape[0], 6, 3), dtype=np.uint8)
            row.append(pad)
            row_w += 6
        row.append(cell)
        row_w += cell.shape[1]
    if row:
        rows.append(np.hstack(row))

    packed: list[np.ndarray] = []
    for row_im in rows:
        if row_im.shape[1] < max_w:
            canvas = np.zeros((row_im.shape[0], max_w, 3), dtype=np.uint8)
            canvas[:, : row_im.shape[1]] = row_im
            packed.append(canvas)
        else:
            packed.append(row_im[:, :max_w])

    header = np.zeros((26, max_w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        "texture crops: left=query block  right=gallery block  (same M# = claimed match)",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([top, header] + packed)


def build_bg_match_texture_debug(
    *,
    data_root: Path,
    query_name: str,
    ref_name: str,
    query_year: str = "2018",
    gallery_year: str = "2017",
) -> tuple[np.ndarray, dict]:
    """Load pair, vote, return (panel_bgr, vote_meta)."""
    data_root = Path(data_root)
    q_path = data_root / query_year / Path(query_name).name
    r_path = data_root / gallery_year / Path(ref_name).name
    qb = cv2.imread(str(q_path))
    rb = cv2.imread(str(r_path))
    if qb is None or rb is None:
        raise FileNotFoundError(f"missing {q_path} or {r_path}")
    q_res = extract_bg_blocks(q_path.name, qb, year=query_year)
    r_res = extract_bg_blocks(r_path.name, rb, year=gallery_year)
    q_sig = blocks_result_to_sig(q_res)
    r_sig = blocks_result_to_sig(r_res)
    if q_sig is None or r_sig is None:
        raise RuntimeError("empty blocks")
    vote = score_block_vote(q_sig, r_sig)
    panel = compose_bg_match_texture_debug(q_res, r_res, vote)
    return panel, vote


def _cell_hsv_hist(hsv: np.ndarray, bg: np.ndarray) -> np.ndarray | None:
    if int((bg > 0).sum()) < 40:
        return None
    # OpenCV calcHist with mask
    h = cv2.calcHist([hsv], [0], bg, [_H_BINS], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], bg, [_S_BINS], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], bg, [_V_BINS], [0, 256]).ravel()
    vec = np.concatenate([h, s, v]).astype(np.float64)
    n = float(np.linalg.norm(vec)) + 1e-9
    return (vec / n).astype(np.float32)


def extract_bg_signature(
    image_id: str,
    bgr: np.ndarray,
    *,
    year: str | None = None,
) -> BgSig | None:
    """Build spatial HSV hist on background pixels only."""
    mr = estimate_blade_mask(bgr, year=year)
    blade = mr.mask
    small, _ = _resize_max_side(bgr, _MAX_SIDE)
    # resize mask with same geometry
    blade_s = cv2.resize(
        blade, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    # dilate in resized space (~ proportional)
    dil = max(3, int(round(_BLADE_DILATE * small.shape[1] / max(bgr.shape[1], 1))))
    bg = background_mask_from_blade(blade_s, dilate_px=dil)
    bg_frac = float((bg > 0).mean())
    if bg_frac < 0.08:
        return None

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = small.shape[:2]
    parts: list[np.ndarray] = []
    for ri in range(_GRID_ROWS):
        for ci in range(_GRID_COLS):
            y0 = int(ri * h / _GRID_ROWS)
            y1 = int((ri + 1) * h / _GRID_ROWS)
            x0 = int(ci * w / _GRID_COLS)
            x1 = int((ci + 1) * w / _GRID_COLS)
            cell_bg = bg[y0:y1, x0:x1]
            cell_hsv = hsv[y0:y1, x0:x1]
            hist = _cell_hsv_hist(cell_hsv, cell_bg)
            if hist is None:
                parts.append(np.zeros(_H_BINS + _S_BINS + _V_BINS, dtype=np.float32))
            else:
                parts.append(hist)
    vec = np.concatenate(parts).astype(np.float64)
    n = float(np.linalg.norm(vec)) + 1e-9
    vec = (vec / n).astype(np.float32)
    return BgSig(
        name=image_id,
        hist=vec,
        bg_frac=bg_frac,
        mask_detail=mr.detail[:160],
    )


def bg_cosine(a: BgSig, b: BgSig) -> float:
    return float(np.dot(a.hist, b.hist))


def cache_path(year: str, name: str, cache_root: Path | None = None) -> Path:
    root = cache_root or _DEFAULT_CACHE_ROOT
    return root / year / f"{Path(name).stem}.npz"


def save_bg_sig(path: Path, sig: BgSig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array(sig.name),
        tag=np.array(CACHE_TAG),
        usable_v=np.int32(FILTER_VERSION),
        hist=sig.hist,
        bg_frac=np.float64(sig.bg_frac),
        grid=np.array([_GRID_ROWS, _GRID_COLS, _H_BINS, _S_BINS, _V_BINS]),
    )


def load_bg_sig(path: Path) -> BgSig | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if str(data["tag"]) != CACHE_TAG or int(data["usable_v"]) != FILTER_VERSION:
            return None
        grid = np.array(data["grid"]).tolist()
        if grid != [_GRID_ROWS, _GRID_COLS, _H_BINS, _S_BINS, _V_BINS]:
            return None
        return BgSig(
            name=str(data["name"]),
            hist=np.array(data["hist"], copy=True).astype(np.float32),
            bg_frac=float(data["bg_frac"]),
            mask_detail="cached",
        )
    except (KeyError, ValueError, TypeError):
        return None
    finally:
        data.close()


def build_bg_sig_from_path(path: Path, *, year: str) -> BgSig | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return extract_bg_signature(path.name, bgr, year=year)


def draw_bg_debug(bgr: np.ndarray, *, year: str | None = None) -> np.ndarray:
    """Preview: blade tinted, background kept — for eyeball / handoff."""
    mr = estimate_blade_mask(bgr, year=year)
    small, _ = _resize_max_side(bgr, 960)
    blade = cv2.resize(
        mr.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    dil = max(3, int(round(_BLADE_DILATE * small.shape[1] / max(bgr.shape[1], 1))))
    bg = background_mask_from_blade(blade, dilate_px=dil)
    out = small.copy()
    # dim blade
    blade_area = blade > 0
    out[blade_area] = (out[blade_area] * 0.25).astype(np.uint8)
    # grid
    h, w = out.shape[:2]
    for ri in range(1, _GRID_ROWS):
        y = int(ri * h / _GRID_ROWS)
        cv2.line(out, (0, y), (w - 1, y), (0, 200, 255), 1, cv2.LINE_AA)
    for ci in range(1, _GRID_COLS):
        x = int(ci * w / _GRID_COLS)
        cv2.line(out, (x, 0), (x, h - 1), (0, 200, 255), 1, cv2.LINE_AA)
    # outline bg
    cnts, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (80, 255, 80), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"bg_frac={(bg>0).mean():.2f}  (blade dimmed)",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def search_bg_gallery(
    *,
    data_root: Path,
    query_name: str,
    query_year: str = "2018",
    gallery_year: str = "2017",
    top_k: int = 8,
    limit: int = 0,
    use_cache: bool = True,
    cache_root: Path | None = None,
    preview_dir: Path | None = None,
    gold_name: str = "DJI_0752.JPG",
    log_print: bool = False,
) -> dict:
    """Sunny query → overcast/other-year usable gallery by background similarity."""
    data_root = Path(data_root)
    q_path = data_root / query_year / Path(query_name).name
    if not q_path.is_file():
        raise FileNotFoundError(f"query missing: {q_path}")

    refs, filt = list_usable_near_images(
        data_root / gallery_year, limit=5000, year=gallery_year
    )
    if limit > 0:
        refs = refs[:limit]

    t0 = time.time()
    q_sig = build_bg_sig_from_path(q_path, year=query_year)
    if q_sig is None:
        raise RuntimeError("query bg signature failed (empty bg?)")

    if log_print:
        print(
            f"query {query_year}/{q_path.name} → {gallery_year} usable×{len(refs)} "
            f"(pool≈{filt.get('usable')}) bg_frac={q_sig.bg_frac:.3f}"
        )

    ranked: list[dict] = []
    hit = miss = skip = mem_hit = 0
    for i, path in enumerate(refs, 1):
        key = f"{gallery_year}/{path.name}"
        r_sig = _BG_HSV_MEM.get(key) if use_cache else None
        src = "mem" if r_sig is not None else None
        if r_sig is None and use_cache:
            r_sig = load_bg_sig(cache_path(gallery_year, path.name, cache_root))
            if r_sig is not None:
                _BG_HSV_MEM[key] = r_sig
                src = "disk"
        if r_sig is None:
            miss += 1
            r_sig = build_bg_sig_from_path(path, year=gallery_year)
            if r_sig is None:
                skip += 1
                continue
            if use_cache:
                save_bg_sig(cache_path(gallery_year, path.name, cache_root), r_sig)
                _BG_HSV_MEM[key] = r_sig
            src = "built"
        else:
            hit += 1
            if src == "mem":
                mem_hit += 1

        score = bg_cosine(q_sig, r_sig)
        ranked.append(
            {
                "name": r_sig.name,
                "score": score,
                "bg_frac": r_sig.bg_frac,
            }
        )
        if log_print and (i % 10 == 0 or i == len(refs)):
            print(
                f"  {i}/{len(refs)} hit={hit} mem={mem_hit} miss={miss} skip={skip}"
            )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    top = ranked[:top_k]
    gold_rank = next((i for i, r in enumerate(ranked, 1) if r["name"] == gold_name), None)

    preview_paths: dict[str, str] = {}
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        qb = cv2.imread(str(q_path))
        if qb is not None:
            q_dbg = draw_bg_debug(qb, year=query_year)
            q_out = preview_dir / f"bg_query_{Path(query_name).stem}.jpg"
            cv2.imwrite(str(q_out), q_dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            preview_paths["query"] = q_out.name
        # side-by-side query | top1
        if top:
            rb = cv2.imread(str(data_root / gallery_year / top[0]["name"]))
            if qb is not None and rb is not None:
                left = draw_bg_debug(qb, year=query_year)
                right = draw_bg_debug(rb, year=gallery_year)
                h = max(left.shape[0], right.shape[0])
                def pad(im):
                    if im.shape[0] == h:
                        return im
                    canvas = np.zeros((h, im.shape[1], 3), dtype=np.uint8)
                    canvas[: im.shape[0]] = im
                    return canvas
                left, right = pad(left), pad(right)
                gap = np.zeros((h, 8, 3), dtype=np.uint8)
                panel = np.hstack([left, gap, right])
                cv2.putText(
                    panel,
                    f"bg cos={top[0]['score']:.3f}  {query_year}/{Path(query_name).name} | "
                    f"{gallery_year}/{top[0]['name']}",
                    (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                p1 = preview_dir / "bg_search_top1.jpg"
                cv2.imwrite(str(p1), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                preview_paths["top1"] = p1.name

    return {
        "query": f"{query_year}/{Path(query_name).name}",
        "query_year": query_year,
        "query_id": Path(query_name).name,
        "gallery_year": gallery_year,
        "gallery_n": len(refs),
        "gallery_pool": "usable",
        "cache_hit": hit,
        "cache_mem": mem_hit,
        "cache_miss": miss,
        "skip": skip,
        "gold_name": gold_name,
        "gold_rank": gold_rank,
        "secs": round(time.time() - t0, 1),
        "query_bg_frac": q_sig.bg_frac,
        "direction_note": (
            "sunny query → gallery: background/station prior (spatial HSV on ~blade)"
        ),
        "method": CACHE_TAG,
        "preview": preview_paths,
        "top": [
            {
                "rank": i,
                "name": r["name"],
                "ref_id": r["name"],
                "query_id": Path(query_name).name,
                "score": r["score"],
                "bg_frac": r["bg_frac"],
            }
            for i, r in enumerate(top, 1)
        ],
    }


# --- v2: color+edge block voting -------------------------------------------

CACHE_TAG_V2 = "bg_sig_v2.2"
_DEFAULT_CACHE_ROOT_V2 = (
    Path(__file__).resolve().parents[2] / "data" / "gallery_features" / CACHE_TAG_V2
)

# Process-lifetime caches (survive across HTTP requests; die on server restart)
_BG_BLOCK_MEM: dict[str, BgBlockSig] = {}
_BG_HSV_MEM: dict[str, BgSig] = {}

# Soft label pairs (sunny river often lands as other)
_LABEL_COMPAT = {
    ("sky", "sky"): 1.0,
    ("water", "water"): 1.0,
    ("veg", "veg"): 1.0,
    ("structure", "structure"): 1.0,
    ("other", "other"): 0.6,
    ("water", "other"): 0.75,
    ("other", "water"): 0.75,
    ("sky", "other"): 0.45,
    ("other", "sky"): 0.45,
    ("structure", "other"): 0.55,
    ("other", "structure"): 0.55,
    ("veg", "other"): 0.35,
    ("other", "veg"): 0.35,
    ("water", "sky"): 0.25,  # overcast confusion — weak credit only
    ("sky", "water"): 0.25,
}


@dataclass
class BgBlockFeat:
    """Serializable anchor (no pixel mask) for cache / matching."""

    idx: int
    label: str
    area: int
    cx: float
    cy: float
    edge_mean: float
    mean_h: float
    mean_s: float
    mean_v: float


@dataclass
class BgBlockSig:
    name: str
    bg_frac: float
    anchors: list[BgBlockFeat]


def _block_to_feat(b: BgBlock) -> BgBlockFeat:
    return BgBlockFeat(
        idx=b.idx,
        label=b.label,
        area=b.area,
        cx=b.cx,
        cy=b.cy,
        edge_mean=b.edge_mean,
        mean_h=b.mean_h,
        mean_s=b.mean_s,
        mean_v=b.mean_v,
    )


def blocks_result_to_sig(res: BgBlocksResult) -> BgBlockSig | None:
    if not res.anchors:
        return None
    return BgBlockSig(
        name=res.image_id,
        bg_frac=res.bg_frac,
        anchors=[_block_to_feat(b) for b in res.anchors],
    )


def build_bg_block_sig_from_path(path: Path, *, year: str) -> BgBlockSig | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    res = extract_bg_blocks(path.name, bgr, year=year)
    return blocks_result_to_sig(res)


def cache_path_v2(year: str, name: str, cache_root: Path | None = None) -> Path:
    root = cache_root or _DEFAULT_CACHE_ROOT_V2
    return root / year / f"{Path(name).stem}.npz"


def save_bg_block_sig(path: Path, sig: BgBlockSig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(sig.anchors)
    labels = np.array([a.label for a in sig.anchors])
    np.savez_compressed(
        path,
        name=np.array(sig.name),
        tag=np.array(CACHE_TAG_V2),
        usable_v=np.int32(FILTER_VERSION),
        bg_frac=np.float64(sig.bg_frac),
        n=np.int32(n),
        idx=np.array([a.idx for a in sig.anchors], dtype=np.int32),
        labels=labels,
        area=np.array([a.area for a in sig.anchors], dtype=np.int32),
        cx=np.array([a.cx for a in sig.anchors], dtype=np.float32),
        cy=np.array([a.cy for a in sig.anchors], dtype=np.float32),
        edge=np.array([a.edge_mean for a in sig.anchors], dtype=np.float32),
        mean_h=np.array([a.mean_h for a in sig.anchors], dtype=np.float32),
        mean_s=np.array([a.mean_s for a in sig.anchors], dtype=np.float32),
        mean_v=np.array([a.mean_v for a in sig.anchors], dtype=np.float32),
    )


def _bg_mem_key(year: str, name: str) -> str:
    return f"{CACHE_TAG_V2}/{year}/{Path(name).name}"


def resolve_bg_block_sig(
    path: Path,
    *,
    year: str,
    use_cache: bool = True,
    cache_root: Path | None = None,
) -> tuple[BgBlockSig | None, str]:
    """Return (sig, source) source ∈ mem|disk|built|skip."""
    key = _bg_mem_key(year, path.name)
    if key in _BG_BLOCK_MEM:
        return _BG_BLOCK_MEM[key], "mem"
    if use_cache:
        cached = load_bg_block_sig(cache_path_v2(year, path.name, cache_root))
        if cached is not None:
            _BG_BLOCK_MEM[key] = cached
            return cached, "disk"
    sig = build_bg_block_sig_from_path(path, year=year)
    if sig is None:
        return None, "skip"
    if use_cache:
        save_bg_block_sig(cache_path_v2(year, path.name, cache_root), sig)
    _BG_BLOCK_MEM[key] = sig
    return sig, "built"


def load_bg_block_sig(path: Path) -> BgBlockSig | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if str(data["tag"]) != CACHE_TAG_V2 or int(data["usable_v"]) != FILTER_VERSION:
            return None
        n = int(data["n"])
        labels = data["labels"]
        anchors = [
            BgBlockFeat(
                idx=int(data["idx"][i]),
                label=str(labels[i]),
                area=int(data["area"][i]),
                cx=float(data["cx"][i]),
                cy=float(data["cy"][i]),
                edge_mean=float(data["edge"][i]),
                mean_h=float(data["mean_h"][i]),
                mean_s=float(data["mean_s"][i]),
                mean_v=float(data["mean_v"][i]),
            )
            for i in range(n)
        ]
        return BgBlockSig(
            name=str(data["name"]), bg_frac=float(data["bg_frac"]), anchors=anchors
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    finally:
        data.close()


def _pair_affinity(q: BgBlockFeat, r: BgBlockFeat) -> float:
    """0..1 how well two anchors correspond (class + pose + edge)."""
    lab = _LABEL_COMPAT.get((q.label, r.label), 0.0)
    if lab <= 0:
        return 0.0
    dist = float(np.hypot(q.cx - r.cx, q.cy - r.cy))
    # landmarks are sparse across years — allow a wider pose gate
    dist_lim = 0.62 if q.label in ("structure", "water") else 0.42
    if dist > dist_lim:
        return 0.0
    pos = max(0.0, 1.0 - dist / dist_lim)
    # same vertical band bonus
    q_band = 0 if q.cy < 0.33 else (2 if q.cy > 0.66 else 1)
    r_band = 0 if r.cy < 0.33 else (2 if r.cy > 0.66 else 1)
    if q_band == r_band:
        pos = min(1.0, pos + 0.12)
    e_max = max(q.edge_mean, r.edge_mean, 1.0)
    edge_sim = max(0.0, 1.0 - abs(q.edge_mean - r.edge_mean) / (e_max + 12.0))
    return float(lab * (0.55 * pos + 0.25 * edge_sim + 0.20))


# Single-anchor hit weight (veg everywhere → downweight)
_HIT_WEIGHT = {
    "structure": 1.35,
    "water": 1.25,
    "sky": 0.55,
    "veg": 0.35,
    "other": 0.40,
}

# Adjacency edge weight by unordered label pair (layout > fine texture)
_ADJ_PAIR_WEIGHT = {
    frozenset({"structure", "veg"}): 1.6,
    frozenset({"structure", "sky"}): 1.0,
    frozenset({"structure", "water"}): 1.4,
    frozenset({"structure", "other"}): 0.9,
    frozenset({"water", "veg"}): 1.5,
    frozenset({"water", "sky"}): 0.7,
    frozenset({"water", "other"}): 1.1,
    frozenset({"sky", "veg"}): 0.55,
    frozenset({"veg", "other"}): 0.35,
    frozenset({"veg", "veg"}): 0.12,
    frozenset({"sky", "sky"}): 0.15,
    frozenset({"structure", "structure"}): 0.8,
}


def _rel_dir(a: BgBlockFeat, b: BgBlockFeat) -> str | None:
    """Where b sits relative to a: above|below|left|right."""
    dx = b.cx - a.cx
    dy = b.cy - a.cy
    if abs(dx) < 0.06 and abs(dy) < 0.06:
        return None
    if abs(dy) >= abs(dx) * 0.85:
        return "below" if dy > 0 else "above"
    return "right" if dx > 0 else "left"


def _adjacency_edges(
    anchors: list[BgBlockFeat], *, max_dist: float = 0.58
) -> list[dict]:
    """Directed layout edges between nearby anchors."""
    edges: list[dict] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
            if dist < 0.06 or dist > max_dist:
                continue
            d_ab = _rel_dir(a, b)
            if d_ab is None:
                continue
            w = _ADJ_PAIR_WEIGHT.get(frozenset({a.label, b.label}), 0.2)
            edges.append(
                {
                    "a_idx": a.idx,
                    "b_idx": b.idx,
                    "a_label": a.label,
                    "b_label": b.label,
                    "dir": d_ab,  # b relative to a
                    "weight": w,
                    "dist": dist,
                }
            )
    return edges


def _labels_compat(la: str, lb: str) -> float:
    return _LABEL_COMPAT.get((la, lb), 0.0)


def _match_adjacency(
    q_edges: list[dict], r_edges: list[dict]
) -> tuple[float, list[dict]]:
    """Match query layout edges to gallery; returns (score, matched list)."""
    used: set[int] = set()
    matched: list[dict] = []
    score = 0.0
    # Prefer heavy landmark edges first
    q_sorted = sorted(q_edges, key=lambda e: e["weight"], reverse=True)
    for qe in q_sorted:
        if qe["weight"] < 0.25:
            continue  # ignore weak veg–veg clutter
        best_i = -1
        best_s = 0.0
        for i, re in enumerate(r_edges):
            if i in used:
                continue
            if re["dir"] != qe["dir"]:
                continue
            ca = _labels_compat(qe["a_label"], re["a_label"])
            cb = _labels_compat(qe["b_label"], re["b_label"])
            if ca < 0.35 or cb < 0.35:
                continue
            # soft: similar separation
            dd = abs(qe["dist"] - re["dist"])
            if dd > 0.35:
                continue
            s = qe["weight"] * 0.5 * (ca + cb) * (1.0 - dd / 0.35)
            if s > best_s:
                best_s = s
                best_i = i
        if best_i < 0 or best_s < 0.22:
            continue
        used.add(best_i)
        re = r_edges[best_i]
        def _edge_tag(e: dict) -> str:
            # "b is {dir} of a" — use T≠S codes
            ca = _LABEL_CODE.get(e["a_label"], "O")
            cb = _LABEL_CODE.get(e["b_label"], "O")
            return f"#{e['b_idx']}{cb} {e['dir']} #{e['a_idx']}{ca}"

        matched.append(
            {
                "q": _edge_tag(qe),
                "r": _edge_tag(re),
                "w": round(best_s, 3),
            }
        )
        score += best_s
    return score, matched


def _prune_redundant_veg_hits(pairs: list[dict]) -> list[dict]:
    """At most one veg hit unless T/W already in the hit set."""
    if len(pairs) <= 1:
        return pairs
    has_landmark = any(p["q_label"] in ("structure", "water") for p in pairs)
    if has_landmark:
        return pairs
    non_veg = [p for p in pairs if p["q_label"] != "veg"]
    veg = sorted(
        [p for p in pairs if p["q_label"] == "veg"],
        key=lambda p: p["aff"],
        reverse=True,
    )
    return non_veg + veg[:1]


def _hit_layout_consistency(
    pairs: list[dict],
    q_by_idx: dict[int, BgBlockFeat],
    r_by_idx: dict[int, BgBlockFeat],
) -> tuple[float, int, int]:
    """Among hit pairs, fraction whose relative direction agrees (Q vs R)."""
    if len(pairs) < 2:
        return 1.0, 0, 0
    ok = 0
    tot = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            qa = q_by_idx.get(int(pairs[i]["q"]))
            qb = q_by_idx.get(int(pairs[j]["q"]))
            ra = r_by_idx.get(int(pairs[i]["r"]))
            rb = r_by_idx.get(int(pairs[j]["r"]))
            if qa is None or qb is None or ra is None or rb is None:
                continue
            dq = _rel_dir(qa, qb)
            dr = _rel_dir(ra, rb)
            if dq is None or dr is None:
                continue
            qd = float(np.hypot(qa.cx - qb.cx, qa.cy - qb.cy))
            rd = float(np.hypot(ra.cx - rb.cx, ra.cy - rb.cy))
            tot += 1
            # direction must match; separation within soft band
            if dq == dr and abs(qd - rd) <= 0.30:
                ok += 1
    if tot == 0:
        return 1.0, 0, 0
    return float(ok) / float(tot), ok, tot


def score_block_vote(
    q: BgBlockSig, r: BgBlockSig, *, min_hits: int = 2
) -> dict:
    """Multi-anchor vote + adjacency + hit-layout consistency (v2.2)."""
    if not q.anchors or not r.anchors:
        return {
            "score": 0.0,
            "hits": 0,
            "pairs": [],
            "adj_score": 0.0,
            "adj_pairs": [],
            "layout_ok": 1.0,
        }
    q_by_idx = {a.idx: a for a in q.anchors}
    r_by_idx = {a.idx: a for a in r.anchors}
    used_r: set[int] = set()
    pairs: list[dict] = []
    for qa in q.anchors:
        best_j = -1
        best_a = 0.0
        for j, ra in enumerate(r.anchors):
            if j in used_r:
                continue
            a = _pair_affinity(qa, ra)
            if a > best_a:
                best_a = a
                best_j = j
        if best_j < 0 or best_a < 0.28:
            continue
        used_r.add(best_j)
        ra = r.anchors[best_j]
        w = _HIT_WEIGHT.get(qa.label, 0.4)
        pairs.append(
            {
                "q": qa.idx,
                "q_label": qa.label,
                "r": ra.idx,
                "r_label": ra.label,
                "aff": round(best_a, 3),
                "w": round(w, 2),
            }
        )

    # Drop stacked open-field V votes before scoring / layout check
    pairs = _prune_redundant_veg_hits(pairs)

    # If multi-hit layout disagrees, keep largest coherent subset (greedy by aff)
    layout_ok, layout_agree, layout_tot = _hit_layout_consistency(
        pairs, q_by_idx, r_by_idx
    )
    if layout_tot >= 1 and layout_ok < 0.5:
        ordered = sorted(pairs, key=lambda p: p["aff"], reverse=True)
        kept: list[dict] = []
        for cand in ordered:
            trial = kept + [cand]
            ok, _, tot = _hit_layout_consistency(trial, q_by_idx, r_by_idx)
            if tot == 0 or ok >= 0.99:
                kept = trial
        pairs = kept
        layout_ok, layout_agree, layout_tot = _hit_layout_consistency(
            pairs, q_by_idx, r_by_idx
        )

    hit_score = 0.0
    for p in pairs:
        hit_score += float(p["w"]) * (0.55 + 0.45 * float(p["aff"]))

    hits = len(pairs)
    labs = {p["q_label"] for p in pairs}
    # Adjacency only on HIT anchors (no free float from unmatched T edges)
    q_hit = [q_by_idx[int(p["q"])] for p in pairs if int(p["q"]) in q_by_idx]
    r_hit = [r_by_idx[int(p["r"])] for p in pairs if int(p["r"]) in r_by_idx]
    q_edges_hit = _adjacency_edges(q_hit)
    r_edges_hit = _adjacency_edges(r_hit)
    adj_score, adj_pairs = _match_adjacency(q_edges_hit, r_edges_hit)

    # Full-graph landmark edges: used only for "query had T/W layout" penalty
    q_edges_all = _adjacency_edges(q.anchors)
    q_landmark = sum(1 for e in q_edges_all if e["weight"] >= 1.0)

    score = hit_score + 1.35 * adj_score
    if hits < min_hits and adj_score < 0.5:
        score *= 0.35
    if len(labs) >= 2:
        score += 0.35
    # query has landmark adjacency but hits carried none → bury pure-veg wins
    if q_landmark >= 1 and adj_score < 0.35:
        score *= 0.50
    elif q_landmark >= 2 and adj_score < 0.8:
        score *= 0.70
    # pure V/sky hits: no T/W in pair set → hard cap (layout prior needs landmarks)
    if hits >= 1 and not (labs & {"structure", "water"}):
        score *= 0.45
    # residual layout disagreement (shouldn't happen after prune) → bury
    if layout_tot >= 1 and layout_ok < 0.67:
        score *= 0.25

    return {
        "score": float(score),
        "hits": hits,
        "pairs": pairs,
        "adj_score": round(adj_score, 3),
        "adj_pairs": adj_pairs,
        "q_landmark_edges": q_landmark,
        "layout_ok": round(layout_ok, 3),
        "layout_agree": layout_agree,
        "layout_tot": layout_tot,
    }


def search_bg_blocks_gallery(
    *,
    data_root: Path,
    query_name: str,
    query_year: str = "2018",
    gallery_year: str = "2017",
    top_k: int = 8,
    limit: int = 0,
    use_cache: bool = True,
    cache_root: Path | None = None,
    preview_dir: Path | None = None,
    gold_name: str = "DJI_0752.JPG",
    log_print: bool = False,
) -> dict:
    """Sunny query → gallery by color+edge block voting (v2)."""
    data_root = Path(data_root)
    q_path = data_root / query_year / Path(query_name).name
    if not q_path.is_file():
        raise FileNotFoundError(f"query missing: {q_path}")

    refs, filt = list_usable_near_images(
        data_root / gallery_year, limit=5000, year=gallery_year
    )
    if limit > 0:
        refs = refs[:limit]

    t0 = time.time()
    q_bgr = cv2.imread(str(q_path))
    if q_bgr is None:
        raise RuntimeError("query unreadable")
    q_res = extract_bg_blocks(q_path.name, q_bgr, year=query_year)
    q_sig = blocks_result_to_sig(q_res)
    if q_sig is None:
        raise RuntimeError("query bg blocks empty")

    if log_print:
        print(
            f"query {query_year}/{q_path.name} → {gallery_year} usable×{len(refs)} "
            f"(pool≈{filt.get('usable')}) anchors={len(q_sig.anchors)} "
            f"method={CACHE_TAG_V2}"
        )

    ranked: list[dict] = []
    hit = miss = skip = mem_hit = 0
    for i, path in enumerate(refs, 1):
        r_sig, src = resolve_bg_block_sig(
            path, year=gallery_year, use_cache=use_cache, cache_root=cache_root
        )
        if r_sig is None:
            skip += 1
            continue
        if src == "mem":
            mem_hit += 1
            hit += 1
        elif src == "disk":
            hit += 1
        elif src == "built":
            miss += 1
        else:
            skip += 1
            continue

        vote = score_block_vote(q_sig, r_sig)
        ranked.append(
            {
                "name": r_sig.name,
                "score": vote["score"],
                "hits": vote["hits"],
                "pairs": vote["pairs"],
                "adj_score": vote.get("adj_score", 0.0),
                "adj_pairs": vote.get("adj_pairs", []),
                "layout_ok": vote.get("layout_ok", 1.0),
                "bg_frac": r_sig.bg_frac,
            }
        )
        if log_print and (i % 10 == 0 or i == len(refs)):
            print(
                f"  {i}/{len(refs)} hit={hit} mem={mem_hit} miss={miss} skip={skip}"
            )

    ranked.sort(key=lambda r: (r["score"], r["hits"]), reverse=True)
    top = ranked[:top_k]
    gold_rank = next((i for i, r in enumerate(ranked, 1) if r["name"] == gold_name), None)

    preview_paths: dict[str, str] = {}
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        q_dbg = draw_bg_blocks_debug(q_res)
        q_out = preview_dir / f"bg_blocks_query_{Path(query_name).stem}.jpg"
        cv2.imwrite(str(q_out), q_dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        preview_paths["query_blocks"] = q_out.name
        if top:
            rb = cv2.imread(str(data_root / gallery_year / top[0]["name"]))
            if rb is not None:
                r_res = extract_bg_blocks(top[0]["name"], rb, year=gallery_year)
                r_sig = blocks_result_to_sig(r_res)
                vote0 = (
                    score_block_vote(q_sig, r_sig)
                    if r_sig is not None
                    else {
                        "score": top[0]["score"],
                        "hits": top[0]["hits"],
                        "pairs": top[0].get("pairs") or [],
                        "adj_score": top[0].get("adj_score", 0),
                    }
                )
                panel = compose_bg_match_texture_debug(q_res, r_res, vote0)
                p1 = preview_dir / "bg_blocks_search_top1.jpg"
                cv2.imwrite(str(p1), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                preview_paths["top1"] = p1.name
                preview_paths["match_textures"] = p1.name

    return {
        "query": f"{query_year}/{Path(query_name).name}",
        "query_year": query_year,
        "query_id": Path(query_name).name,
        "gallery_year": gallery_year,
        "gallery_n": len(refs),
        "gallery_pool": "usable",
        "cache_hit": hit,
        "cache_mem": mem_hit,
        "cache_miss": miss,
        "skip": skip,
        "gold_name": gold_name,
        "gold_rank": gold_rank,
        "secs": round(time.time() - t0, 1),
        "query_bg_frac": q_sig.bg_frac,
        "query_anchors": [
            {
                "idx": a.idx,
                "label": a.label,
                "cx": round(a.cx, 3),
                "cy": round(a.cy, 3),
            }
            for a in q_sig.anchors
        ],
        "direction_note": (
            "sunny query → gallery: super-anchor vote + hit-layout "
            "(merge same-label; T/W required for high score)"
        ),
        "method": "bg_sig_v2.2",
        "preview": preview_paths,
        "top": [
            {
                "rank": i,
                "name": r["name"],
                "ref_id": r["name"],
                "query_id": Path(query_name).name,
                "score": round(r["score"], 3),
                "hits": r["hits"],
                "pairs": r["pairs"],
                "adj_score": r.get("adj_score", 0.0),
                "adj_pairs": r.get("adj_pairs", []),
                "layout_ok": r.get("layout_ok", 1.0),
                "bg_frac": r["bg_frac"],
            }
            for i, r in enumerate(top, 1)
        ],
    }


# --- v3: background land-boundary layout lines (weak station prior) ---------
#
# Color-block voting approximates "what's there"; this compares the *skeleton*
# of field / hedge / road / shore boundaries — more stable across years than hue.

CACHE_TAG_LAYOUT = "bg_layout_v1d"  # bump when extract thresholds change
_DEFAULT_CACHE_ROOT_LAYOUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "gallery_features"
    / CACHE_TAG_LAYOUT
)
_LAYOUT_MAX_SIDE = 640
_LAYOUT_INK_W = 160
_LAYOUT_ORI_BINS = 8
_BG_LAYOUT_MEM: dict[str, "BgLayoutSig"] = {}


@dataclass
class BgLayoutResult:
    image_id: str
    small_bgr: np.ndarray
    bg: np.ndarray
    ink: np.ndarray  # uint8 0/255 on small size (long edges only)
    segments: list[tuple[int, int, int, int]]  # x1,y1,x2,y2
    bg_frac: float
    mask_detail: str


@dataclass
class BgLayoutSig:
    name: str
    bg_frac: float
    ink: np.ndarray  # uint8 HxW small fixed width
    ori_hist: np.ndarray  # float32 L2
    band: np.ndarray  # float32 [upper, mid, lower] mass


def extract_bg_layout(
    image_id: str,
    bgr: np.ndarray,
    *,
    year: str | None = None,
) -> BgLayoutResult:
    """Long land-boundary lines outside the blade (not grass micro-texture)."""
    mr = estimate_blade_mask(bgr, year=year)
    small, _ = _resize_max_side(bgr, _LAYOUT_MAX_SIDE)
    blade_s = cv2.resize(
        mr.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    # Extra dilate vs color-blocks: keep blade rim / tower out of layout ink
    dil = max(5, int(round((_BLADE_DILATE + 8) * small.shape[1] / max(bgr.shape[1], 1))))
    bg = background_mask_from_blade(blade_s, dilate_px=dil)
    bg_frac = float((bg > 0).mean())
    h, w = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray[bg > 0])) if (bg > 0).any() else float(np.median(gray))

    def _lines_from_edges(soft_k: int, lo: int, hi: int, min_len: int, *, fill_cc: bool):
        soft = cv2.GaussianBlur(gray, (soft_k, soft_k), 0)
        edges = cv2.Canny(soft, lo, hi)
        edges = cv2.bitwise_and(edges, bg)
        segs = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(14, min_len // 3),
            minLineLength=min_len,
            maxLineGap=max(8, min_len // 6),
        )
        ink_l = np.zeros((h, w), dtype=np.uint8)
        segs_l: list[tuple[int, int, int, int]] = []
        if segs is not None:
            for x1, y1, x2, y2 in segs.reshape(-1, 4):
                length = float(np.hypot(int(x2) - int(x1), int(y2) - int(y1)))
                if length < min_len:
                    continue
                ang = abs(
                    float(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
                )
                ang = min(ang, 180.0 - ang)
                if ang > 55.0:
                    continue
                segs_l.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.line(
                    ink_l,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    3 if ang < 25.0 else 2,
                    cv2.LINE_AA,
                )
        if fill_cc and len(segs_l) < 6:
            num, cc, stats, _ = cv2.connectedComponentsWithStats(
                (edges > 0).astype(np.uint8), 8
            )
            for j in range(1, num):
                area = int(stats[j, cv2.CC_STAT_AREA])
                bw = int(stats[j, cv2.CC_STAT_WIDTH])
                bh = int(stats[j, cv2.CC_STAT_HEIGHT])
                if bw < min_len or bw < bh * 1.8 or area < min_len:
                    continue
                ink_l[cc == j] = 255
        return segs_l, ink_l

    # Pass A: sunny grass-suppressed. Pass B (fallback): softer — 2017 overcast
    # can still have high median (bright sky in bg) yet weak land edges.
    min_a = max(32, int(0.11 * min(h, w)))
    lo_a = max(28, int(0.45 * med))
    hi_a = max(lo_a + 45, int(1.15 * med))
    segments, ink = _lines_from_edges(11, lo_a, hi_a, min_a, fill_cc=False)
    if len(segments) < 3 or float((ink > 0).mean()) < 0.004:
        min_b = max(22, int(0.07 * min(h, w)))
        lo_b = max(10, int(0.18 * med))
        hi_b = max(lo_b + 20, int(0.55 * med))
        segments, ink = _lines_from_edges(5, lo_b, hi_b, min_b, fill_cc=True)

    ink = cv2.bitwise_and(ink, bg)
    return BgLayoutResult(
        image_id=image_id,
        small_bgr=small,
        bg=bg,
        ink=ink,
        segments=segments,
        bg_frac=bg_frac,
        mask_detail=mr.detail[:160],
    )


def _layout_ori_hist(segments: list[tuple[int, int, int, int]], h: int, w: int) -> np.ndarray:
    hist = np.zeros(_LAYOUT_ORI_BINS, dtype=np.float32)
    if not segments:
        return hist
    diag = float(np.hypot(h, w)) or 1.0
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
        bin_i = int(ang / (180.0 / _LAYOUT_ORI_BINS)) % _LAYOUT_ORI_BINS
        hist[bin_i] += length / diag
    n = float(np.linalg.norm(hist))
    if n > 1e-6:
        hist /= n
    return hist


def _layout_band(ink: np.ndarray) -> np.ndarray:
    h = ink.shape[0]
    on = ink > 0
    tot = float(on.sum()) or 1.0
    a, b = h // 3, 2 * h // 3
    return np.array(
        [on[:a].sum() / tot, on[a:b].sum() / tot, on[b:].sum() / tot],
        dtype=np.float32,
    )


def layout_result_to_sig(res: BgLayoutResult) -> BgLayoutSig:
    h, w = res.ink.shape[:2]
    ink_s = cv2.resize(
        res.ink, (_LAYOUT_INK_W, max(48, int(round(_LAYOUT_INK_W * h / max(w, 1))))),
        interpolation=cv2.INTER_AREA,
    )
    ink_s = (ink_s > 40).astype(np.uint8) * 255
    return BgLayoutSig(
        name=res.image_id,
        bg_frac=res.bg_frac,
        ink=ink_s,
        ori_hist=_layout_ori_hist(res.segments, h, w),
        band=_layout_band(res.ink),
    )


def draw_bg_layout_debug(res: BgLayoutResult) -> np.ndarray:
    """Cyan layout lines on dimmed blade; title for eyeball."""
    out = res.small_bgr.copy()
    out[res.bg == 0] = (out[res.bg == 0] * 0.18).astype(np.uint8)
    tint = out.copy()
    tint[res.ink > 0] = (40, 220, 255)  # BGR cyan-yellow
    on = res.ink > 0
    out[on] = cv2.addWeighted(out, 0.35, tint, 0.65, 0)[on]
    for x1, y1, x2, y2 in res.segments:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{res.image_id}  segs={len(res.segments)} ink={(res.ink > 0).mean():.3f} bg={res.bg_frac:.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "layout lines = land boundaries (field/hedge/road/shore), not grass texture",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def compose_bg_layout_pair_debug(
    q: BgLayoutResult, r: BgLayoutResult, *, score: float
) -> np.ndarray:
    left = draw_bg_layout_debug(q)
    right = draw_bg_layout_debug(r)
    h = max(left.shape[0], right.shape[0])

    def _pad(im: np.ndarray) -> np.ndarray:
        if im.shape[0] == h:
            return im
        canvas = np.zeros((h, im.shape[1], 3), dtype=np.uint8)
        canvas[: im.shape[0]] = im
        return canvas

    left, right = _pad(left), _pad(right)
    gap = np.zeros((h, 10, 3), dtype=np.uint8)
    top = np.hstack([left, gap, right])
    bar = np.zeros((36, top.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        f"layout score={score:.3f}  (chamfer + ori + band)  Q={q.image_id}  R={r.image_id}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, top])


def score_bg_layout(q: BgLayoutSig, r: BgLayoutSig) -> dict:
    """Higher is more similar layout skeleton (0..~1)."""
    # Align ink shapes; thicken so near-miss lines still score
    qi, ri = q.ink, r.ink
    if qi.shape != ri.shape:
        ri = cv2.resize(ri, (qi.shape[1], qi.shape[0]), interpolation=cv2.INTER_AREA)
        ri = (ri > 40).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    qi_t = cv2.dilate(qi, k)
    ri_t = cv2.dilate(ri, k)

    def _chamfer_sim(a: np.ndarray, b: np.ndarray) -> float:
        if not (a > 0).any() or not (b > 0).any():
            return 0.0
        dt = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2, 3)
        d = float(dt[a > 0].mean())
        diag = float(np.hypot(a.shape[0], a.shape[1])) or 1.0
        # soft falloff — cross-year lines rarely pixel-align
        return float(np.exp(-d / max(8.0, 0.10 * diag)))

    c_qr = _chamfer_sim(qi_t, ri_t)
    c_rq = _chamfer_sim(ri_t, qi_t)
    chamfer = 0.5 * (c_qr + c_rq)

    ori = float(np.dot(q.ori_hist, r.ori_hist))
    band = float(max(0.0, 1.0 - 0.5 * np.abs(q.band - r.band).sum()))

    q_mass = float((qi > 0).mean())
    r_mass = float((ri > 0).mean())
    if q_mass < 0.0015 or r_mass < 0.0015:
        score = 0.12 * ori + 0.08 * band
    else:
        score = 0.50 * chamfer + 0.30 * ori + 0.20 * band

    return {
        "score": float(score),
        "chamfer": round(chamfer, 3),
        "ori": round(ori, 3),
        "band": round(band, 3),
        "q_ink_frac": round(q_mass, 4),
        "r_ink_frac": round(r_mass, 4),
    }


def cache_path_layout(year: str, name: str, cache_root: Path | None = None) -> Path:
    root = cache_root or _DEFAULT_CACHE_ROOT_LAYOUT
    return root / year / f"{Path(name).stem}.npz"


def save_bg_layout_sig(path: Path, sig: BgLayoutSig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tag=np.array(CACHE_TAG_LAYOUT),
        usable_v=np.int32(FILTER_VERSION),
        name=np.array(sig.name),
        bg_frac=np.float32(sig.bg_frac),
        ink=sig.ink.astype(np.uint8),
        ori_hist=sig.ori_hist.astype(np.float32),
        band=sig.band.astype(np.float32),
    )


def load_bg_layout_sig(path: Path) -> BgLayoutSig | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if str(data["tag"]) != CACHE_TAG_LAYOUT or int(data["usable_v"]) != FILTER_VERSION:
            return None
        return BgLayoutSig(
            name=str(data["name"]),
            bg_frac=float(data["bg_frac"]),
            ink=np.asarray(data["ink"], dtype=np.uint8),
            ori_hist=np.asarray(data["ori_hist"], dtype=np.float32),
            band=np.asarray(data["band"], dtype=np.float32),
        )
    except (KeyError, ValueError, TypeError):
        return None
    finally:
        data.close()


def _layout_mem_key(year: str, name: str) -> str:
    return f"{CACHE_TAG_LAYOUT}/{year}/{Path(name).name}"


def build_bg_layout_sig_from_path(path: Path, *, year: str) -> BgLayoutSig | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return layout_result_to_sig(extract_bg_layout(path.name, bgr, year=year))


def resolve_bg_layout_sig(
    path: Path,
    *,
    year: str,
    use_cache: bool = True,
    cache_root: Path | None = None,
) -> tuple[BgLayoutSig | None, str]:
    key = _layout_mem_key(year, path.name)
    if key in _BG_LAYOUT_MEM:
        return _BG_LAYOUT_MEM[key], "mem"
    if use_cache:
        cached = load_bg_layout_sig(cache_path_layout(year, path.name, cache_root))
        if cached is not None:
            _BG_LAYOUT_MEM[key] = cached
            return cached, "disk"
    sig = build_bg_layout_sig_from_path(path, year=year)
    if sig is None:
        return None, "skip"
    if use_cache:
        save_bg_layout_sig(cache_path_layout(year, path.name, cache_root), sig)
    _BG_LAYOUT_MEM[key] = sig
    return sig, "built"


def search_bg_layout_gallery(
    *,
    data_root: Path,
    query_name: str,
    query_year: str = "2018",
    gallery_year: str = "2017",
    top_k: int = 8,
    limit: int = 0,
    use_cache: bool = True,
    cache_root: Path | None = None,
    preview_dir: Path | None = None,
    gold_name: str = "DJI_0752.JPG",
    log_print: bool = False,
) -> dict:
    """Sunny query → gallery by land-boundary layout lines."""
    data_root = Path(data_root)
    q_path = data_root / query_year / Path(query_name).name
    if not q_path.is_file():
        raise FileNotFoundError(f"query missing: {q_path}")

    refs, filt = list_usable_near_images(
        data_root / gallery_year, limit=5000, year=gallery_year
    )
    if limit > 0:
        refs = refs[:limit]

    t0 = time.time()
    q_bgr = cv2.imread(str(q_path))
    if q_bgr is None:
        raise RuntimeError("query unreadable")
    q_res = extract_bg_layout(q_path.name, q_bgr, year=query_year)
    q_sig = layout_result_to_sig(q_res)
    if log_print:
        print(
            f"query {query_year}/{q_path.name} → {gallery_year} usable×{len(refs)} "
            f"(pool≈{filt.get('usable')}) segs={len(q_res.segments)} "
            f"method={CACHE_TAG_LAYOUT}"
        )

    ranked: list[dict] = []
    hit = miss = mem_hit = skip = 0
    for i, rpath in enumerate(refs, 1):
        r_sig, src = resolve_bg_layout_sig(
            rpath, year=gallery_year, use_cache=use_cache, cache_root=cache_root
        )
        if src == "mem":
            mem_hit += 1
        elif src == "disk":
            hit += 1
        elif src == "built":
            miss += 1
        else:
            skip += 1
            continue
        vote = score_bg_layout(q_sig, r_sig)
        ranked.append(
            {
                "name": r_sig.name,
                "score": vote["score"],
                "chamfer": vote["chamfer"],
                "ori": vote["ori"],
                "band": vote["band"],
                "bg_frac": r_sig.bg_frac,
            }
        )
        if log_print and (i % 10 == 0 or i == len(refs)):
            print(
                f"  {i}/{len(refs)} hit={hit} mem={mem_hit} miss={miss} skip={skip}"
            )

    ranked.sort(key=lambda r: r["score"], reverse=True)
    top = ranked[:top_k]
    gold_rank = next((i for i, r in enumerate(ranked, 1) if r["name"] == gold_name), None)

    preview_paths: dict[str, str] = {}
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        q_dbg = draw_bg_layout_debug(q_res)
        q_out = preview_dir / f"bg_layout_query_{Path(query_name).stem}.jpg"
        cv2.imwrite(str(q_out), q_dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        preview_paths["query_layout"] = q_out.name
        # gold + top1 pair panels when available
        for tag, name in (("gold", gold_name), ("top1", top[0]["name"] if top else "")):
            if not name:
                continue
            rb = cv2.imread(str(data_root / gallery_year / name))
            if rb is None:
                continue
            r_res = extract_bg_layout(name, rb, year=gallery_year)
            r_sig = layout_result_to_sig(r_res)
            sc = score_bg_layout(q_sig, r_sig)["score"]
            panel = compose_bg_layout_pair_debug(q_res, r_res, score=sc)
            pout = preview_dir / f"bg_layout_{Path(query_name).stem}_vs_{Path(name).stem}.jpg"
            cv2.imwrite(str(pout), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            preview_paths[tag] = pout.name

    return {
        "query": f"{query_year}/{Path(query_name).name}",
        "query_year": query_year,
        "query_id": Path(query_name).name,
        "gallery_year": gallery_year,
        "gallery_n": len(refs),
        "gallery_pool": "usable",
        "cache_hit": hit,
        "cache_mem": mem_hit,
        "cache_miss": miss,
        "skip": skip,
        "gold_name": gold_name,
        "gold_rank": gold_rank,
        "secs": round(time.time() - t0, 1),
        "query_bg_frac": q_sig.bg_frac,
        "query_segments": len(q_res.segments),
        "direction_note": (
            "sunny query → gallery: land-boundary layout lines "
            "(chamfer + orientation + vertical band)"
        ),
        "method": CACHE_TAG_LAYOUT,
        "preview": preview_paths,
        "top": [
            {
                "rank": i,
                "name": r["name"],
                "ref_id": r["name"],
                "query_id": Path(query_name).name,
                "score": round(r["score"], 3),
                "chamfer": r["chamfer"],
                "ori": r["ori"],
                "band": r["band"],
                "bg_frac": r["bg_frac"],
            }
            for i, r in enumerate(top, 1)
        ],
    }
