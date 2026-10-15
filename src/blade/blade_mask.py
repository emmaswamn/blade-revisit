"""1.1 Turbine mask — texture + edges + hub (no white-threshold primary).

Spec: docs/pipeline-stages.md §1.1  method=edge_hub

Pipeline:
  1) Low local-texture regions (blade & sky are smooth; fields are not)
  2) Reject high-saturation vegetation/soil (color as REJECT, not「取白」)
  3) Strong long edges help peel sky off blade when they merge
  4) Morph-close fills interior holes
  5) Hub = distance-transform peak; keep hub-connected star-like mass;
     drop distant building-like blobs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MaskResult:
    mask: np.ndarray
    work_size: tuple[int, int]
    area_px: int
    coverage: float
    num_components_kept: int
    detail: str
    hub_xy: tuple[int, int] | None = None
    usable: bool = True
    reject_reason: str = "ok"


@dataclass
class WhiteRefineDebug:
    """Intermediates for the 1.1b step (sunny white-grow / overcast interior fill)."""

    work_bgr: np.ndarray
    seed: np.ndarray
    core: np.ndarray
    domain: np.ndarray
    grown: np.ndarray
    added: np.ndarray
    hub_xy: tuple[int, int] | None
    white_frac: float
    max_dist: float
    note: str
    scale: float  # work → full (`full = work * scale` for coords)
    lighting: str = "sunny"
    seed_detail: str = ""
    rejected: np.ndarray | None = None  # overcast: holes refused by fill gate


@dataclass
class Geo2Debug:
    """Pass-2 geometry: seed vs white-grow delta trim (inspect lobes)."""

    work_bgr: np.ndarray
    seed: np.ndarray
    grown: np.ndarray
    cleaned: np.ndarray
    kept_added: np.ndarray
    cut_added: np.ndarray
    hub_xy: tuple[int, int] | None
    arm_samples: list[tuple[int, int]]
    tip_end: tuple[int, int] | None
    tip_dir: tuple[float, float] | None
    allow_pts: list[tuple[int, int]]  # envelope dots for helper lines
    mode: str
    note: str
    scale: float


@dataclass
class ExpandLinesDebug:
    """1.3 steps 3a–3d: principal axis + outer contour line pick (draw only)."""

    work_bgr: np.ndarray
    mask: np.ndarray
    axis_origin: tuple[float, float] | None
    axis_dir: tuple[float, float] | None
    axis_half_len: float
    # segments as ((x1,y1),(x2,y2), score)
    all_segs: list[tuple[tuple[int, int], tuple[int, int], float]]
    ring_segs: list[tuple[tuple[int, int], tuple[int, int], float]]
    left_seg: tuple[tuple[int, int], tuple[int, int], float] | None
    right_seg: tuple[tuple[int, int], tuple[int, int], float] | None
    phantom_seg: tuple[tuple[int, int], tuple[int, int]] | None
    paired: bool
    note: str
    scale: float
    lighting: str = "sunny"
    ring_inner: int = 6
    ring_outer: int = 40


@dataclass
class ExpandCloseDebug:
    """1.3 close+fill: yellow hull (legacy) or trap-edge polygon (1.3c)."""

    work_bgr: np.ndarray
    base_mask: np.ndarray
    filled: np.ndarray
    added: np.ndarray
    hull_pts: list[tuple[int, int]]
    extended_segs: list[tuple[tuple[int, int], tuple[int, int]]]
    note: str
    scale: float
    lighting: str = "sunny"
    n_yellow: int = 0
    add_px: int = 0
    mode: str = "yellow_hull"  # yellow_hull | trap_poly
    close_poly: list[tuple[int, int]] | None = None
    gate: str = ""


@dataclass
class ExpandTrapDebug:
    """1.3c: trend trapezoid ROI probe — draw new edge lines only (no fill)."""

    work_bgr: np.ndarray
    mask: np.ndarray
    axis_origin: tuple[float, float] | None
    axis_dir: tuple[float, float] | None
    axis_half_len: float
    axis_half_w: float
    # context from expand_lines (unchanged yellow/orange)
    yellow_segs: list[tuple[tuple[int, int], tuple[int, int], float]]
    orange_segs: list[tuple[tuple[int, int], tuple[int, int], float]]
    trap_polys: list[np.ndarray]  # Nx2 float polys
    roi_mask: np.ndarray
    edges: np.ndarray
    accepted: list[list[tuple[int, int]]]  # edge chains kept
    rejected: list[list[tuple[int, int]]]  # edge chains rejected (sky/veg/water)
    reject_pts: list[tuple[int, int]]  # sample marks
    note: str
    scale: float
    lighting: str = "sunny"
    n_traps: int = 0
    n_accept: int = 0
    n_reject: int = 0
    # fitted rim extended until a closing edge (photo border or the other side)
    extend_segs: list[tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=list
    )


@dataclass
class TipEdgeDebug:
    """Tip completion via ROI crop + edge following (inspect before trusting)."""

    work_bgr: np.ndarray
    base_mask: np.ndarray
    roi_mask: np.ndarray
    edges: np.ndarray
    left_poly: list[tuple[int, int]]
    right_poly: list[tuple[int, int]]
    tip_point: tuple[int, int] | None
    tip_fill: np.ndarray
    hub_xy: tuple[int, int] | None
    tip_end: tuple[int, int] | None
    tip_dir: tuple[float, float] | None
    needed: bool
    need_note: str
    note: str
    scale: float


@dataclass
class TipNeedDebug:
    """Gate: whether tip completion should run at all."""

    work_bgr: np.ndarray
    base_mask: np.ndarray
    hub_xy: tuple[int, int] | None
    tip_end: tuple[int, int] | None
    tip_dir: tuple[float, float] | None
    needed: bool
    reason: str
    tip_w: float
    mid_w: float
    elong: float
    paint_gap: int
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


def _vegetation_mask(bgr: np.ndarray) -> np.ndarray:
    """True where high-saturation plant/soil — reject from turbine candidates."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv)
    # greens + yellow-browns typical of fields
    green = (h >= 30) & (h <= 95) & (s >= 35)
    brown = ((h <= 30) | (h >= 170)) & (s >= 40)
    return green | brown


def _peel_sky_with_edges(cand: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """If smooth sky merges with blade, cut using strong nearly-horizontal edges."""
    h, w = cand.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    # horizontal-ish energy per row
    row_e = edges.mean(axis=1)
    # find strong edge rows in upper 70% that sit inside candidate mass
    cand_rows = cand.max(axis=1) > 0
    if not cand_rows.any():
        return cand

    # For each CC touching the top, try cutting at strongest edge below top band
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cand > 0).astype(np.uint8), 8
    )
    out = np.zeros_like(cand)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.003 * h * w:
            continue
        touch_top = int((labels[0 : max(1, h // 40), :] == i).sum())
        comp = labels == i
        if touch_top < 0.08 * w:
            out[comp] = 255
            continue

        y0 = int(stats[i, cv2.CC_STAT_TOP])
        y1 = y0 + int(stats[i, cv2.CC_STAT_HEIGHT])
        band = row_e[y0 : min(y1, int(h * 0.75))]
        if band.size < 5:
            out[comp] = 255
            continue
        # strongest edge row in this component's upper part
        cut_local = int(np.argmax(band))
        cut = y0 + cut_local
        # require the edge to be somewhat real
        if band[cut_local] < 2.0:
            # fallback: cut where candidate row width shrinks after a wide top (sky)
            widths = comp.sum(axis=1)
            top_w = float(np.median(widths[y0 : y0 + max(3, (y1 - y0) // 8)]))
            cut = y0
            for y in range(y0, min(y1, int(h * 0.7))):
                if widths[y] < 0.55 * top_w and top_w > 0.25 * w:
                    cut = y
                    break
        trial = comp.copy()
        trial[:cut, :] = False
        if trial.sum() < 0.004 * h * w:
            # cut ate everything — keep original lower 60% of bbox
            cut = y0 + int(0.35 * (y1 - y0))
            trial = comp.copy()
            trial[:cut, :] = False
        out[trial] = 255
    return out


def _smooth_candidates(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    std = _local_std(gray, 15)
    # fields noisy; blade/sky smooth. percentile gate adapts per image
    thr = float(np.percentile(std, 35))
    thr = max(thr, 4.0)
    smooth = std <= thr

    veg = _vegetation_mask(bgr)
    cand = (smooth & (~veg)).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k5)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k9, iterations=2)

    cand = _peel_sky_with_edges(cand, gray)
    # fill interior holes after peel
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k9, iterations=2)
    return cand


def _feature_density(gray: np.ndarray, *, ksize: int = 41) -> np.ndarray:
    """Corner density — nacelle lettering / seams beat flat sky-water."""
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=1000, qualityLevel=0.01, minDistance=6, blockSize=5
    )
    dens = np.zeros_like(gray, dtype=np.float32)
    if corners is not None:
        for c in corners[:, 0]:
            x, y = int(round(c[0])), int(round(c[1]))
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                dens[y, x] = 1.0
    dens = cv2.GaussianBlur(dens, (ksize | 1, ksize | 1), 0)
    if dens.max() > 0:
        dens /= dens.max()
    return dens


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _lab_robust_mean_std(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(pts, axis=0).astype(np.float32)
    mad = np.median(np.abs(pts.astype(np.float32) - med), axis=0).astype(np.float32)
    std = np.clip(1.4826 * mad + 1e-3, [3.5, 1.5, 1.5], [12.0, 7.0, 7.0]).astype(
        np.float32
    )
    return med, std


def _sample_lab(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    pts = lab[mask]
    if len(pts) < 120:
        return None, None
    return _lab_robust_mean_std(pts)


def _match_lab(
    lab: np.ndarray, mean: np.ndarray, std: np.ndarray, *, k: float
) -> np.ndarray:
    d = (lab.astype(np.float32) - mean) / std
    return np.sqrt((d**2).sum(axis=2)) <= k


@dataclass
class SkyWaterDebug:
    """Overcast: sky/water color subtract → edge barrier → contour foreground."""

    work_bgr: np.ndarray
    sky_cut: np.ndarray  # bool or 0/255
    water_cut: np.ndarray
    seed: np.ndarray
    hub_xy: tuple[int, int] | None
    note: str
    sky_l: float | None
    water_l: float | None
    scale: float
    edge_barrier: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.uint8))


def _overcast_edge_barrier(gray: np.ndarray) -> np.ndarray:
    """Canny edges as basin walls (0713: leaf vs river stay separate plates)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 120)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(canny, k3, iterations=1)


def _sky_water_ban_masks(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lab sky/water reject masks (bool HxW) for overcast peel + interior fill.

    Mid-frame grey is NOT sampled as water (≈ blade paint). Bottom band only.
    Extra: top-down bright-sky flood (blue-sky-rescue style horizon stop) so
    leftover overcast sky above the blade is cut without eating darker paint.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    std = _local_std(gray, 15)
    veg = _vegetation_mask(bgr)
    # dilate veg slightly — tree crowns / field edges as hard ground ban
    k_veg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    veg_hard = cv2.dilate(veg.astype(np.uint8), k_veg, iterations=1) > 0
    dens = _feature_density(gray)
    edge = _sobel_mag(gray)
    L = lab[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    protect = (dens > 0.10) | (edge > 26)

    sky_h = max(10, int(0.14 * h))
    sky_samp = np.zeros((h, w), dtype=bool)
    sky_samp[:sky_h] = (std[:sky_h] < 4.5) & (~veg[:sky_h]) & (S[:sky_h] < 55)
    sky_mean, sky_std = _sample_lab(lab, sky_samp)
    if sky_mean is None:
        sky_samp[:sky_h] = (std[:sky_h] < 8.0) & (~veg[:sky_h])
        sky_mean, sky_std = _sample_lab(lab, sky_samp)

    sky_m = (
        _match_lab(lab, sky_mean, sky_std, k=2.2)
        if sky_mean is not None
        else np.zeros((h, w), dtype=bool)
    )
    sky_cut = sky_m & (~protect) & (std < 9.0) & (S < 60)
    sky_cut[:sky_h] |= sky_m[:sky_h] & (std[:sky_h] < 11) & (~protect[:sky_h])

    # Top-down bright sky: stay close to sky_mean L, stop at veg / strong edge / L drop.
    # Margin must stay tight — 0624 skyL≈203 vs pale blade; 0735 skyL≈230 vs blade≈145.
    if sky_mean is not None:
        sky_l0 = float(sky_mean[0])
        l_floor = max(sky_l0 - 14.0, 0.90 * sky_l0)
        bright_like = (
            (L >= l_floor)
            & (S < 45)
            & (std < 7.5)
            & (~veg_hard)
            & (V >= max(150.0, sky_l0 - 40.0))
        )
        allow = bright_like.astype(np.uint8) * 255
        barrier = _overcast_edge_barrier(gray)
        allow[barrier > 0] = 0
        allow[veg_hard] = 0
        # column top-down; do not march below mid-frame (blade body lives there)
        y_max = int(0.52 * h)
        gap_need = 2
        flooded = np.zeros((h, w), dtype=bool)
        for x in range(w):
            gap = 0
            last = -1
            for y in range(y_max):
                if allow[y, x] > 0 or (y < sky_h and sky_cut[y, x]):
                    last = y
                    gap = 0
                else:
                    gap += 1
                    if last >= 0 and gap >= gap_need:
                        break
            if last >= 0:
                flooded[: last + 1, x] = (bright_like[: last + 1, x] | sky_cut[: last + 1, x])
        top_touch = flooded.copy()
        top_touch[sky_h:, :] = False
        if top_touch.any():
            num, labels = cv2.connectedComponents(flooded.astype(np.uint8), 8)
            keep = np.zeros_like(flooded)
            top_labels = set(
                int(v) for v in np.unique(labels[:sky_h][top_touch[:sky_h]]) if v > 0
            )
            for lab_i in top_labels:
                keep[labels == lab_i] = True
            sky_cut = sky_cut | keep

    bot0 = int(0.82 * h)
    water_samp = np.zeros((h, w), dtype=bool)
    if sky_mean is not None:
        water_samp[bot0:] = (
            (std[bot0:] < 6.0)
            & (~veg[bot0:])
            & (S[bot0:] < 40)
            & (L[bot0:] < float(sky_mean[0]) - 22.0)
            & (V[bot0:] > 55)
            & (np.abs(lab[bot0:, :, 1].astype(np.float32) - float(sky_mean[1])) < 6)
            & (np.abs(lab[bot0:, :, 2].astype(np.float32) - float(sky_mean[2])) < 6)
        )
    water_mean, water_std = _sample_lab(lab, water_samp)
    water_m = (
        _match_lab(lab, water_mean, water_std, k=2.2)
        if water_mean is not None
        else np.zeros((h, w), dtype=bool)
    )
    water_cut = np.zeros((h, w), dtype=bool)
    if water_mean is not None:
        water_cut[bot0:] = (
            water_m[bot0:] & (~protect[bot0:]) & (std[bot0:] < 7.5) & (S[bot0:] < 45)
        )

    # Ground structure ban (houses): high-edge walls near vegetation belt only.
    ground_struct = (
        (np.arange(h)[:, None] > int(0.48 * h))
        & (~veg_hard)
        & (edge > 28)
        & (std > 8.0)
        & (S < 55)
        & (V > 90)
    )
    veg_near = cv2.dilate(veg_hard.astype(np.uint8), k_veg, iterations=2) > 0
    ground_struct = ground_struct & veg_near

    meta = {
        "sky_l": float(sky_mean[0]) if sky_mean is not None else None,
        "water_l": float(water_mean[0]) if water_mean is not None else None,
        "veg": veg_hard,  # use dilated veg as hard ban downstream
        "veg_raw": veg,
        "ground_struct": ground_struct,
        "std": std,
        "S": S,
        "V": V,
        "lab": lab,
        "gray": gray,
        "dens": dens,
        "edge": edge,
    }
    # fold ground structure into water_cut channel for subtract (orange in sky_water view)
    # actually better separate — use in fg formula below via meta
    return sky_cut, water_cut, meta


def _mid_water_comp(
    comp: np.ndarray,
    *,
    h: int,
    edge: np.ndarray,
    std: np.ndarray,
    frame: float,
    L: np.ndarray | None = None,
    water_l: float | None = None,
) -> bool:
    """True if CC looks like a flat mid-frame water/shore band.

    0685: elongated marsh/water blotch beat the real blade on elong score;
    catch by Lab≈water_l + mid/lower hub even when area is small.
    """
    ys, xs = np.where(comp)
    if len(xs) < 30:
        return False
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    area = int(comp.sum())
    aspect = bw / max(bh, 1)
    cy = y0 + bh / 2
    fill = area / max(bw * bh, 1)
    u8 = (comp.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ring = (cv2.dilate(u8, k) - cv2.erode(u8, k)) > 0
    bg = float(edge[ring].mean()) if ring.sum() > 30 else 0.0
    mean_std = float(std[comp].mean())
    if (
        aspect >= 2.4
        and bh < 0.32 * h
        and cy > 0.40 * h
        and fill > 0.38
        and bg < 14.0
        and mean_std < 7.0
    ):
        return True
    if (
        aspect >= 1.8
        and bh < 0.40 * h
        and cy > 0.35 * h
        and fill > 0.30
        and mean_std < 8.5
        and bg < 18.0
        and area > 0.04 * frame
    ):
        return True
    # 0685-class: small/mid water plate with Lab near sampled water (blade left unseeded)
    if (
        L is not None
        and water_l is not None
        and cy > 0.38 * h
        and area < 0.10 * frame
        and mean_std < 11.0
        and abs(float(L[comp].mean()) - float(water_l)) < 22.0
        and aspect >= 1.35
    ):
        return True
    return False


def _sky_shelf_comp(
    comp: np.ndarray,
    *,
    h: int,
    w: int,
    edge: np.ndarray,
    std: np.ndarray,
    L: np.ndarray,
    sky_l: float | None,
    frame: float,
) -> bool:
    """True if CC is a flat horizontal overcast cloud/sky shelf (0578 left band).

    Often welded to a vertical blade by close; hub sits on the left/right border.
    """
    ys, xs = np.where(comp)
    if len(xs) < 40:
        return False
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    area = int(comp.sum())
    aspect = bw / max(bh, 1)
    cy = y0 + bh / 2.0
    fill = area / max(bw * bh, 1)
    mean_std = float(std[comp].mean())
    mean_l = float(L[comp].mean())
    u8 = (comp.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ring = (cv2.dilate(u8, k) - cv2.erode(u8, k)) > 0
    bg = float(edge[ring].mean()) if ring.sum() > 30 else 0.0
    left_touch = float(comp[:, :4].sum()) / float(max(area, 1))
    right_touch = float(comp[:, w - 4 :].sum()) / float(max(area, 1))
    border = left_touch > 0.002 or right_touch > 0.002
    # mid/upper horizontal flat grey, often touching side
    if (
        aspect >= 2.2
        and bh < 0.28 * h
        and 0.18 * h < cy < 0.62 * h
        and fill > 0.28
        and mean_std < 6.5
        and bg < 16.0
        and (border or area > 0.04 * frame)
    ):
        if sky_l is None or mean_l >= float(sky_l) - 55.0:
            return True
    # thinner shelf glued to side (0578 basin i=3 style)
    if (
        aspect >= 2.0
        and bh < 0.22 * h
        and cy < 0.55 * h
        and mean_std < 4.5
        and border
        and mean_l >= 140.0
    ):
        return True
    return False


def _edge_basin_mask(base: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    """Foreground minus edge walls → separate leaf/water plates (no global shred)."""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inv = base.copy()
    inv[barrier > 0] = 0
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k3)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    deep = (dt >= 4.0).astype(np.uint8) * 255
    rec = deep.copy()
    for _ in range(8):
        rec = cv2.bitwise_and(cv2.dilate(rec, k3), inv)
    rec = cv2.morphologyEx(rec, cv2.MORPH_CLOSE, k5, iterations=1)
    return cv2.bitwise_and(rec, inv)


def _sky_water_contour_seed(
    bgr: np.ndarray,
    *,
    ban: tuple[np.ndarray, np.ndarray, dict] | None = None,
) -> tuple[
    np.ndarray,
    tuple[int, int] | None,
    str,
    np.ndarray,
    np.ndarray,
    float | None,
    float | None,
    np.ndarray,
]:
    """Sky/water Lab subtract, then score mild-close CCs ∪ edge-basin CCs.

    Mid-frame water is rejected geometrically (horizontal flat weak-edge CCs).
    Do NOT sample mid-frame grey as water — that eats the blade (same Lab as paint).
    Edge basins fix 0713 leaf/river weld without shredding nacelle shots (0680).

    `ban` optional: (sky_cut, water_cut, ban_meta) — sunny reuses this skeleton
    with its own bans; default is overcast `_sky_water_ban_masks`.
    """
    h, w = bgr.shape[:2]
    if ban is None:
        sky_cut, water_cut, ban_meta = _sky_water_ban_masks(bgr)
    else:
        sky_cut, water_cut, ban_meta = ban
    sky_cut = sky_cut.copy()
    water_cut = water_cut.copy()
    veg = ban_meta["veg"]
    ground = ban_meta.get("ground_struct")
    if ground is None:
        ground = np.zeros_like(veg)
    dens = ban_meta["dens"]
    edge = ban_meta["edge"]
    std = ban_meta["std"]
    V = ban_meta["V"]
    gray = ban_meta["gray"]
    L = ban_meta["lab"][:, :, 0].astype(np.float32)
    sky_l = ban_meta["sky_l"]
    water_l = ban_meta["water_l"]

    fg = (~sky_cut) & (~water_cut) & (~veg) & (~ground) & (V >= 55)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    base = cv2.morphologyEx((fg.astype(np.uint8)) * 255, cv2.MORPH_OPEN, k5)
    barrier = _overcast_edge_barrier(gray)

    # pool A: classic close + contour fill (good for nacelle; may weld leaf/river)
    cand_a = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k11, iterations=2)
    filled_a = np.zeros_like(cand_a)
    cs, _ = cv2.findContours(cand_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = 0.004 * h * w
    for c in cs:
        if cv2.contourArea(c) >= min_a:
            cv2.drawContours(filled_a, [c], -1, 255, -1)

    # pool B: edge basins (0713 leaf vs river plates)
    filled_b = _edge_basin_mask(base, barrier)

    frame = float(h * w)

    def _score_comp(
        tag: str, comp: np.ndarray
    ) -> tuple[float, tuple[int, int], int, float, float] | None:
        area = int((comp > 0).sum())
        if area < 0.006 * frame or area > 0.55 * frame:
            return None
        hub, dt = _hub_from_distance(comp)
        if hub is None:
            return None
        x0, y0 = max(0, hub[0] - 30), max(0, hub[1] - 30)
        x1, y1 = min(w, hub[0] + 31), min(h, hub[1] + 31)
        feat = float(dens[y0:y1, x0:x1].mean())
        star, _ = _arm_taper_score(comp, hub, dt)
        elong = _elongation(comp)
        rect = _rectangularity(comp)
        star_w = star * (1.0 if elong >= 2.6 else 0.35)
        # prefer tall/vertical arms (blade tip-down) over wide sky shelves
        ys, xs = np.where(comp > 0)
        bw = int(xs.max() - xs.min()) + 1
        bh = int(ys.max() - ys.min()) + 1
        aspect = bw / max(bh, 1)
        score = (
            star_w * 1.4
            + min(elong, 12.0) * 1.8
            + np.sqrt(area / frame) * 5.0
            + feat * 80.0
            - rect * 4.0
        )
        if aspect < 1.0 and elong >= 3.0:
            score += 22.0  # vertical blade bonus (0578 tip-down)
        top_frac = float((comp[: max(1, int(0.12 * h))] > 0).sum()) / float(max(area, 1))
        # tip-down blade entering from top of frame (0578 basin blade vs tiny feat scrap)
        if aspect < 0.9 and elong >= 4.0 and top_frac > 0.12:
            score += 40.0
        if aspect >= 2.2 and elong < 3.2:
            score -= 28.0  # wide flat shelf penalty
        if area < 0.02 * frame and feat < 0.15:
            score -= 30.0  # drop tiny scraps that beat real blade on feat alone
        if feat >= 0.08:
            score += 40.0
        if hub[1] < 0.10 * h:
            score -= 20.0
        # tip-from-top hubs are OK for vertical blades — undo part of the top penalty
        if aspect < 0.9 and elong >= 4.0 and hub[1] < 0.10 * h:
            score += 18.0
        # 0578: border hub on sky shelf at cov~0.14 — was only penalized above 0.18
        if (hub[0] <= 6 or hub[0] >= w - 7) and area > 0.08 * frame and feat < 0.03:
            score -= 70.0
        elif (hub[0] <= 6 or hub[0] >= w - 7) and area > 0.18 * frame:
            score -= 55.0
        if elong < 2.3 and area > 0.18 * frame:
            score -= 35.0
        if hub[0] >= w - 7 and feat < 0.02 and elong < 2.5:
            score -= 30.0
        if tag == "basin" and (hub[0] <= 6 or hub[0] >= w - 7) and area < 0.08 * frame:
            score -= 40.0
        if tag == "basin" and elong >= 2.8:
            score += 16.0
        # 0685: water/marsh CC matches water Lab + sits mid/lower — dump hard
        mean_L = float(L[comp > 0].mean()) if (comp > 0).any() else 0.0
        if (
            water_l is not None
            and abs(mean_L - float(water_l)) < 20.0
            and hub[1] > 0.35 * h
            and feat < 0.12
        ):
            score -= 55.0
        return float(score), hub, area, feat, float(elong)

    def _collect(tag: str, filled: np.ndarray) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (filled > 0).astype(np.uint8), 8
        )
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 0.004 * frame:
                continue
            comp = labels == i
            if _mid_water_comp(
                comp,
                h=h,
                edge=edge,
                std=std,
                frame=frame,
                L=L,
                water_l=water_l,
            ):
                water_cut[comp] = True
                continue
            if _sky_shelf_comp(
                comp, h=h, w=w, edge=edge, std=std, L=L, sky_l=sky_l, frame=frame
            ):
                sky_cut[comp] = True
                continue
            out.append((tag, (comp.astype(np.uint8)) * 255))
        return out

    close_cands = _collect("close", filled_a)
    best: tuple[float, np.ndarray, tuple[int, int], int, float, str, float] | None = None
    for tag, comp in close_cands:
        sc = _score_comp(tag, comp)
        if sc is None:
            continue
        score, hub, area, feat, elong = sc
        top_frac = float((comp[: max(1, int(0.12 * h))] > 0).sum()) / float(max(area, 1))
        ys, xs = np.where(comp > 0)
        bw = int(xs.max() - xs.min()) + 1
        bh = int(ys.max() - ys.min()) + 1
        aspect = bw / max(bh, 1)
        if (
            (
                elong < 2.5
                and area > 0.18 * frame
                and (hub[0] <= 6 or hub[0] >= w - 7)
            )
            # fat low-texture sky/leaf weld (0752); keep high-feat nacelle (0680)
            or (elong < 2.2 and area > 0.35 * frame and feat < 0.05)
            # 0624: blade+horizon/sky welded by close; not border-hub, not huge enough
            # for the 0.35 rule — low-feat + top-band leak → drop, let basin compete
            or (
                elong < 2.0
                and area > 0.20 * frame
                and feat < 0.05
                and top_frac > 0.03
            )
            # 0578: close welds vertical blade + left sky shelf; hub on border
            or (
                (hub[0] <= 6 or hub[0] >= w - 7)
                and feat < 0.03
                and area > 0.08 * frame
                and aspect >= 1.8
            )
            # 0685: after water blotch dump, leftover close/basin welds blade↔horizon water
            or (feat < 0.02 and elong < 2.5 and area > 0.12 * frame)
            or score < 0.0
        ):
            continue  # drop welded leaf/river / fat sky blob / sky shelf weld
        if best is None or score > best[0]:
            best = (score, comp, hub, area, feat, tag, elong)

    # Always score basins too (0624: basin elong≈3.6 beats fat close≈1.7).
    # High-feat nacelle close (0680) still wins on score.
    for tag, comp in _collect("basin", filled_b):
        sc = _score_comp(tag, comp)
        if sc is None:
            continue
        score, hub, area, feat, elong = sc
        if score < 0.0 or (feat < 0.02 and elong < 2.5 and area > 0.12 * frame):
            continue
        if best is None or score > best[0]:
            best = (score, comp, hub, area, feat, tag, elong)

    if best is None:
        # last resort: old heavy close path (may weld; fallback still better than empty)
        heavy = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k11, iterations=2)
        filled = np.zeros_like(heavy)
        cs, _ = cv2.findContours(heavy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) >= min_a:
                cv2.drawContours(filled, [c], -1, 255, -1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (filled > 0).astype(np.uint8), 8
        )
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 0.006 * frame or area > 0.72 * frame:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            hub, dt = _hub_from_distance(comp)
            if hub is None:
                continue
            elong = _elongation(comp)
            score = min(elong, 12.0) + np.sqrt(area / frame) * 6.0
            if best is None or score > best[0]:
                best = (float(score), comp, hub, area, 0.0, "heavy", float(elong))

    if best is None:
        return (
            np.zeros((h, w), np.uint8),
            None,
            "sw_empty",
            sky_cut,
            water_cut,
            sky_l,
            water_l,
            barrier,
        )

    seed, hub = best[1], best[2]
    seed = _cut_distant_bridges(seed, hub, k=21)
    # morph-close may re-weld sky/ground — strip bans again
    seed[sky_cut] = 0
    seed[veg] = 0
    seed[ground] = 0
    flood = seed.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    if hub is not None and 0 <= hub[0] < w and 0 <= hub[1] < h and flood[hub[1], hub[0]] > 0:
        cv2.floodFill(flood, ff, hub, 128, flags=4 | (255 << 8))
        seed = np.where(flood == 128, 255, 0).astype(np.uint8)
    else:
        # hub landed on peeled sky/ground — pick thickest remaining
        hub2, _ = _hub_from_distance(seed)
        hub = hub2
        if hub is not None and seed[hub[1], hub[0]] > 0:
            flood = seed.copy()
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ff, hub, 128, flags=4 | (255 << 8))
            seed = np.where(flood == 128, 255, 0).astype(np.uint8)
    seed = _fill_holes(seed)
    # show ground ban in water_cut channel for sky_water debug (orange)
    water_cut = water_cut | ground
    note = (
        f"sw_score={best[0]:.1f} feat={best[4]:.3f} elong={best[6]:.2f} "
        f"area={int((seed > 0).sum())} skyL={sky_l} waterL={water_l} "
        f"hub={hub} src={best[5]} skyCut={int(sky_cut.sum())} gnd={int(ground.sum())}"
    )
    return seed, hub, note, sky_cut, water_cut, sky_l, water_l, barrier


def debug_sky_water_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
) -> SkyWaterDebug:
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    small, scale = _resize_max_side_bgr(bgr, max_side)
    seed, hub, note, sky_cut, water_cut, sky_l, water_l, barrier = _sky_water_contour_seed(
        small
    )
    return SkyWaterDebug(
        work_bgr=small,
        sky_cut=(sky_cut.astype(np.uint8) * 255),
        water_cut=(water_cut.astype(np.uint8) * 255),
        seed=seed,
        hub_xy=hub,
        note=note,
        sky_l=sky_l,
        water_l=water_l,
        scale=scale,
        edge_barrier=barrier,
    )


def overlay_sky_water_debug(dbg: SkyWaterDebug) -> np.ndarray:
    """Red=sky cut · orange=water cut · cyan=contour seed."""
    out = dbg.work_bgr.copy()
    tint = out.copy()
    tint[dbg.sky_cut > 0] = (0, 0, 255)
    tint[dbg.water_cut > 0] = (0, 165, 255)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    if dbg.seed.max() > 0:
        out = overlay_mask_with_hub(out, dbg.seed, dbg.hub_xy)
    lines = [
        "RED=sky  ORANGE=water  CYAN=contour seed  (edge barrier on)",
        dbg.note,
        f"skyL={dbg.sky_l} waterL={dbg.water_l}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def overlay_edge_seed_debug(dbg: SkyWaterDebug) -> np.ndarray:
    """Yellow=edge barrier · cyan=seed · prove cut killed river weld."""
    out = (dbg.work_bgr.astype(np.float32) * 0.55).astype(np.uint8)
    if dbg.seed.max() > 0:
        m = dbg.seed > 0
        out[m] = (out[m] * 0.45 + np.array([220, 220, 40], np.float32) * 0.55).astype(
            np.uint8
        )
    if dbg.edge_barrier.size and dbg.edge_barrier.max() > 0:
        out[dbg.edge_barrier > 0] = (0, 255, 255)
    if dbg.hub_xy is not None:
        cv2.circle(out, dbg.hub_xy, 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
    lines = [
        "YELLOW=edge barrier  CYAN=seed  (basin when close welds)",
        dbg.note,
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Sunny contour seed: same skeleton as overcast (ban → basin/close → feat score)
# Water sampling is sunny-specific (blue sheets, not bottom-band-only).
# ---------------------------------------------------------------------------


@dataclass
class SunnyWaterDebug:
    """Sunny contour seed debug — same channels as SkyWaterDebug + edge barrier."""

    work_bgr: np.ndarray
    sky_cut: np.ndarray  # 0/255
    water_cut: np.ndarray  # 0/255
    seed: np.ndarray
    hub_xy: tuple[int, int] | None
    note: str
    sky_l: float | None
    water_l: float | None
    scale: float
    edge_barrier: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.uint8))


def _sunny_water_sheet_ok(
    stats_row: np.ndarray, *, h: int, w: int, frame: float
) -> bool:
    """Wide/flat water body geometry — reject tall blade/tower blue reflections."""
    ar = int(stats_row[cv2.CC_STAT_AREA])
    bw = int(stats_row[cv2.CC_STAT_WIDTH])
    bh = int(stats_row[cv2.CC_STAT_HEIGHT])
    aspect = bw / max(bh, 1)
    bh_frac = bh / float(max(h, 1))
    if bh_frac > 0.42:
        return False
    # distant thin sea strip (0352)
    if ar >= int(0.0025 * frame) and aspect >= 4.0 and bh_frac <= 0.12:
        return True
    if ar >= int(0.003 * frame) and aspect >= 3.0 and bh_frac <= 0.35:
        return True
    if ar >= int(0.008 * frame) and aspect >= 2.0 and bh_frac <= 0.38:
        return True
    if ar >= int(0.018 * frame) and aspect >= 1.7 and bh_frac <= 0.40:
        return True
    return False


def _sunny_water_sheet_ban(
    bgr: np.ndarray, ban_meta: dict
) -> tuple[np.ndarray, dict]:
    """Sunny blue water sheets → edge-limited flood (do not cross blade Canny).

    Reject seed sheets glued to high-feat/paint protect (blade strips vs open sea).
    """
    h, w = bgr.shape[:2]
    frame = float(h * w)
    lab = ban_meta["lab"]
    std = ban_meta["std"]
    veg = ban_meta["veg"]
    dens = ban_meta["dens"]
    edge = ban_meta["edge"]
    gray = ban_meta["gray"]
    S = ban_meta["S"]
    V = ban_meta["V"]
    bb = lab[:, :, 2].astype(np.float32)
    B, G, R = cv2.split(bgr)
    protect = (dens > 0.10) | (edge > 30) | ((S <= 55) & (V >= 160))
    prot_near = cv2.dilate(protect.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    y0 = int(0.06 * h)
    blue = (B.astype(np.int16) - G.astype(np.int16) >= 6) & (
        B.astype(np.int16) - R.astype(np.int16) >= 3
    )
    samp = np.zeros((h, w), dtype=bool)
    samp[y0:] = (
        (std[y0:] < 6.0)
        & (~veg[y0:])
        & (S[y0:] < 110)
        & (V[y0:] > 45)
        & (V[y0:] < 175)
        & blue[y0:]
        & (bb[y0:] < 128)
        & (~protect[y0:])
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(samp.astype(np.uint8), 8)
    seeds = np.zeros((h, w), dtype=bool)
    for i in range(1, num):
        if not _sunny_water_sheet_ok(stats[i], h=h, w=w, frame=frame):
            continue
        m = labels == i
        # open water is away from nacelle/paint; blade-colored strips hug protect
        if float(prot_near[m].mean()) > 0.45:
            continue
        seeds[m] = True
    mean, stdv = _sample_lab(lab, seeds)
    meta = {
        "water_l": float(mean[0]) if mean is not None else None,
        "samp_px": int(seeds.sum()),
        "cut_px": 0,
    }
    if mean is None:
        return np.zeros((h, w), dtype=bool), meta
    matched = _match_lab(lab, mean, stdv, k=2.35)
    allow = (
        matched
        & (~protect)
        & (~veg)
        & (std < 7.5)
        & (S < 120)
        & (V < 185)
        & (V > 40)
        & blue
        & ~((S <= 50) & (V >= 170))
    )
    allow[:y0] = False
    barrier = _overcast_edge_barrier(gray)
    allow_u8 = (allow.astype(np.uint8) * 255)
    allow_u8[barrier > 0] = 0
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    markers = cv2.bitwise_and(cv2.dilate(seeds.astype(np.uint8) * 255, k5), allow_u8)
    if markers.max() == 0:
        return np.zeros((h, w), dtype=bool), meta
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rec = markers.copy()
    for _ in range(80):
        nxt = cv2.bitwise_and(cv2.dilate(rec, k3), allow_u8)
        if np.array_equal(nxt, rec):
            break
        rec = nxt
    out = rec > 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    keep = np.zeros((h, w), dtype=bool)
    for i in range(1, num):
        ar = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = bw / max(bh, 1)
        m = labels == i
        if bh / float(max(h, 1)) > 0.45:
            continue
        if float(prot_near[m].mean()) > 0.55 and aspect > 4.0 and bh / h < 0.16:
            continue
        if ar >= int(0.002 * frame) and (aspect >= 1.8 or ar >= int(0.01 * frame)):
            keep[m] = True
    meta["cut_px"] = int(keep.sum())
    return keep, meta


def _sunny_hsv_sky_water(
    bgr: np.ndarray, meta: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sunny sky/water bans from hand-picked HSV (deg/% → OpenCV).

    Hand samples (2026-10-11):
      sky   hsv(215.6, 15.6, 80.4) → OCV H≈108 S≈40  V≈205
      water hsv(216.9, 49.6, 55.3) → OCV H≈108 S≈126 V≈141
    Paint (no hand sample yet): low-S bright cream — protect only.
    Do NOT fold dens/edge into water protect (horizon waves looked "near paint").
    """
    h, w = bgr.shape[:2]
    frame = float(h * w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    B, G, R = cv2.split(bgr)
    veg = meta["veg"]
    # cream / white paint — never sky or water
    paint = (S <= 30) & (V >= 170)

    # sky: mid sat around hand S≈40, bright (light haze down to V≈160)
    sky_raw = (
        (H >= 95)
        & (H <= 130)
        & (S >= 24)
        & (S <= 72)
        & (V >= 160)
        & (~paint)
        & (~veg)
    )
    # water: high sat around hand S≈126; allow hazier S≳80 / darker V≳70
    water_raw = (
        (H >= 95)
        & (H <= 130)
        & (S >= 80)
        & (S <= 175)
        & (V >= 70)
        & (V <= 165)
        & (B.astype(np.int16) > G.astype(np.int16) + 6)
        & (B.astype(np.int16) > R.astype(np.int16) + 4)
        & (~paint)
        & (~veg)
    )

    # sky: only CCs touching the literal top edge (avoid nacelle/hub)
    edge_h = max(3, int(0.02 * h))
    y_cap = int(0.40 * h)
    sky_cut = np.zeros((h, w), dtype=bool)
    if sky_raw.any():
        num, labels = cv2.connectedComponents(sky_raw.astype(np.uint8), 8)
        top_labs = {int(v) for v in np.unique(labels[:edge_h]) if v > 0}
        for i in top_labs:
            comp = labels == i
            if comp[y_cap:].any():
                comp = comp & (np.arange(h)[:, None] <= y_cap)
                if not comp[:edge_h].any():
                    continue
            sky_cut |= comp
        sky_cut[edge_h:] &= ~paint[edge_h:]

    # water: wide sheets; reject only if glued to cream paint (not dens/edge)
    paint_near = cv2.dilate(paint.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    water_cut = np.zeros((h, w), dtype=bool)
    if water_raw.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            water_raw.astype(np.uint8), 8
        )
        for i in range(1, num):
            ar = int(stats[i, cv2.CC_STAT_AREA])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = bw / max(bh, 1)
            bh_frac = bh / float(max(h, 1))
            # slightly looser thin distant strip than shared sheet_ok (0390 sea line)
            thin_ok = ar >= int(0.0018 * frame) and aspect >= 4.0 and bh_frac <= 0.14
            if not (
                thin_ok or _sunny_water_sheet_ok(stats[i], h=h, w=w, frame=frame)
            ):
                continue
            m = labels == i
            if float(paint_near[m].mean()) > 0.55:
                continue
            water_cut |= m

    info = {
        "sky_l": float(meta["lab"][:, :, 0][sky_cut].mean()) if sky_cut.any() else None,
        "water_l": (
            float(meta["lab"][:, :, 0][water_cut].mean()) if water_cut.any() else None
        ),
        "hand_hsv": True,
        "paint_px": int(paint.sum()),
        "sky_px": int(sky_cut.sum()),
        "water_px": int(water_cut.sum()),
    }
    return sky_cut, water_cut, info


def _sunny_sky_water_ban_masks(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sunny bans: hand-picked HSV sky/water (not overcast Lab expand)."""
    _sky_ignored, _water_ignored, meta = _sky_water_ban_masks(bgr)
    sky_cut, water_cut, info = _sunny_hsv_sky_water(bgr, meta)
    meta = dict(meta)
    if info.get("sky_l") is not None:
        meta["sky_l"] = info["sky_l"]
    if info.get("water_l") is not None:
        meta["water_l"] = info["water_l"]
    meta["sunny_hsv"] = info
    return sky_cut, water_cut, meta


def _sunny_contour_seed(
    bgr: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[int, int] | None,
    str,
    np.ndarray,
    np.ndarray,
    float | None,
    float | None,
    np.ndarray,
]:
    """Sunny: same contour/basin/feat path as overcast, sunny water bans."""
    return _sky_water_contour_seed(bgr, ban=_sunny_sky_water_ban_masks(bgr))


def debug_sunny_water_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
) -> SunnyWaterDebug:
    """Same seed path as sunny `_geometry_seed` (contour, else smooth fallback)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    small, scale = _resize_max_side_bgr(bgr, max_side)
    seed, hub, note, sky_cut, water_cut, sky_l, water_l, barrier = _sunny_contour_seed(
        small
    )
    h, w = small.shape[:2]
    min_px = int(0.01 * h * w)
    if seed.max() == 0 or int((seed > 0).sum()) < min_px:
        seed2, hub2, _n2, geo2 = _keep_hub_connected(_smooth_candidates(small))
        seed, hub = seed2, hub2
        note = f"FALLBACK smooth|{note}|{geo2}"
    else:
        note = f"contour|{note}"
    return SunnyWaterDebug(
        work_bgr=small,
        sky_cut=(sky_cut.astype(np.uint8) * 255),
        water_cut=(water_cut.astype(np.uint8) * 255),
        seed=seed,
        hub_xy=hub,
        note=note,
        sky_l=sky_l,
        water_l=water_l,
        scale=scale,
        edge_barrier=barrier,
    )


def overlay_sunny_water_debug(dbg: SunnyWaterDebug) -> np.ndarray:
    """Red=sky · orange=water · cyan=contour seed (sunny skeleton = overcast)."""
    out = dbg.work_bgr.copy()
    tint = out.copy()
    tint[dbg.sky_cut > 0] = (0, 0, 255)
    tint[dbg.water_cut > 0] = (0, 165, 255)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    if dbg.seed.max() > 0:
        out = overlay_mask_with_hub(out, dbg.seed, dbg.hub_xy)
    lines = [
        "SUNNY hand-HSV  RED=sky  ORANGE=water  CYAN=seed  (basin+feat)",
        dbg.note,
        f"skyL={dbg.sky_l} waterL={dbg.water_l}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# 1.1a seed auxiliary lines (draw only — link broken rim edges; no mask cut yet)
# ---------------------------------------------------------------------------


@dataclass
class SeedAuxDebug:
    """1.1a: broken rim Hough segs + linked polylines (inspect before cutting)."""

    work_bgr: np.ndarray
    seed: np.ndarray
    hub_xy: tuple[int, int] | None
    raw_segs: list[tuple[tuple[int, int], tuple[int, int]]]
    lower_segs: list[tuple[tuple[int, int], tuple[int, int]]]
    upper_segs: list[tuple[tuple[int, int], tuple[int, int]]]
    linked_chains: list[list[tuple[int, int]]]
    note: str
    scale: float
    seed_note: str = ""


def _seed_rim_band(seed: np.ndarray, *, band: int = 18) -> np.ndarray:
    """Pixels near seed contour (inside+outside)."""
    if seed.max() == 0:
        return np.zeros_like(seed)
    u8 = (seed > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    dil = cv2.dilate(u8, k)
    ero = cv2.erode(u8, k)
    return ((dil > 0) & (ero == 0)).astype(np.uint8) * 255


def _seg_near_mask(
    p1: tuple[int, int],
    p2: tuple[int, int],
    band: np.ndarray,
    *,
    min_hit: float = 0.35,
) -> bool:
    h, w = band.shape[:2]
    pts = _sample_seg_points(p1, p2, n=14)
    hit = 0
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h and band[y, x] > 0:
            hit += 1
    return hit / max(len(pts), 1) >= min_hit


def _seg_side_of_seed(
    p1: tuple[int, int],
    p2: tuple[int, int],
    seed: np.ndarray,
    hub_xy: tuple[int, int] | None,
) -> str:
    """Classify rim seg as lower / upper / other relative to seed centroid."""
    ys, xs = np.where(seed > 0)
    if len(xs) < 20:
        return "other"
    cx = float(xs.mean())
    cy = float(ys.mean())
    if hub_xy is not None:
        cx, cy = float(hub_xy[0]), float(hub_xy[1])
    mid = 0.5 * (np.array(p1, dtype=np.float64) + np.array(p2, dtype=np.float64))
    # prefer vertical offset from centroid (blade spans horizontally on 0735)
    if mid[1] >= cy + 8:
        return "lower"
    if mid[1] <= cy - 8:
        return "upper"
    return "other"


def _seg_is_blade_boundary(
    p1: tuple[int, int],
    p2: tuple[int, int],
    seed: np.ndarray,
    veg: np.ndarray,
    dt_seed: np.ndarray,
    *,
    max_inside: float = 7.0,
) -> bool:
    """Keep segs that straddle seed|outside on the lower rim; drop sky/house junk.

    Rejects:
    - floaters deep inside seed (sky edges welded into cyan)
    - roof/field edges with both sides outside seed
    - segs mostly on vegetation (house roofs / tree texture, not blade silhouette)
    """
    h, w = seed.shape[:2]
    direction, length, mid = _seg_geometry(p1, p2)
    if length < 18:
        return False
    # lower rim is near-horizontal on 0735-class shots
    ang = abs(float(np.arctan2(direction[1], direction[0])))
    ang = min(ang, abs(np.pi - ang))
    if ang > np.deg2rad(28.0):
        return False

    mx, my = int(round(mid[0])), int(round(mid[1]))
    if not (0 <= mx < w and 0 <= my < h):
        return False
    if float(dt_seed[my, mx]) > max_inside:
        return False  # deep inside seed → sky floater

    # normal pointing "downward-ish"
    n = np.array([-direction[1], direction[0]], dtype=np.float64)
    if n[1] < 0:
        n = -n
    n = n / (float(np.linalg.norm(n)) + 1e-9)

    def _probe(sign: float, dist: float = 10.0) -> tuple[float, float, float]:
        """Return (seed_frac, veg_frac, valid_frac) along offset samples."""
        pts = _sample_seg_points(p1, p2, n=10)
        seed_hit = veg_hit = valid = 0
        for x, y in pts:
            qx = int(round(x + sign * n[0] * dist))
            qy = int(round(y + sign * n[1] * dist))
            if 0 <= qx < w and 0 <= qy < h:
                valid += 1
                if seed[qy, qx] > 0:
                    seed_hit += 1
                if veg[qy, qx]:
                    veg_hit += 1
        if valid < 4:
            return 0.0, 0.0, 0.0
        return seed_hit / valid, veg_hit / valid, 1.0

    # +n = below rim, -n = above rim
    seed_below, veg_below, ok_b = _probe(+1.0)
    seed_above, veg_above, ok_a = _probe(-1.0)
    if ok_a < 1.0 or ok_b < 1.0:
        return False

    # must straddle: more seed above than below
    if seed_above < 0.45 or seed_below > 0.55:
        return False
    if seed_above - seed_below < 0.18:
        return False

    # house roofs / orchard texture sit below with high veg; allow some,
    # but reject if the SEG ITSELF rides vegetation (roof ridge)
    pts = _sample_seg_points(p1, p2, n=12)
    veg_on = 0
    valid = 0
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            valid += 1
            if veg[y, x]:
                veg_on += 1
    if valid and veg_on / valid > 0.35:
        return False

    return True


def _chain_length(pts: list[tuple[int, int]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        total += float(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i])))
    return total


def _link_broken_segs(
    segs: list[tuple[tuple[int, int], tuple[int, int]]],
    *,
    angle_tol_deg: float = 12.0,
    gap_px: float = 40.0,
    cross_px: float = 12.0,
    max_dy: float = 14.0,
) -> list[list[tuple[int, int]]]:
    """Greedy link near-collinear broken segments into polylines.

    Grow by attaching collinear neighbors, then re-sort each chain by PCA
    projection so joints don't zigzag (0735 lower rim).
    ``max_dy`` blocks sky-floater ↔ house-roof bridges.
    """
    if not segs:
        return []

    items: list[dict] = []
    for p1, p2 in segs:
        direction, length, mid = _seg_geometry(p1, p2)
        if length < 8:
            continue
        ang = float(np.arctan2(direction[1], direction[0]))
        if ang < 0:
            ang += np.pi
        items.append(
            {
                "p1": p1,
                "p2": p2,
                "dir": direction,
                "len": length,
                "mid": mid,
                "ang": ang,
                "used": False,
            }
        )
    if not items:
        return []

    items.sort(key=lambda d: -d["len"])
    chains: list[list[tuple[int, int]]] = []
    ang_tol = np.deg2rad(angle_tol_deg)

    def _ang_diff(a: float, b: float) -> float:
        d = abs(a - b) % np.pi
        return float(min(d, np.pi - d))

    def _cross_dist(origin, direction, pt) -> float:
        v = pt - origin
        return abs(float(v[0] * direction[1] - v[1] * direction[0]))

    def _order_chain(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(pts) <= 2:
            return pts
        arr = np.array(pts, dtype=np.float64)
        mean = arr.mean(axis=0)
        _, eig = cv2.PCACompute(arr.astype(np.float32), mean=None)
        d = eig[0].astype(np.float64)
        proj = (arr - mean) @ d
        ordered = [pts[int(i)] for i in np.argsort(proj)]
        cleaned: list[tuple[int, int]] = [ordered[0]]
        for p in ordered[1:]:
            if float(np.linalg.norm(np.array(p) - np.array(cleaned[-1]))) >= 3.0:
                cleaned.append(p)
        return cleaned

    for seed_item in items:
        if seed_item["used"]:
            continue
        seed_item["used"] = True
        members = [seed_item]
        changed = True
        while changed:
            changed = False
            mids = np.array([m["mid"] for m in members], dtype=np.float64)
            mean = mids.mean(axis=0)
            if len(members) == 1:
                chain_dir = members[0]["dir"]
                chain_ang = members[0]["ang"]
            else:
                _, eig = cv2.PCACompute(mids.astype(np.float32), mean=None)
                chain_dir = eig[0].astype(np.float64)
                n = float(np.linalg.norm(chain_dir)) + 1e-9
                chain_dir = chain_dir / n
                chain_ang = float(np.arctan2(chain_dir[1], chain_dir[0]))
                if chain_ang < 0:
                    chain_ang += np.pi

            ends: list[np.ndarray] = []
            for m in members:
                ends.append(np.array(m["p1"], dtype=np.float64))
                ends.append(np.array(m["p2"], dtype=np.float64))
            ends_arr = np.stack(ends, axis=0)
            proj = (ends_arr - mean) @ chain_dir
            head = ends_arr[int(np.argmin(proj))]
            tail = ends_arr[int(np.argmax(proj))]
            mean_y = float(mids[:, 1].mean())

            best_j = -1
            best_gap = 1e9
            for j, cand in enumerate(items):
                if cand["used"]:
                    continue
                if _ang_diff(chain_ang, cand["ang"]) > ang_tol:
                    continue
                if abs(float(cand["mid"][1]) - mean_y) > max_dy:
                    continue  # don't bridge sky ↔ roof
                if _cross_dist(mean, chain_dir, cand["mid"]) > cross_px + 6.0:
                    continue
                for ep in (
                    np.array(cand["p1"], dtype=np.float64),
                    np.array(cand["p2"], dtype=np.float64),
                ):
                    gap = min(
                        float(np.linalg.norm(ep - head)),
                        float(np.linalg.norm(ep - tail)),
                    )
                    if gap <= gap_px and gap < best_gap:
                        if abs(float(ep[1]) - mean_y) > max_dy + 4.0:
                            continue
                        if _cross_dist(mean, chain_dir, ep) > cross_px + 8.0:
                            continue
                        best_j = j
                        best_gap = gap
            if best_j >= 0:
                items[best_j]["used"] = True
                members.append(items[best_j])
                changed = True

        pts: list[tuple[int, int]] = []
        for m in members:
            pts.append(m["p1"])
            pts.append(m["p2"])
        ordered = _order_chain(pts)
        if len(ordered) >= 2:
            chains.append(ordered)

    chains.sort(key=lambda c: -_chain_length(c))
    return chains


def _seed_aux_from_seed(
    work_bgr: np.ndarray,
    seed: np.ndarray,
    hub_xy: tuple[int, int] | None,
    *,
    seed_note: str = "",
) -> SeedAuxDebug:
    """Draw-only: Hough near seed rim → keep straddling lower boundary → link."""
    h, w = work_bgr.shape[:2]
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    veg = _vegetation_mask(work_bgr)
    dt_seed = cv2.distanceTransform((seed > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # shorter segs + larger gap so 0735 broken lower rim still appears
    raw = _detect_hough_segments(gray, min_len=22, max_gap=28)
    band = _seed_rim_band(seed, band=max(14, int(0.018 * max(h, w))))
    rim: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for p1, p2 in raw:
        _, length, _ = _seg_geometry(p1, p2)
        if length < 18:
            continue
        if _seg_near_mask(p1, p2, band, min_hit=0.28):
            rim.append((p1, p2))

    lower: list[tuple[tuple[int, int], tuple[int, int]]] = []
    upper: list[tuple[tuple[int, int], tuple[int, int]]] = []
    other: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for seg in rim:
        side = _seg_side_of_seed(seg[0], seg[1], seed, hub_xy)
        if side == "lower":
            lower.append(seg)
        elif side == "upper":
            upper.append(seg)
        else:
            other.append(seg)

    # only link segs that look like blade|background boundary (no sky/house)
    boundary = [
        seg
        for seg in lower
        if _seg_is_blade_boundary(seg[0], seg[1], seed, veg, dt_seed)
    ]
    if len(boundary) < 2:
        # slightly looser inside-depth if too few
        boundary = [
            seg
            for seg in lower
            if _seg_is_blade_boundary(
                seg[0], seg[1], seed, veg, dt_seed, max_inside=11.0
            )
        ]

    chains = _link_broken_segs(
        boundary, angle_tol_deg=12.0, gap_px=40.0, cross_px=12.0, max_dy=14.0
    )
    note = (
        f"seed_aux rim={len(rim)} lower={len(lower)} upper={len(upper)} "
        f"boundary={len(boundary)} chains={len(chains)} "
        f"best_len={_chain_length(chains[0]) if chains else 0:.0f}"
    )
    return SeedAuxDebug(
        work_bgr=work_bgr,
        seed=seed,
        hub_xy=hub_xy,
        raw_segs=rim,
        lower_segs=boundary,  # show filtered lower (not all orange junk)
        upper_segs=upper,
        linked_chains=chains,
        note=note,
        scale=1.0,
        seed_note=seed_note,
    )


def debug_seed_aux_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
) -> SeedAuxDebug:
    """1.1a auxiliary lines on current overcast/sunny geometric seed (draw only)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    # reuse white-refine work size so seed matches 1.1a view
    wr = debug_white_refine(bgr, max_side=max_side, year=year)
    dbg = _seed_aux_from_seed(
        wr.work_bgr, wr.seed, wr.hub_xy, seed_note=wr.seed_detail or wr.note
    )
    dbg.scale = wr.scale
    return dbg


def overlay_seed_aux_debug(dbg: SeedAuxDebug) -> np.ndarray:
    """1.1a aux: cyan seed · yellow rim segs · blue upper · orange lower · green linked."""
    out = dbg.work_bgr.copy()
    if dbg.seed.max() > 0:
        tint = out.copy()
        tint[dbg.seed > 0] = (255, 255, 0)
        out = cv2.addWeighted(tint, 0.22, out, 0.78, 0)
        cnts, _ = cv2.findContours(
            (dbg.seed > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, cnts, -1, (255, 220, 0), 1, lineType=cv2.LINE_AA)

    def _draw(seg, color, thickness=1):
        cv2.line(out, seg[0], seg[1], color, thickness, lineType=cv2.LINE_AA)

    for seg in dbg.raw_segs:
        _draw(seg, (0, 255, 255), 1)  # yellow = all rim
    for seg in dbg.upper_segs:
        _draw(seg, (255, 180, 80), 2)  # blue-ish = upper (usually OK)
    for seg in dbg.lower_segs:
        _draw(seg, (0, 140, 255), 2)  # orange = lower (contamination side)

    for ci, chain in enumerate(dbg.linked_chains[:6]):
        color = (0, 255, 0) if ci == 0 else (80, 220, 80)  # green linked
        th = 3 if ci == 0 else 2
        for i in range(len(chain) - 1):
            cv2.line(out, chain[i], chain[i + 1], color, th, lineType=cv2.LINE_AA)
        # mark joints (where broken segs were bridged)
        for p in chain:
            cv2.circle(out, p, 3, color, -1, lineType=cv2.LINE_AA)

    if dbg.hub_xy is not None:
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    lines = [
        "1.1a SEED AUX (draw only — no cut yet)",
        "CYAN=seed  YELLOW=rim Hough  BLUE=upper  ORANGE=boundary",
        "GREEN=linked (no sky/house bridge)  joints=dots",
        dbg.note,
    ]
    for i, text in enumerate(lines):
        col = (80, 220, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA
        )
    return out


def _geometry_seed(
    bgr: np.ndarray, lighting: str
) -> tuple[np.ndarray, tuple[int, int] | None, int, str]:
    """Sunny/overcast: ban → contour∪basin → feat score; fallback smooth→hub."""
    h, w = bgr.shape[:2]
    min_px = int(0.01 * h * w)

    if lighting != "overcast":
        seed, hub, note, _sky, _water, _sl, _wl, _bar = _sunny_contour_seed(bgr)
        if seed.max() == 0 or int((seed > 0).sum()) < min_px:
            seed2, hub2, n2, geo2 = _keep_hub_connected(_smooth_candidates(bgr))
            return seed2, hub2, n2, f"seed=sunny_fallback|{note}|{geo2}"
        n = 1 if seed.max() > 0 else 0
        return seed, hub, n, f"seed=sunny_sw|{note}"

    seed, hub, note, _sky, _water, _sl, _wl, _bar = _sky_water_contour_seed(bgr)
    if seed.max() == 0 or int((seed > 0).sum()) < min_px:
        seed2, hub2, n2, geo2 = _keep_hub_connected(_smooth_candidates(bgr))
        return seed2, hub2, n2, f"seed=overcast_fallback|{note}|{geo2}"
    n = 1 if seed.max() > 0 else 0
    return seed, hub, n, f"seed=overcast_sw|{note}"


def _hub_from_distance(mask: np.ndarray) -> tuple[tuple[int, int] | None, np.ndarray]:
    if mask.max() == 0:
        return None, np.zeros_like(mask, dtype=np.float32)
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(dt)
    if max_v < 2.0:
        return None, dt
    return (int(max_l[0]), int(max_l[1])), dt


def _arm_taper_score(
    mask: np.ndarray,
    hub_xy: tuple[int, int],
    dt: np.ndarray,
    *,
    n_rays: int = 48,
) -> tuple[float, int]:
    h, w = mask.shape[:2]
    hx, hy = hub_xy
    long_arms = 0
    taper_hits = 0
    for i in range(n_rays):
        ang = 2.0 * np.pi * i / n_rays
        dx, dy = np.cos(ang), np.sin(ang)
        widths: list[float] = []
        length = 0
        for step in range(1, max(h, w)):
            x = int(round(hx + dx * step))
            y = int(round(hy + dy * step))
            if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
                break
            widths.append(float(dt[y, x]))
            length = step
        if length < 0.08 * max(h, w):
            continue
        long_arms += 1
        if len(widths) < 6:
            continue
        root = float(np.mean(widths[: max(2, len(widths) // 5)]))
        tip = float(np.mean(widths[-max(2, len(widths) // 5) :]))
        if root > tip * 1.12 and root > 2.0:
            taper_hits += 1
    return long_arms + 1.5 * taper_hits, long_arms


def _rectangularity(comp_mask: np.ndarray) -> float:
    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 30:
        return 1.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]
    rect_area = max(float(rw), 1.0) * max(float(rh), 1.0)
    return float(len(xs)) / rect_area


def _elongation(comp_mask: np.ndarray) -> float:
    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 30:
        return 1.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eig = cv2.PCACompute(pts, mean=None)
    proj = (pts - mean) @ eig.T
    extent = proj.max(axis=0) - proj.min(axis=0)
    return float(max(extent[0], extent[1]) / (min(extent[0], extent[1]) + 1e-3))


def _keep_hub_connected(candidates: np.ndarray) -> tuple[np.ndarray, tuple[int, int] | None, int, str]:
    h, w = candidates.shape[:2]
    if candidates.max() == 0:
        return candidates, None, 0, "empty_candidates"

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidates > 0).astype(np.uint8), connectivity=8
    )
    frame = float(h * w)
    scored: list[tuple[float, int, tuple[int, int]]] = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.008 * frame or area > 0.7 * frame:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        hub, dt = _hub_from_distance(comp)
        if hub is None:
            continue
        star, _n_arms = _arm_taper_score(comp, hub, dt)
        elong = _elongation(comp)
        rect = _rectangularity(comp)
        # turbine: elongated or multi-arm; buildings: rectangular & compact
        score = (
            star * 2.0
            + min(elong, 12.0) * 1.2
            + np.sqrt(area / frame) * 4.0
            - rect * 5.0
        )
        scored.append((float(score), i, hub))

    if not scored:
        best_i, best_a = -1, 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 0.008 * frame <= area <= 0.55 * frame and area > best_a:
                best_a, best_i = area, i
        if best_i < 0:
            return np.zeros_like(candidates), None, 0, "no_component"
        out = (labels == best_i).astype(np.uint8) * 255
        hub, _ = _hub_from_distance(out)
        return out, hub, 1, "fallback_largest"

    scored.sort(reverse=True)
    best_score, best_i, best_hub = scored[0]
    out = (labels == best_i).astype(np.uint8) * 255
    kept = 1
    hub_r = 0.05 * max(h, w)
    for score, i, _hub in scored[1:]:
        cx, cy = centroids[i]
        if np.hypot(cx - best_hub[0], cy - best_hub[1]) > hub_r * 4.0:
            continue
        if score < best_score * 0.3:
            continue
        out[labels == i] = 255
        kept += 1

    # Cut tip-bridges (tree line / hedge) before any dilation can thicken them.
    out = _cut_distant_bridges(out, best_hub, k=21)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grown = cv2.dilate(out, k, iterations=1)
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_in = grown.copy()
    sx, sy = best_hub
    if grown[sy, sx] == 0:
        ys, xs = np.where(grown > 0)
        if len(xs) == 0:
            return np.zeros_like(candidates), None, 0, "hub_outside"
        j = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
        sx, sy = int(xs[j]), int(ys[j])
        best_hub = (sx, sy)
    cv2.floodFill(flood_in, ff_mask, (sx, sy), 128, flags=4 | (255 << 8))
    connected = np.where(flood_in == 128, 255, 0).astype(np.uint8)
    # Fill interior holes only. A wide close here re-bridges the blade tip to
    # distant hedges (058) and the white grow then swallows the road.
    connected = _fill_holes(connected)

    detail = f"best_score={best_score:.2f} kept_cc={kept} hub=({best_hub[0]},{best_hub[1]})"
    return connected, best_hub, kept, detail


def _bright_lowsat(bgr: np.ndarray, *, s_max: int = 60, v_min: int = 120) -> np.ndarray:
    """Bright + low-saturation — used to GROW edges / CULL non-turbine, not to seed.

    白色不能用来判断「是风机」，但可以判断「还像漆面、可以往外补」
    以及「明显不是漆面（绿田/深色）该剔除」。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    return (s <= s_max) & (v >= v_min)


def _strip_sky_from_domain(domain: np.ndarray) -> np.ndarray:
    """Drop huge bright blobs that hug the top border (sky), keep turbine white."""
    h, w = domain.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (domain > 0).astype(np.uint8), 8
    )
    out = domain.copy()
    top = max(1, h // 40)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        touch = int((labels[:top, :] == i).sum())
        if touch > 0.2 * w * top and area > 0.12 * h * w:
            out[labels == i] = 0
    return out


def _cut_distant_bridges(
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
    *,
    k: int = 21,
) -> np.ndarray:
    """Drop appendages joined by a neck thinner than the turbine body.

    058-class: a smooth tree line touches the blade tip and becomes one CC.
    Opening breaks that neck; thickness is restored only inside the original
    component, with a radius too small to seal the cut.

    0713-class: overcast sky/ground peel can thin a real blade mid-body so the
    same open splits tip vs lower leaf. Keep large *elongated* secondary pieces
    (blade continuation); still drop compact scraps (road / hedge).
    """
    if mask.max() == 0 or hub_xy is None:
        return mask
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    h, w = mask.shape[:2]
    hx = int(np.clip(hub_xy[0], 0, w - 1))
    hy = int(np.clip(hub_xy[1], 0, h - 1))
    # Opening erased the hub: the turbine body is thinner than the kernel
    # (0752 close-up). Keep the original component instead of jumping to a field blob.
    if opened[hy, hx] == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (opened > 0).astype(np.uint8), 8
    )
    best_i = int(labels[hy, hx])
    if best_i == 0 or int(stats[best_i, cv2.CC_STAT_AREA]) < 300:
        return mask
    core = (labels == best_i).astype(np.uint8) * 255
    core_area = int(stats[best_i, cv2.CC_STAT_AREA])
    frame = float(h * w)
    # Re-attach blade-like shards split by a thinned mid-body neck (0713).
    min_sec = max(0.12 * core_area, 0.012 * frame)
    for i in range(1, num):
        if i == best_i:
            continue
        area_i = int(stats[i, cv2.CC_STAT_AREA])
        if area_i < min_sec:
            continue
        shard = (labels == i).astype(np.uint8) * 255
        if _elongation(shard) >= 5.5:
            core = cv2.bitwise_or(core, shard)
    rad = max(3, k // 2 - 4)
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
    restored = cv2.bitwise_and(cv2.dilate(core, ker2), mask)
    return restored


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes that do not touch the image border.

    A blade that spans the full width must not swallow the ground on the
    other side (0752). Only cavities sealed inside the component are filled.
    """
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 128)
    holes = padded[1:-1, 1:-1] == 0
    out = mask.copy()
    out[holes] = 255
    return out


def _fill_small_interior_holes(
    mask: np.ndarray, *, max_area: int = 2500
) -> np.ndarray:
    """Fill tiny sealed holes (0752 rust punched by veg ban after close).

    Only component holes fully enclosed in the mask; open bays untouched.
    Caps area so large false interiors are not swallowed.
    """
    m = (mask > 0).astype(np.uint8) * 255
    if m.max() == 0:
        return m
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return m
    out = m.copy()
    for i, c in enumerate(cnts):
        if hier[0, i, 3] < 0:
            continue  # external
        if cv2.contourArea(c) > max_area:
            continue
        cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out


def _outer_contour_solid(mask: np.ndarray) -> np.ndarray:
    """Filled external contours of ``mask`` (silhouette shell)."""
    solid = np.zeros_like(mask)
    if mask.max() == 0:
        return solid
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = 200.0
    for c in cnts:
        if cv2.contourArea(c) >= min_a:
            cv2.drawContours(solid, [c], -1, 255, thickness=cv2.FILLED)
    return solid


def _gated_interior_fill(
    bgr: np.ndarray, seed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fill holes inside the outer contour when they look like blade interior.

    Returns (grown, filled_added, rejected_holes, stats).
    Never grows past the outer silhouette — exterior missing chunks are 1.3.
    """
    h, w = seed.shape[:2]
    seed_u = (seed > 0).astype(np.uint8) * 255
    seed_area = max(int((seed_u > 0).sum()), 1)

    # Bridge hairline gaps so the outer contour is readable, but clamp bloom.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(seed_u, cv2.MORPH_CLOSE, k, iterations=1)
    dil = cv2.dilate(seed_u, np.ones((5, 5), np.uint8))
    closed = cv2.bitwise_and(closed, dil)
    closed = cv2.bitwise_or(closed, seed_u)

    solid = _outer_contour_solid(closed)
    interior = ((solid > 0) & (closed == 0)).astype(np.uint8) * 255

    sky_ban, water_ban, ban_meta = _sky_water_ban_masks(bgr)
    veg = ban_meta["veg"]
    ban = sky_ban | water_ban | veg

    filled = np.zeros_like(seed_u)
    rejected = np.zeros_like(seed_u)
    n_fill = 0
    n_reject = 0
    fill_px = 0
    reject_px = 0

    if interior.max() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (interior > 0).astype(np.uint8), 8
        )
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            hole = labels == i
            # Border-touching "holes" are usually open background pockets.
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            touches_border = (
                x0 <= 1
                or y0 <= 1
                or (x0 + bw) >= (w - 2)
                or (y0 + bh) >= (h - 2)
            )
            ban_frac = float((hole & ban).sum()) / float(area)
            too_big = area > 0.22 * seed_area
            # Sky/water-like cavity: bright, flat, low sat (even if Lab ban missed).
            roi_std = float(ban_meta["std"][hole].mean()) if area else 99.0
            roi_s = float(ban_meta["S"][hole].mean()) if area else 99.0
            looks_skywater = roi_std < 3.5 and roi_s < 35 and ban_frac > 0.12

            if touches_border or too_big or ban_frac > 0.35 or looks_skywater:
                rejected[hole] = 255
                n_reject += 1
                reject_px += area
                continue
            filled[hole] = 255
            n_fill += 1
            fill_px += area

    grown = cv2.bitwise_or(closed, filled)
    # Absolute clamp: never leave the outer silhouette.
    grown = cv2.bitwise_and(grown, solid) if solid.max() > 0 else grown
    grown = cv2.bitwise_or(grown, seed_u)

    added = cv2.bitwise_and(grown, cv2.bitwise_not(seed_u))
    stats_out = {
        "n_fill": n_fill,
        "n_reject": n_reject,
        "fill_px": fill_px,
        "reject_px": reject_px,
        "close_add": int(((closed > 0) & (seed_u == 0)).sum()),
    }
    return grown, added, rejected, stats_out


def _refine_mask_overcast_interior_fill(
    bgr: np.ndarray, seed_mask: np.ndarray
) -> tuple[np.ndarray, str, dict]:
    """Overcast 1.1b: skip far white-grow; gated fill inside outer contour.

    Exterior missing blade chunks are left for 1.3 (trend-line / rim grow).
    """
    seed = (seed_mask > 0).astype(np.uint8) * 255
    white = _bright_lowsat(bgr, s_max=70, v_min=110)
    veg = _vegetation_mask(bgr)
    domain = (white & (~veg)).astype(np.uint8) * 255
    core = cv2.bitwise_and(seed, domain)
    seed_count = max(int((seed > 0).sum()), 1)
    white_frac = float((core > 0).sum()) / float(seed_count)

    if seed.max() == 0:
        empty = np.zeros_like(seed)
        return (
            seed,
            "refine=overcast_interior_fill+empty",
            {
                "seed": seed,
                "core": core,
                "domain": domain,
                "white_frac": white_frac,
                "max_dist": 0.0,
                "added": empty,
                "rejected": empty,
                "mode": "overcast_interior_fill",
            },
        )

    grown, added, rejected, st = _gated_interior_fill(bgr, seed)
    note = (
        f"refine=overcast_interior_fill+skip_white_grow"
        f"+holes_fill={st['n_fill']}/{st['fill_px']}px"
        f"+holes_reject={st['n_reject']}/{st['reject_px']}px"
        f"+close={st['close_add']}"
        f"+dmax=0+white_frac={white_frac:.2f}+add={int((added > 0).sum())}"
    )
    dbg = {
        "seed": seed,
        "core": core,
        "domain": domain,
        "white_frac": white_frac,
        "max_dist": 0.0,
        "added": added,
        "rejected": rejected,
        "mode": "overcast_interior_fill",
    }
    return grown, note, dbg


def _refine_mask_white_rules(
    bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    max_grow_iters: int = 120,
    lighting: str = "sunny",
) -> tuple[np.ndarray, str]:
    """Sunny: cull non-white + paint grow. Overcast: gated interior fill."""
    grown, note, _dbg = _refine_mask_white_rules_ex(
        bgr, seed_mask, max_grow_iters=max_grow_iters, lighting=lighting
    )
    return grown, note


def _refine_mask_white_rules_ex(
    bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    max_grow_iters: int = 120,
    lighting: str = "sunny",
) -> tuple[np.ndarray, str, dict]:
    """Same as white refine, plus intermediates for step debug overlays."""
    if lighting == "overcast":
        return _refine_mask_overcast_interior_fill(bgr, seed_mask)

    h, w = bgr.shape[:2]
    veg = _vegetation_mask(bgr)
    white = _bright_lowsat(bgr, s_max=70, v_min=110)
    domain = (white & (~veg)).astype(np.uint8) * 255
    domain = _strip_sky_from_domain(domain)
    # Bridge dark lettering / thin gaps on nacelle so blades stay one CC with tower
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    domain = cv2.morphologyEx(domain, cv2.MORPH_CLOSE, k_bridge, iterations=2)

    seed = (seed_mask > 0).astype(np.uint8) * 255
    # Geometric seed stays, even when paint is field-tinted (058). White is only
    # a grow/cull signal for pixels outside that seed.
    protected = seed
    core = cv2.bitwise_and(seed, domain)
    culled = int((seed > 0).sum() - (core > 0).sum())
    seed_count = max(int((seed > 0).sum()), 1)
    white_frac = float((core > 0).sum()) / float(seed_count)
    if (core > 0).sum() < 0.05 * max((seed > 0).sum(), 1):
        core = cv2.bitwise_and(
            cv2.dilate(seed, np.ones((5, 5), np.uint8)), domain
        )
    empty_dbg = {
        "seed": protected,
        "core": core,
        "domain": domain,
        "white_frac": white_frac,
        "max_dist": 0.0,
        "added": np.zeros_like(protected),
        "mode": "sunny_white",
    }
    if (core > 0).sum() == 0:
        return protected, "refine=keep_seed_no_white", empty_dbg

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    touch = cv2.bitwise_and(cv2.dilate(core, k), domain)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (domain > 0).astype(np.uint8), 8
    )
    grown = np.zeros_like(seed)
    step_px = 2.0
    # Paint-white seeds (tower/nacelle, 0004) may grow out to the blades.
    # Tinted seeds (058) only take a narrow rim — a long radius reaches the
    # road and distant white roofs that sit next to the blade tip.
    if white_frac >= 0.55:
        max_dist = float(
            max(0.50 * max(h, w), min(max_grow_iters * step_px, 0.70 * max(h, w)))
        )
    else:
        max_dist = 18.0
    core_inv = np.where(core > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(core_inv, cv2.DIST_L2, 3)

    kept_cc = 0
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        roi_lab = labels[y : y + bh, x : x + bw]
        roi_core = core[y : y + bh, x : x + bw]
        roi_touch = touch[y : y + bh, x : x + bw]
        roi_dist = dist[y : y + bh, x : x + bw]
        inside = roi_lab == i
        if not inside.any():
            continue
        overlap_core = int(np.logical_and(inside, roi_core > 0).sum())
        overlap_touch = int(np.logical_and(inside, roi_touch > 0).sum())
        if overlap_core < 8 and overlap_touch < 40:
            continue
        dmin = float(roi_dist[inside].min())
        if overlap_core < 8 and dmin > min(40.0, max_dist * 0.25):
            continue
        # Never take a whole mega-CC (blade+road+buildings). Keep only paint
        # within max_dist of the geometric seed core.
        take = inside & (roi_dist <= max_dist)
        if overlap_core >= 8:
            # always keep pixels that already overlap core
            take = take | (inside & (roi_core > 0))
        if int(take.sum()) < 20:
            continue
        grown[y : y + bh, x : x + bw][take] = 255
        kept_cc += 1

    if grown.max() == 0:
        grown = core.copy()

    grown = cv2.bitwise_or(grown, core)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k9, iterations=2)
    grown = _fill_holes(grown)
    # Solidify hollow blades: fill external contours then keep paint-like pixels
    solid = np.zeros_like(grown)
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 400:
            continue
        cv2.drawContours(solid, [c], -1, 255, thickness=cv2.FILLED)
    # Paint-white fill stays inside the grow radius. Geometric seed is kept
    # even where the blade is field-tinted, so the interior is not hollowed out.
    grown = cv2.bitwise_or(cv2.bitwise_and(solid, domain), protected)
    prot_inv = np.where(protected > 0, 0, 255).astype(np.uint8)
    prot_dist = cv2.distanceTransform(prot_inv, cv2.DIST_L2, 3)
    grown[(protected == 0) & (prot_dist > max_dist)] = 0
    # final non-paint / vegetation strip outside the seed (white =「不是风机」)
    grown[(grown > 0) & veg & (protected == 0)] = 0
    grown[(grown > 0) & (~white) & (protected == 0)] = 0
    # 0335 compromise: keep long dmax (0190 rust needs the grow), but drop
    # added CCs that look like far road / roof lobes. Never touch seed pixels.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dens = _feature_density(gray)
    edge = _sobel_mag(gray)
    seed_rad = float(
        cv2.distanceTransform((protected > 0).astype(np.uint8), cv2.DIST_L2, 3).max()
    )
    added_m = (grown > 0) & (protected == 0)
    cc_cut = 0
    if added_m.any():
        num_a, lab_a, st_a, _ = cv2.connectedComponentsWithStats(
            added_m.astype(np.uint8), 8
        )
        for i in range(1, num_a):
            m = lab_a == i
            ar = int(st_a[i, cv2.CC_STAT_AREA])
            if ar < 80:
                continue
            mean_d = float(prot_dist[m].mean())
            mean_e = float(edge[m].mean())
            mean_f = float(dens[m].mean())
            bw = int(st_a[i, cv2.CC_STAT_WIDTH])
            bh = int(st_a[i, cv2.CC_STAT_HEIGHT])
            aspect = bw / max(bh, 1)
            far = mean_d > max(50.0, seed_rad * 1.35)
            structured = mean_e > 28.0 or mean_f > 0.09
            roadish = aspect >= 2.8 and mean_d > 36.0 and bh < 0.12 * h
            if (far and structured) or roadish:
                grown[m] = 0
                cc_cut += ar
    note = (
        f"refine=cull_nonwhite:{culled}+white_cc:{kept_cc}"
        f"+dmax={max_dist:.0f}+white_frac={white_frac:.2f}+cc_cut={cc_cut}"
    )
    added = cv2.bitwise_and(grown, cv2.bitwise_not(protected))
    dbg = {
        "seed": protected,
        "core": core,
        "domain": domain,
        "white_frac": white_frac,
        "max_dist": max_dist,
        "added": added,
        "mode": "sunny_white",
    }
    return grown, note, dbg


def _trace_dominant_arm(
    mask: np.ndarray,
    hub_xy: tuple[int, int],
    *,
    n_rays: int = 72,
) -> tuple[float, float, list[tuple[int, int, float]]] | None:
    """Longest hub→tip ray with local half-widths from distance transform.

    Returns (dx, dy, samples) where samples are (x, y, half_width) along the arm.
    """
    hub, dt = _hub_from_distance(mask)
    if hub is None:
        hub = hub_xy
    hx, hy = hub
    h, w = mask.shape[:2]
    best: list[tuple[int, int, float]] = []
    best_dir = (1.0, 0.0)
    for i in range(n_rays):
        ang = 2.0 * np.pi * i / n_rays
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        samples: list[tuple[int, int, float]] = []
        for step in range(1, max(h, w)):
            x = int(round(hx + dx * step))
            y = int(round(hy + dy * step))
            if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
                break
            samples.append((x, y, float(dt[y, x])))
        if len(samples) > len(best):
            best = samples
            best_dir = (dx, dy)
    if len(best) < max(40, int(0.08 * max(h, w))):
        return None
    return best_dir[0], best_dir[1], best


def _perpendicular_half_width(
    mask: np.ndarray,
    x: int,
    y: int,
    dx: float,
    dy: float,
    *,
    max_r: int = 80,
) -> float:
    """Half-width of mask along the normal at (x,y)."""
    h, w = mask.shape[:2]
    nx, ny = -dy, dx
    nrm = float(np.hypot(nx, ny)) + 1e-6
    nx, ny = nx / nrm, ny / nrm

    a = 0
    for r in range(1, max_r + 1):
        xx = int(round(x + nx * r))
        yy = int(round(y + ny * r))
        if xx < 0 or yy < 0 or xx >= w or yy >= h or mask[yy, xx] == 0:
            a = r - 1
            break
        a = r
    b = 0
    for r in range(1, max_r + 1):
        xx = int(round(x - nx * r))
        yy = int(round(y - ny * r))
        if xx < 0 or yy < 0 or xx >= w or yy >= h or mask[yy, xx] == 0:
            b = r - 1
            break
        b = r
    return 0.5 * float(a + b)


def _strip_widening_tip(
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
) -> tuple[np.ndarray, str]:
    """Cut mask past the first sudden tip widening (road / hedge lobe)."""
    if mask.max() == 0 or hub_xy is None:
        return mask, "tip_strip=skip"
    traced = _trace_dominant_arm(mask, hub_xy)
    if traced is None:
        return mask, "tip_strip=no_arm"
    dx, dy, samples = traced
    # Re-measure widths with perpendicular scan (more stable than DT alone)
    widths = [
        _perpendicular_half_width(mask, x, y, dx, dy) for x, y, _ in samples
    ]
    n = len(widths)
    if n < 30:
        return mask, "tip_strip=short"

    # Look in the outer half of the arm for a jump after a tapering trend
    start = n // 2
    cut_idx: int | None = None
    for i in range(start + 5, n - 2):
        prev = float(np.median(widths[max(start, i - 8) : i]))
        cur = float(np.median(widths[i : min(n, i + 4)]))
        # sudden widen after already relatively thin → road/hedge
        if prev > 1.5 and cur > prev * 1.55 and cur - prev > 4.0:
            cut_idx = i
            break
        # also: after clear taper, width rises again
        early = float(np.median(widths[start : start + max(5, (i - start) // 3)]))
        if early > cur * 1.25 and cur > 0 and i > start + 10:
            later_peak = float(np.max(widths[i:]))
            if later_peak > cur * 1.7 and later_peak - cur > 5.0:
                # find where it starts rising
                for j in range(i, n - 1):
                    if widths[j + 1] > widths[j] * 1.2 + 2.0:
                        cut_idx = j
                        break
                if cut_idx is not None:
                    break

    if cut_idx is None:
        return mask, "tip_strip=none"

    # Keep only the hub-connected part before the cut plane
    cx, cy, _ = samples[cut_idx]
    out = mask.copy()
    # Zero everything beyond the cut along the arm direction (dot > 0 from cut)
    ys, xs = np.where(out > 0)
    if len(xs) == 0:
        return mask, "tip_strip=empty"
    proj = (xs - cx) * dx + (ys - cy) * dy
    # also require not too far sideways past a wide lobe
    out[ys[proj > 2.0], xs[proj > 2.0]] = 0
    # Re-keep only hub-connected
    hub, _ = _hub_from_distance(out)
    if hub is None:
        hub = hub_xy
    hx = int(np.clip(hub[0], 0, out.shape[1] - 1))
    hy = int(np.clip(hub[1], 0, out.shape[0] - 1))
    if out[hy, hx] == 0:
        return mask, "tip_strip=hub_lost"
    flood = out.copy()
    ff = np.zeros((out.shape[0] + 2, out.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (hx, hy), 128, flags=4 | (255 << 8))
    kept = np.where(flood == 128, 255, 0).astype(np.uint8)
    dropped = int((mask > 0).sum() - (kept > 0).sum())
    return kept, f"tip_strip=cut@{cut_idx}/{n}+drop={dropped}"


def _second_geometry_trim(
    seed: np.ndarray,
    grown: np.ndarray,
    hub_xy: tuple[int, int] | None,
    *,
    rim_px: float = 22.0,
) -> tuple[np.ndarray, str, dict]:
    """Pass-2 geometry: keep seed; only keep white-grow that looks like a rim / arm / tip.

    Compares geometric seed vs white-grown contour. Obvious lateral expansion
    (road / roofs welded by dmax) is cut; a narrow paint rim and a tip corridor
    along the seed centerline are kept.

    Aggressive when white-grow collapses elongation (059-class). Light when
    grow keeps a multi-arm / star shape (0004-class tower→blades).
    """
    h, w = seed.shape[:2]
    seed_u = (seed > 0).astype(np.uint8) * 255
    grown_u = (grown > 0).astype(np.uint8) * 255
    empty = {
        "kept_added": np.zeros((h, w), np.uint8),
        "cut_added": np.zeros((h, w), np.uint8),
        "arm_samples": [],
        "tip_end": None,
        "tip_dir": None,
        "allow_pts": [],
        "mode": "skip",
    }
    if seed_u.max() == 0 or hub_xy is None:
        return grown_u, "geo2=skip", empty

    added = (grown_u > 0) & (seed_u == 0)
    if not np.any(added):
        empty["mode"] = "no_added"
        return seed_u.copy(), "geo2=no_added", empty

    inv = np.where(seed_u > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    rim = added & (dist <= rim_px)
    keep_added = rim.copy()

    seed_e = _elongation(seed_u)
    grown_e = _elongation(grown_u)
    collapsed = seed_e >= 4.0 and grown_e < max(2.5, 0.40 * seed_e)

    arm_samples: list[tuple[int, int]] = []
    tip_end: tuple[int, int] | None = None
    tip_dir: tuple[float, float] | None = None
    allow_pts: list[tuple[int, int]] = []

    traced = _trace_dominant_arm(seed_u, hub_xy)
    # Far white-grow past the seed tip → road (059). Far grow around hub → other
    # blades from a tower seed (0004). Only the tip-side case goes aggressive.
    mode = "light"
    if collapsed and traced is not None:
        dx0, dy0, samples0 = traced
        hx0, hy0 = hub_xy
        sx0, sy0, _ = samples0[-1]
        tip_len0 = float(np.hypot(float(sx0 - hx0), float(sy0 - hy0)))
        far = added & (dist >= max(48.0, rim_px * 2.0))
        fys, fxs = np.where(far)
        if len(fxs) >= 400:
            fproj = (fxs.astype(np.float64) - hx0) * dx0 + (
                fys.astype(np.float64) - hy0
            ) * dy0
            tip_side = int((fproj > tip_len0 * 0.62).sum())
            if tip_side >= int(0.55 * len(fxs)):
                mode = "aggressive"

    if traced is None:
        # No arm ruler → cannot judge lobes; keep white-grow as-is.
        keep_added = added.copy()
        mode = "light"
    elif traced is not None:
        dx, dy, samples = traced
        tip_dir = (dx, dy)
        arm_samples = [(int(x), int(y)) for x, y, _ in samples[:: max(1, len(samples) // 60)]]
        widths = [
            _perpendicular_half_width(seed_u, x, y, dx, dy) for x, y, _ in samples
        ]
        n = len(samples)
        tip_w = float(np.median(widths[-max(4, n // 8) :]))
        sx, sy, _ = samples[-1]
        tip_end = (int(sx), int(sy))
        hx, hy = hub_xy

        # Approximate arc-length index along hub→tip ray (samples are 1px steps).
        ys, xs = np.where(added)
        if len(xs) > 0:
            proj = (xs.astype(np.float64) - hx) * dx + (ys.astype(np.float64) - hy) * dy
            perp = (xs.astype(np.float64) - hx) * (-dy) + (ys.astype(np.float64) - hy) * dx
            tip_len = float(
                np.hypot(float(sx - hx), float(sy - hy))
            )
            # Map proj → width sample index
            idx = np.clip(np.rint(proj).astype(np.int32), 0, n - 1)
            seed_w_at = np.asarray(widths, dtype=np.float64)[idx]

            if mode == "aggressive":
                # Body: stay inside seed envelope (+ small pad). Tip corridor beyond.
                allow_body = seed_w_at * 1.40 + 10.0
                max_ext = float(min(220.0, max(48.0, tip_w * 12.0)))
                beyond = proj - tip_len
                in_body = (proj >= 0.0) & (beyond <= 2.0)
                in_tip = (beyond > 0.0) & (beyond <= max_ext)
                shrink = np.clip(1.0 - 0.65 * beyond / max(max_ext, 1.0), 0.25, 1.15)
                allow_tip = np.maximum(2.0, tip_w * 1.20 * shrink)
                ok = (in_body & (np.abs(perp) <= allow_body)) | (
                    in_tip & (np.abs(perp) <= allow_tip)
                )
                keep_added[ys[ok], xs[ok]] = True

                # Helper dots: envelope along tip corridor
                for step in range(0, int(max_ext) + 1, 8):
                    aw = max(2.0, tip_w * 1.20 * (1.0 - 0.65 * step / max(max_ext, 1.0)))
                    cx = int(round(sx + dx * step))
                    cy = int(round(sy + dy * step))
                    nx, ny = -dy, dx
                    for sign in (-1.0, 1.0):
                        allow_pts.append(
                            (int(round(cx + sign * nx * aw)), int(round(cy + sign * ny * aw)))
                        )
            else:
                # Light / 0004-class: do not reshape white-grow (tower→blades).
                # Only draw tip helpers so 1.1c still shows the second-geometry rulers.
                keep_added = added.copy()
                max_ext = float(min(120.0, max(24.0, tip_w * 6.0)))
                for step in range(0, int(max_ext) + 1, 8):
                    cx = int(round(sx + dx * step))
                    cy = int(round(sy + dy * step))
                    nx, ny = -dy, dx
                    aw = tip_w * 1.35 + 8.0
                    for sign in (-1.0, 1.0):
                        allow_pts.append(
                            (int(round(cx + sign * nx * aw)), int(round(cy + sign * ny * aw)))
                        )

    # Always keep the paint rim even if arm math missed it
    keep_added = keep_added | rim

    if mode == "light":
        # Preserve white-grow connectivity (0004 multi-blade).
        cleaned = grown_u.copy()
        cut_added = np.zeros((h, w), dtype=bool)
        keep_added = added.copy()
        note = (
            f"geo2={mode}+cut=0+keep_add={int(keep_added.sum())}"
            f"+rim={int(rim.sum())}+se={seed_e:.1f}+ge={grown_e:.1f}"
        )
        dbg = {
            "kept_added": (keep_added.astype(np.uint8) * 255),
            "cut_added": np.zeros((h, w), np.uint8),
            "arm_samples": arm_samples,
            "tip_end": tip_end,
            "tip_dir": tip_dir,
            "allow_pts": allow_pts,
            "mode": mode,
        }
        return cleaned, note, dbg

    cut_added = added & (~keep_added)

    cleaned = np.where(seed_u > 0, 255, 0).astype(np.uint8)
    cleaned[keep_added] = 255

    # Drop islands not connected to hub
    hx = int(np.clip(hub_xy[0], 0, w - 1))
    hy = int(np.clip(hub_xy[1], 0, h - 1))
    if cleaned[hy, hx] == 0:
        ys, xs = np.where(cleaned > 0)
        if len(xs):
            j = int(np.argmin((xs - hx) ** 2 + (ys - hy) ** 2))
            hx, hy = int(xs[j]), int(ys[j])
    if cleaned[hy, hx] > 0:
        flood = cleaned.copy()
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff, (hx, hy), 128, flags=8 | (255 << 8))
        cleaned = np.where(flood == 128, 255, 0).astype(np.uint8)
        # anything we intended to keep but lost to flood stays cut
        keep_added = keep_added & (cleaned > 0)
        cut_added = added & (cleaned == 0)

    note = (
        f"geo2={mode}+cut={int(cut_added.sum())}+keep_add={int(keep_added.sum())}"
        f"+rim={int(rim.sum())}+se={seed_e:.1f}+ge={grown_e:.1f}"
    )
    dbg = {
        "kept_added": (keep_added.astype(np.uint8) * 255),
        "cut_added": (cut_added.astype(np.uint8) * 255),
        "arm_samples": arm_samples,
        "tip_end": tip_end,
        "tip_dir": tip_dir,
        "allow_pts": allow_pts,
        "mode": mode,
    }
    return cleaned, note, dbg


def _judge_tip_needed(
    bgr: np.ndarray,
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
) -> tuple[bool, str, dict]:
    """Decide whether tip completion should run.

    Most near shots already cover enough tip for stage-1 features. Only run
    tip edge / taper when the arm is single-blade-like, the tip is blunt, and
    a modest paint-like gap remains ahead of the mouth.
    """
    h, w = mask.shape[:2]
    dbg = {
        "tip_end": None,
        "tip_dir": None,
        "tip_w": 0.0,
        "mid_w": 0.0,
        "elong": 1.0,
        "paint_gap": 0,
    }
    if mask.max() == 0 or hub_xy is None:
        return False, "tip_need=skip_empty", dbg

    elong = _elongation(mask)
    dbg["elong"] = elong
    if elong < 4.5:
        return False, "tip_need=skip_stubby", dbg

    traced = _trace_dominant_arm(mask, hub_xy)
    if traced is None:
        return False, "tip_need=skip_no_arm", dbg
    dx, dy, samples = traced
    n = len(samples)
    if n < 40:
        return False, "tip_need=skip_short", dbg

    tip_seg = samples[max(0, n - max(40, n // 8)) :]
    p0 = np.array(tip_seg[0][:2], dtype=np.float64)
    p1 = np.array(tip_seg[-1][:2], dtype=np.float64)
    d = p1 - p0
    nrm = float(np.hypot(d[0], d[1])) + 1e-6
    dx, dy = float(d[0] / nrm), float(d[1] / nrm)
    widths = [
        _perpendicular_half_width(mask, x, y, dx, dy) for x, y, _ in samples
    ]
    tip_w = float(np.median(widths[-max(4, n // 8) :]))
    mid_w = float(np.median(widths[n // 2 : 3 * n // 4]))
    tx, ty, _ = samples[-1]
    dbg.update(
        {
            "tip_end": (int(tx), int(ty)),
            "tip_dir": (dx, dy),
            "tip_w": tip_w,
            "mid_w": mid_w,
        }
    )

    margin = 4
    if tx <= margin or ty <= margin or tx >= w - 1 - margin or ty >= h - 1 - margin:
        return False, "tip_need=skip_at_border", dbg

    last = np.asarray(widths[-max(12, n // 12) :], dtype=np.float64)
    # already converging hard to a point
    if tip_w <= 5.0 or (float(last[0]) > tip_w * 1.45 and tip_w <= 10.0):
        return False, "tip_need=skip_tapered", dbg

    # Close-up fat tip: mask already owns a thick tip section — leave alone (0190).
    if tip_w >= 28.0:
        return False, f"tip_need=skip_fat_tip+tw={tip_w:.1f}", dbg

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nx, ny = -dy, dx
    paint_gap = 0
    for step in range(1, 100):
        cx = int(round(tx + dx * step))
        cy = int(round(ty + dy * step))
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            break
        if mask[cy, cx] > 0:
            continue
        half = max(1.5, tip_w * 0.35)
        band = 0
        paint = 0
        for t in range(-int(half) - 1, int(half) + 2):
            xx = int(round(cx + nx * t))
            yy = int(round(cy + ny * t))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                continue
            if abs(t) > half:
                continue
            band += 1
            if mask[yy, xx] == 0 and float(gray[yy, xx]) >= 100.0:
                paint += 1
        if band > 0 and paint >= max(2, int(0.35 * band)):
            paint_gap += 1
    dbg["paint_gap"] = int(paint_gap)

    if paint_gap < 12:
        return False, f"tip_need=skip_no_gap+gap={paint_gap}", dbg

    blunt = tip_w >= mid_w * 0.55 or float(np.std(last)) <= max(1.5, tip_w * 0.12)
    if blunt and paint_gap >= 12:
        return True, (
            f"tip_need=yes+blunt_gap+tw={tip_w:.1f}+mw={mid_w:.1f}+gap={paint_gap}"
        ), dbg

    return False, f"tip_need=skip_other+tw={tip_w:.1f}+gap={paint_gap}", dbg


def _tip_roi_polygon(
    tip_xy: tuple[int, int],
    dx: float,
    dy: float,
    tip_w: float,
    *,
    max_fwd: int = 170,
    back: int = 12,
) -> np.ndarray:
    """Trapezoid ROI ahead of the current mask tip (along centerline)."""
    nx, ny = -dy, dx
    half0 = max(8.0, tip_w * 2.3)
    half1 = max(3.0, tip_w * 0.45)
    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    pts = [
        (tx - dx * back - nx * half0, ty - dy * back - ny * half0),
        (tx - dx * back + nx * half0, ty - dy * back + ny * half0),
        (tx + dx * max_fwd + nx * half1, ty + dy * max_fwd + ny * half1),
        (tx + dx * max_fwd - nx * half1, ty + dy * max_fwd - ny * half1),
    ]
    return np.array(pts, dtype=np.float32)


def _mouth_edge_points(
    mask: np.ndarray,
    tip_xy: tuple[int, int],
    dx: float,
    dy: float,
    tip_w: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Left/right mask edge at the tip cross-section (along normal)."""
    h, w = mask.shape[:2]
    nx, ny = -dy, dx
    tx, ty = tip_xy
    hits: list[tuple[float, int, int]] = []
    max_r = int(max(6, tip_w * 2.5 + 8))
    for t in range(-max_r, max_r + 1):
        x = int(round(tx + nx * t))
        y = int(round(ty + ny * t))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            hits.append((float(t), x, y))
    if len(hits) < 4:
        return None
    hits.sort(key=lambda z: z[0])
    return (hits[0][1], hits[0][2]), (hits[-1][1], hits[-1][2])


def _follow_tip_edge(
    grad: np.ndarray,
    edges: np.ndarray,
    roi: np.ndarray,
    start: tuple[int, int],
    dx: float,
    dy: float,
    side_sign: float,
    *,
    max_steps: int = 150,
    inward: float = 0.12,
    edge_only: bool = False,
    gap: int = 0,
) -> list[tuple[int, int]]:
    """Greedy walk on Canny edges from mouth point toward the tip.

    ``side_sign``: -1 left / +1 right along the outward normal.
    Default (sunny tip fill): slight inward bias so the two sides can meet,
    and a strong gradient may count when Canny drops out.

    ``edge_only`` (1.3c trap rim): stay on Canny. A same-pixel sample is a
    stall, not the end — take the next distinct edge pixel. A short ``gap``
    may bridge a break, but a large sideways jump stops the walk.
    """
    h, w = grad.shape[:2]
    nx, ny = -dy, dx
    ox, oy = side_sign * nx, side_sign * ny
    x, y = float(start[0]), float(start[1])
    pts: list[tuple[int, int]] = [(int(start[0]), int(start[1]))]
    if not edge_only:
        miss = 0
        for _ in range(max_steps):
            best = None
            best_score = -1e9
            for ahead in (1, 2, 3, 4):
                for lat in range(-4, 5):
                    px = x + dx * ahead + ox * (lat * 0.7) - ox * (inward * ahead)
                    py = y + dy * ahead + oy * (lat * 0.7) - oy * (inward * ahead)
                    ix, iy = int(round(px)), int(round(py))
                    if ix < 1 or iy < 1 or ix >= w - 1 or iy >= h - 1:
                        continue
                    if roi[iy, ix] == 0:
                        continue
                    on_edge = edges[iy, ix] > 0
                    g = float(grad[iy, ix])
                    if not on_edge and g < 25.0:
                        continue
                    score = g + (40.0 if on_edge else 0.0) - abs(lat) * 3.5 - ahead * 0.5
                    if score > best_score:
                        best_score = score
                        best = (px, py, ix, iy)
            if best is None:
                miss += 1
                if miss >= 3:
                    break
                x += dx
                y += dy
                continue
            miss = 0
            x, y, ix, iy = best
            if pts and (ix, iy) == pts[-1]:
                break
            pts.append((ix, iy))
        return pts

    axis = np.array([dx, dy], dtype=np.float64)
    normal = np.array([-dy, dx], dtype=np.float64)
    bridges = 0

    def _canny_steps(aheads, lats, lat_scale, lat_pen, ahead_pen, recent: int):
        found: list[tuple[float, float, float, int, int]] = []
        blocked = set(pts[-recent:]) if recent else set()
        for ahead in aheads:
            for lat in lats:
                px = x + dx * ahead + ox * (lat * lat_scale)
                py = y + dy * ahead + oy * (lat * lat_scale)
                ix, iy = int(round(px)), int(round(py))
                if ix < 1 or iy < 1 or ix >= w - 1 or iy >= h - 1:
                    continue
                if roi[iy, ix] == 0 or edges[iy, ix] == 0:
                    continue
                if (ix, iy) in blocked:
                    continue
                score = 100.0 - abs(lat) * lat_pen - ahead * ahead_pen
                found.append((score, px, py, ix, iy))
        return found

    for _ in range(max_steps):
        found = _canny_steps(range(1, 5), range(-6, 7), 0.9, 3.2, 0.8, 1)
        bridged = False
        if not found and gap > 0 and bridges < 3:
            found = _canny_steps(range(5, gap + 1), range(-7, 8), 0.9, 4.0, 1.5, 4)
            bridged = bool(found)
        if not found:
            break
        _score, px, py, ix, iy = max(found)
        prev = np.array(pts[-1], dtype=np.float64)
        step = np.array([ix, iy], dtype=np.float64) - prev
        if float(np.dot(step, axis)) < 0.25:
            break
        if abs(float(np.dot(step, normal))) > 14.0:
            break
        if bridged:
            bridges += 1
        pts.append((ix, iy))
        x, y = px, py
    return pts


def _complete_tip_by_edges(
    bgr: np.ndarray,
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
    *,
    max_fwd: int = 170,
) -> tuple[np.ndarray, str, dict]:
    """Crop tip ROI → Canny → follow left/right edges → fill taper polygon."""
    h, w = mask.shape[:2]
    empty = {
        "roi_mask": np.zeros((h, w), np.uint8),
        "edges": np.zeros((h, w), np.uint8),
        "left_poly": [],
        "right_poly": [],
        "tip_point": None,
        "tip_fill": np.zeros((h, w), np.uint8),
        "tip_end": None,
        "tip_dir": None,
    }
    if mask.max() == 0 or hub_xy is None:
        return mask, "tip_edge=skip", empty

    traced = _trace_dominant_arm(mask, hub_xy)
    if traced is None:
        return mask, "tip_edge=no_arm", empty
    dx, dy, samples = traced
    n = len(samples)
    if n < 40:
        return mask, "tip_edge=short", empty

    widths = [
        _perpendicular_half_width(mask, x, y, dx, dy) for x, y, _ in samples
    ]
    tip_w = float(np.median(widths[-max(4, n // 8) :]))
    # tip direction from last stretch
    tip_seg = samples[max(0, n - max(40, n // 8)) :]
    p0 = np.array(tip_seg[0][:2], dtype=np.float64)
    p1 = np.array(tip_seg[-1][:2], dtype=np.float64)
    d = p1 - p0
    nrm = float(np.hypot(d[0], d[1])) + 1e-6
    dx, dy = float(d[0] / nrm), float(d[1] / nrm)
    tx, ty, _ = samples[-1]
    tip_xy = (int(tx), int(ty))

    # already at border → nothing to complete
    if tx <= 4 or ty <= 4 or tx >= w - 5 or ty >= h - 5:
        empty["tip_end"] = tip_xy
        empty["tip_dir"] = (dx, dy)
        return mask, "tip_edge=at_border", empty

    mouth = _mouth_edge_points(mask, tip_xy, dx, dy, tip_w)
    if mouth is None:
        return mask, "tip_edge=no_mouth", empty
    left0, right0 = mouth

    poly = _tip_roi_polygon(tip_xy, dx, dy, tip_w, max_fwd=max_fwd)
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.round(poly).astype(np.int32)], 255)
    # don't search inside already-masked blade body (keep a thin mouth band)
    body = mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_erode = cv2.erode(body, k, iterations=1)
    search = cv2.bitwise_and(roi, cv2.bitwise_not(body_erode))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    # suppress vegetation-ish strong green texture a bit via low V / high S? keep simple:
    # mask grad outside search ROI
    grad_roi = grad.copy()
    grad_roi[search == 0] = 0
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.bitwise_and(edges, search)

    left = _follow_tip_edge(grad_roi, edges, search, left0, dx, dy, -1.0, max_steps=max_fwd)
    right = _follow_tip_edge(grad_roi, edges, search, right0, dx, dy, 1.0, max_steps=max_fwd)

    tip_point: tuple[int, int] | None = None
    tip_fill = np.zeros((h, w), np.uint8)
    added = 0
    stop_reason = "open"
    max_add = int(max(800, tip_w * tip_w * 18))

    def _commit_tip_poly(
        chain: list[tuple[int, int]], reason: str
    ) -> tuple[np.ndarray, int, str]:
        if len(chain) < 3:
            return np.zeros((h, w), np.uint8), 0, "open"
        fill = np.zeros((h, w), np.uint8)
        cv2.fillPoly(fill, [np.array(chain, dtype=np.int32)], 255)
        fill = cv2.bitwise_and(fill, roi)
        new_px = (fill > 0) & (mask == 0)
        n_add = int(new_px.sum())
        if n_add > max_add:
            return np.zeros((h, w), np.uint8), 0, "reject_huge"
        if n_add <= 0:
            return fill, 0, reason
        fill = cv2.morphologyEx(
            fill,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        return fill, n_add, reason

    def _mirror_across_tip_axis(
        pts: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Reflect rim across tip centerline — phantom opposite for one-rim seal."""
        o = np.array(tip_xy, dtype=np.float64)
        d = np.array([dx, dy], dtype=np.float64)
        out: list[tuple[int, int]] = []
        for x, y in pts:
            v = np.array([x, y], dtype=np.float64) - o
            proj = o + d * float(np.dot(v, d))
            m = 2.0 * proj - np.array([x, y], dtype=np.float64)
            out.append((int(round(m[0])), int(round(m[1]))))
        return out

    # Walk in lockstep distance: stop when left/right get close or hit road-like wall
    min_len = min(len(left), len(right))
    if min_len >= 8:
        cut = min_len
        for i in range(3, min_len):
            lx, ly = left[i]
            rx, ry = right[i]
            sep = float(np.hypot(lx - rx, ly - ry))
            # road wall: many edge pixels across a short forward band, nearly ⊥ to tip
            cx = int(round(0.5 * (lx + rx)))
            cy = int(round(0.5 * (ly + ry)))
            # sample transverse edge density
            nx, ny = -dy, dx
            edge_hits = 0
            probes = 0
            for t in np.linspace(-sep * 0.6, sep * 0.6, 13):
                xx = int(round(cx + nx * t))
                yy = int(round(cy + ny * t))
                probes += 1
                if 0 <= xx < w and 0 <= yy < h and edges[yy, xx] > 0:
                    edge_hits += 1
            # strong transverse bar spanning the mouth → road/hedge, stop before
            if sep > tip_w * 1.8 and edge_hits >= max(6, int(0.55 * probes)):
                cut = max(3, i - 1)
                stop_reason = "road_wall"
                break
            if sep <= max(3.0, tip_w * 0.35):
                tip_point = (int(round(0.5 * (lx + rx))), int(round(0.5 * (ly + ry))))
                cut = i + 1
                stop_reason = "meet"
                break
        left = left[:cut]
        right = right[:cut]
        if tip_point is None and len(left) >= 4:
            tip_point = (
                int(round(0.5 * (left[-1][0] + right[-1][0]))),
                int(round(0.5 * (left[-1][1] + right[-1][1]))),
            )
            stop_reason = "end"

        # fill polygon: mouth left → left chain → tip → right chain reversed → mouth right
        chain: list[tuple[int, int]] = list(left)
        if tip_point is not None:
            chain.append(tip_point)
        chain.extend(reversed(right))
        tip_fill, added, stop_reason = _commit_tip_poly(chain, stop_reason)

    # 0059-class: one strong rim (orange) + axis mirror / weak stub — seal inside yellow ROI
    if added == 0 and max(len(left), len(right)) >= 8:
        if len(left) >= len(right):
            strong, weak, weak_mouth = left, right, right0
            seal_tag = "one_rim_L"
        else:
            strong, weak, weak_mouth = right, left, left0
            seal_tag = "one_rim_R"
        phantom = _mirror_across_tip_axis(strong)
        tip_point = (
            int(round(0.5 * (strong[-1][0] + phantom[-1][0]))),
            int(round(0.5 * (strong[-1][1] + phantom[-1][1]))),
        )
        # Prefer real short opposite stub near mouth when present
        if len(weak) >= 2:
            chain = list(strong) + [tip_point] + list(reversed(weak))
        else:
            chain = list(strong) + [tip_point] + list(reversed(phantom))
            chain.append(weak_mouth)
        tip_fill, added, stop_reason = _commit_tip_poly(chain, seal_tag)

    out = mask.copy()
    if added > 0:
        out = cv2.bitwise_or(out, tip_fill)
        # keep hub-connected
        hx = int(np.clip(hub_xy[0], 0, w - 1))
        hy = int(np.clip(hub_xy[1], 0, h - 1))
        if out[hy, hx] > 0:
            flood = out.copy()
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ff, (hx, hy), 128, flags=8 | (255 << 8))
            out = np.where(flood == 128, 255, 0).astype(np.uint8)

    note = (
        f"tip_edge={stop_reason}+add={added}+L={len(left)}+R={len(right)}"
        f"+tw={tip_w:.1f}"
    )
    dbg = {
        "roi_mask": roi,
        "edges": edges,
        "left_poly": left,
        "right_poly": right,
        "tip_point": tip_point,
        "tip_fill": tip_fill,
        "tip_end": tip_xy,
        "tip_dir": (dx, dy),
    }
    return out, note, dbg


def _complete_tip_taper(
    bgr: np.ndarray,
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
    *,
    max_extend: int = 200,
) -> tuple[np.ndarray, str]:
    """Blade tip rule (kept deliberately simple):

    1. Trace one centerline from hub along the longest arm.
    2. Width along that line may only stay flat or shrink — never widen.
       A sudden widen at the end is road/hedge → cut it off.
    3. If the mask tip is *not* cut by the photo border, a real blade tip
       must still taper to a point → extend along the line inside a
       shrinking envelope until paint runs out or we hit the border.
    """
    if mask.max() == 0 or hub_xy is None:
        return mask, "tip=skip"

    # --- 1) line + widths ---
    stripped, strip_note = _strip_widening_tip(mask, hub_xy)
    traced = _trace_dominant_arm(stripped, hub_xy)
    if traced is None:
        return stripped, f"{strip_note}|tip=no_arm"
    dx, dy, samples = traced
    widths = [
        _perpendicular_half_width(stripped, x, y, dx, dy) for x, y, _ in samples
    ]
    n = len(widths)
    if n < 40:
        return stripped, f"{strip_note}|tip=short"

    elong = _elongation(stripped)
    if elong < 4.5:
        # whole turbine / stubby — not a single blade chin-to-tip case
        return stripped, f"{strip_note}|tip=skip_stubby"

    outer = widths[n // 2 :]
    root_w = float(np.median(outer[: max(4, len(outer) // 4)]))
    tip_w = float(np.median(outer[-max(4, len(outer) // 8) :]))
    tapering = tip_w <= root_w * 1.05
    if not tapering:
        return stripped, f"{strip_note}|tip=not_tapering"

    # Centerline direction from the tip segment (not the whole hub chord).
    # A long arm can curve a few degrees; the last stretch is the real trend.
    tip_seg = samples[max(0, n - max(40, n // 8)) :]
    p0 = np.array(tip_seg[0][:2], dtype=np.float64)
    p1 = np.array(tip_seg[-1][:2], dtype=np.float64)
    d = p1 - p0
    nrm = float(np.hypot(d[0], d[1])) + 1e-6
    dx, dy = float(d[0] / nrm), float(d[1] / nrm)

    # --- 2) tip already at photo border? then no forced point ---
    tx, ty, _ = samples[-1]
    h, w = stripped.shape[:2]
    margin = 4
    at_border = (
        tx <= margin or ty <= margin or tx >= w - 1 - margin or ty >= h - 1 - margin
    )
    if at_border:
        return stripped, f"{strip_note}|tip=at_border"

    # --- 3) must finish to a point: shrink envelope along the same line ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _hh, ss, vv = cv2.split(hsv)
    # paint-like enough to be blade (allow field tint); reject strong green only
    strong_green = (_hh >= 35) & (_hh <= 85) & (ss >= 55) & (vv < 150)
    arm_v = float(np.median([float(gray[y, x]) for x, y, _ in samples[n * 2 // 3 :]]))
    v_floor = max(75.0, arm_v * 0.55)
    paint = ((vv >= 95) & (ss <= 110) & (~strong_green)) | (
        (gray.astype(np.float32) >= v_floor) & (~strong_green) & (ss <= 90)
    )

    allow_w = max(2.0, tip_w * 0.9)
    # how far until envelope reaches ~1px (geometric tip length estimate)
    if allow_w <= 1.5:
        need = 20
    else:
        need = int(min(max_extend, max(24, allow_w * 8)))

    out = stripped.copy()
    nx, ny = -dy, dx
    nrm = float(np.hypot(nx, ny)) + 1e-6
    nx, ny = nx / nrm, ny / nrm
    added = 0
    last_w = allow_w
    miss = 0

    for step in range(1, max_extend + 1):
        cx = int(round(tx + dx * step))
        cy = int(round(ty + dy * step))
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            break  # photo cut — stop, tip may be off-frame

        # monotone shrink: never wider than previous station
        cur_w = max(1.0, allow_w * (1.0 - step / float(need)))
        if cur_w > last_w * 1.02:
            cur_w = last_w
        last_w = cur_w

        # measured width of paint across the line — must not jump up
        half_probe = int(np.ceil(max(cur_w * 3.0, 8)))
        hit_ts = [
            t
            for t in range(-half_probe, half_probe + 1)
            if 0 <= int(round(cx + nx * t)) < w
            and 0 <= int(round(cy + ny * t)) < h
            and paint[int(round(cy + ny * t)), int(round(cx + nx * t))]
            and float(gray[int(round(cy + ny * t)), int(round(cx + nx * t))]) >= v_floor
        ]
        if hit_ts:
            meas_w = 0.5 * float(max(hit_ts) - min(hit_ts))
            if meas_w > last_w * 1.45 + 3.0 and step > 3:
                # suddenly wider than the taper allows → road / not blade tip
                break

        half = int(np.ceil(cur_w + 1.0))
        placed = 0
        for t in range(-half, half + 1):
            if abs(t) > cur_w + 0.6:
                continue
            xx = int(round(cx + nx * t))
            yy = int(round(cy + ny * t))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                continue
            if not paint[yy, xx]:
                continue
            if float(gray[yy, xx]) < v_floor:
                continue
            if out[yy, xx] == 0:
                out[yy, xx] = 255
                added += 1
                placed += 1
        if placed >= 1:
            miss = 0
        else:
            miss += 1
            if miss >= 5:
                break
        if cur_w <= 1.2 and miss >= 2:
            break

    if added > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tip_zone = np.zeros_like(out)
        for step in range(0, max_extend + 10):
            cx = int(round(tx + dx * step))
            cy = int(round(ty + dy * step))
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                break
            cv2.circle(tip_zone, (cx, cy), int(max(2, tip_w + 3)), 255, -1)
        closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
        out = np.where(tip_zone > 0, closed, out).astype(np.uint8)
        out = cv2.bitwise_or(out, stripped)
        hub, _ = _hub_from_distance(stripped)
        if hub is None:
            hub = hub_xy
        hx = int(np.clip(hub[0], 0, w - 1))
        hy = int(np.clip(hub[1], 0, h - 1))
        if out[hy, hx] > 0:
            flood = out.copy()
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ff, (hx, hy), 128, flags=4 | (255 << 8))
            out = np.where(flood == 128, 255, 0).astype(np.uint8)

    note = (
        f"{strip_note}|tip=add={added}+tw={tip_w:.1f}+rw={root_w:.1f}"
        f"+elong={elong:.1f}"
    )
    return out, note


# ---------------------------------------------------------------------------
# 1.3 expand lines (3a–3d): draw-only geometry + outer contour pick
# ---------------------------------------------------------------------------


def _mask_principal_axis(
    mask: np.ndarray,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, float, float]:
    """3a: PCA major axis of mask (no hub).

    Returns origin, unit dir, half-length (along), half-width (across).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None, None, 0.0, 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eig = cv2.PCACompute(pts, mean=None)
    origin = (float(mean[0, 0]), float(mean[0, 1]))
    # eig rows are eigenvectors; largest variance first
    d = eig[0].astype(np.float64)
    n = float(np.linalg.norm(d)) + 1e-9
    d = d / n
    proj = (pts - mean) @ eig.T
    half = float(max(abs(proj[:, 0].min()), abs(proj[:, 0].max())))
    half_w = float(max(abs(proj[:, 1].min()), abs(proj[:, 1].max())))
    return origin, (float(d[0]), float(d[1])), half, half_w


def _outer_ring(mask: np.ndarray, *, inner: int = 6, outer: int = 40) -> np.ndarray:
    """Pixels outside mask between ``inner`` and ``outer`` dilation radii."""
    if mask.max() == 0:
        return np.zeros_like(mask)
    k_in = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inner * 2 + 1, inner * 2 + 1)
    )
    k_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer * 2 + 1, outer * 2 + 1)
    )
    near = cv2.dilate(mask, k_in)
    far = cv2.dilate(mask, k_out)
    return ((far > 0) & (near == 0)).astype(np.uint8) * 255


def _detect_hough_segments(
    gray: np.ndarray,
    *,
    min_len: int = 36,
    max_gap: int = 14,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=35,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if lines is None:
        return out
    # OpenCV 4: (N,1,4); OpenCV 5: (N,4)
    arr = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in arr:
        out.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return out


def _seg_geometry(
    p1: tuple[int, int], p2: tuple[int, int]
) -> tuple[np.ndarray, float, np.ndarray]:
    a = np.array([p1[0], p1[1]], dtype=np.float64)
    b = np.array([p2[0], p2[1]], dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d)) + 1e-9
    direction = d / length
    mid = 0.5 * (a + b)
    return direction, length, mid


def _sample_seg_points(
    p1: tuple[int, int], p2: tuple[int, int], *, n: int = 12
) -> list[tuple[int, int]]:
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, num=max(2, n))
    return [(int(round(x)), int(round(y))) for x, y in (a + (b - a) * t for t in ts)]


def _seg_attaches_to_subject(
    p1: tuple[int, int],
    p2: tuple[int, int],
    mask: np.ndarray,
    *,
    attach_px: int = 12,
    origin: tuple[float, float] | None = None,
    axis_dir: tuple[float, float] | None = None,
    half_len: float = 0.0,
    half_w: float = 0.0,
    ring_outer: int = 40,
) -> bool:
    """True iff the segment extends from the subject (not a floating road/water line).

    Rules:
    1. At least one *endpoint* in the attach rim (from-subject, not a far graze).
    2. Midpoint not far outside the mask.
    3. If axis known: mid must sit within blade slab (along + across) — kills
       road/water lines that are merely near a fat mask in image space.
    """
    if mask.max() == 0:
        return False
    h, w = mask.shape[:2]
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (attach_px * 2 + 1, attach_px * 2 + 1)
    )
    touch = cv2.dilate(mask, k)
    # endpoint must touch subject rim (延伸自主体)
    end_ok = False
    for x, y in (p1, p2):
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if touch[y, x] > 0 or mask[y, x] > 0:
            end_ok = True
            break
    if not end_ok:
        return False
    dist = cv2.distanceTransform(
        np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
    )
    mx = int(np.clip(0.5 * (p1[0] + p2[0]), 0, w - 1))
    my = int(np.clip(0.5 * (p1[1] + p2[1]), 0, h - 1))
    if float(dist[my, mx]) > float(attach_px + 22):
        return False
    if origin is not None and axis_dir is not None and half_w > 1.0:
        o = np.array(origin, dtype=np.float64)
        ad = np.array(axis_dir, dtype=np.float64)
        mid = np.array([mx, my], dtype=np.float64)
        along = float(np.dot(mid - o, ad))
        across = abs(float((mid - o)[0] * ad[1] - (mid - o)[1] * ad[0]))
        if abs(along) > half_len * 1.20:
            return False
        if across > half_w + float(ring_outer) + 8.0:
            return False
    return True


def _seg_closable_with_border(
    p1: tuple[int, int],
    p2: tuple[int, int],
    *,
    h: int,
    w: int,
    max_ext: float = 280.0,
    border: int = 3,
) -> bool:
    """True if extending the segment along its direction hits the photo border.

    Closure with the image edge counts (user: 可跟照片边界闭合).
    """
    direction, length, mid = _seg_geometry(p1, p2)
    if length < 1.0:
        return False

    def _hits(origin: np.ndarray, direc: np.ndarray) -> bool:
        # already near border?
        x0, y0 = int(origin[0]), int(origin[1])
        if x0 <= border or y0 <= border or x0 >= w - 1 - border or y0 >= h - 1 - border:
            return True
        step = 4.0
        n = int(max_ext / step)
        for i in range(1, n + 1):
            p = origin + direc * (i * step)
            x, y = int(round(p[0])), int(round(p[1]))
            if x <= border or y <= border or x >= w - 1 - border or y >= h - 1 - border:
                return True
            if x < 0 or y < 0 or x >= w or y >= h:
                return True
        return False

    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    return _hits(a, -direction) or _hits(b, direction) or _hits(mid, direction) or _hits(
        mid, -direction
    )


def _score_ring_segments(
    segs: list[tuple[tuple[int, int], tuple[int, int]]],
    *,
    mask: np.ndarray,
    ring: np.ndarray,
    axis_dir: tuple[float, float],
    origin: tuple[float, float],
    half_len: float,
    half_w: float,
    ban: np.ndarray | None,
    min_len: float = 40.0,
    min_align: float = 0.72,
    attach_px: int = 12,
    ring_outer: int = 40,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Score segments attached to subject, in outer ring, axis-aligned.

    Floating road/water lines (not extending from the mask) are dropped.
    Closable with photo border (or already long along axis) gets a score boost.
    """
    h, w = mask.shape[:2]
    dist = cv2.distanceTransform(
        np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
    )
    ad = np.array(axis_dir, dtype=np.float64)
    scored: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for p1, p2 in segs:
        direction, length, mid = _seg_geometry(p1, p2)
        if length < min_len:
            continue
        if not _seg_attaches_to_subject(
            p1,
            p2,
            mask,
            attach_px=attach_px,
            origin=origin,
            axis_dir=axis_dir,
            half_len=half_len,
            half_w=half_w,
            ring_outer=ring_outer,
        ):
            continue
        mx, my = int(np.clip(mid[0], 0, w - 1)), int(np.clip(mid[1], 0, h - 1))
        # require midpoint (or either end) in/near ring
        ends_ok = False
        for px, py in (p1, p2, (mx, my)):
            x = int(np.clip(px, 0, w - 1))
            y = int(np.clip(py, 0, h - 1))
            if ring[y, x] > 0:
                ends_ok = True
                break
        if not ends_ok:
            continue
        if ban is not None and ban[my, mx]:
            continue
        if mask[my, mx] > 0:
            continue
        align = abs(float(np.dot(direction, ad)))
        if align < min_align:
            continue
        d_out = float(dist[my, mx])
        # prefer 6–40px outside mask
        dist_term = 1.0 - abs(d_out - 18.0) / 28.0
        dist_term = float(np.clip(dist_term, 0.0, 1.0))
        score = length * (0.45 + 0.55 * align) * (0.35 + 0.65 * dist_term)
        if _seg_closable_with_border(p1, p2, h=h, w=w):
            score *= 1.35
        scored.append((p1, p2, float(score)))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored


def _side_of_axis(
    mid: np.ndarray, origin: tuple[float, float], axis_dir: tuple[float, float]
) -> float:
    """Signed side: >0 left of axis direction, <0 right (2D cross z)."""
    v = mid - np.array(origin, dtype=np.float64)
    return float(v[0] * axis_dir[1] - v[1] * axis_dir[0])


def _across_abs(
    pt: tuple[float, float] | np.ndarray,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
) -> float:
    """Absolute perpendicular distance from axis (blade half-width sense)."""
    return abs(_side_of_axis(np.asarray(pt, dtype=np.float64), origin, axis_dir))


def _chain_inward_of_outer_pick(
    chain: list[tuple[int, int]],
    outer_seg: tuple[tuple[int, int], tuple[int, int], float],
    *,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    half_w: float,
) -> bool:
    """True if trap chain sits clearly inside a 1.3a outer pick (rust seam).

    713: 1.3c green latches onto rust; 1.3a blue/green pick is the real rim.
    Same-side chain with smaller across-distance than the pick → interior.
    """
    if len(chain) < 4:
        return False
    p1, p2, _sc = outer_seg
    outer_across = 0.5 * (
        _across_abs(p1, origin, axis_dir) + _across_abs(p2, origin, axis_dir)
    )
    step = max(1, len(chain) // 12)
    samples = chain[::step] or chain
    chain_across = float(
        np.median([_across_abs(p, origin, axis_dir) for p in samples])
    )
    # Must clear a gap: interior seams sit well inside the outer rim.
    # 713 rust vs outer pick ≈15px; keep 624 trap rims (margin ~13px OK).
    gap = max(14.0, 0.10 * max(half_w, 1.0))
    return chain_across + gap < outer_across


def _pick_paired_edges(
    ring_scored: list[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    half_len: float,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], float] | None,
    tuple[tuple[int, int], tuple[int, int], float] | None,
    tuple[tuple[int, int], tuple[int, int]] | None,
    bool,
]:
    """3d: best segment each side of axis; phantom parallel if only one side."""
    left = None
    right = None
    for seg in ring_scored:
        p1, p2, sc = seg
        _, _, mid = _seg_geometry(p1, p2)
        side = _side_of_axis(mid, origin, axis_dir)
        if side >= 0:
            if left is None or sc > left[2]:
                left = seg
        else:
            if right is None or sc > right[2]:
                right = seg

    phantom = None
    paired = left is not None and right is not None
    if paired:
        return left, right, None, True

    # one-sided → parallel phantom at mirrored distance from axis
    src = left if left is not None else right
    if src is None:
        return None, None, None, False
    p1, p2, _sc = src
    _, length, mid = _seg_geometry(p1, p2)
    # distance from mid to axis line
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)
    # perpendicular
    perp = np.array([-ad[1], ad[0]], dtype=np.float64)
    side = _side_of_axis(mid, origin, axis_dir)
    # vector from origin to mid projected on perp
    offset = float(np.dot(mid - o, perp))
    if abs(offset) < 4.0:
        offset = 12.0 if side >= 0 else -12.0
    mirror_mid = o + ad * float(np.dot(mid - o, ad)) - perp * offset
    half = 0.5 * length
    # extend along axis (blade edges ≈ axis-parallel)
    a = mirror_mid - ad * half
    b = mirror_mid + ad * half
    phantom = (
        (int(round(a[0])), int(round(a[1]))),
        (int(round(b[0])), int(round(b[1]))),
    )
    # also extend real seg a bit along its own dir for viz consistency
    return left, right, phantom, False


def _expand_lines_from_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    lighting: str = "sunny",
    ring_inner: int = 6,
    ring_outer: int = 40,
) -> ExpandLinesDebug:
    """Run 3a–3d on a work-size BGR + binary mask (no fill)."""
    h, w = mask.shape[:2]
    empty = ExpandLinesDebug(
        work_bgr=bgr,
        mask=mask,
        axis_origin=None,
        axis_dir=None,
        axis_half_len=0.0,
        all_segs=[],
        ring_segs=[],
        left_seg=None,
        right_seg=None,
        phantom_seg=None,
        paired=False,
        note="expand_lines=empty_mask",
        scale=1.0,
        lighting=lighting,
        ring_inner=ring_inner,
        ring_outer=ring_outer,
    )
    if mask.max() == 0:
        return empty

    origin, axis_dir, half, half_w = _mask_principal_axis(mask)
    if origin is None or axis_dir is None:
        empty.note = "expand_lines=no_axis"
        return empty

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ring = _outer_ring(mask, inner=ring_inner, outer=ring_outer)
    all_raw = _detect_hough_segments(gray, min_len=max(28, int(0.04 * max(h, w))))
    attach_px = max(10, int(0.012 * max(h, w)))

    ban = None
    if lighting == "overcast":
        sky_cut, water_cut, _meta = _sky_water_ban_masks(bgr)
        ban = sky_cut | water_cut
        # also ban top sky strip aggressively
        ban = ban.copy()
        ban[: max(1, int(0.10 * h))] = True

    attach_kw = dict(
        attach_px=attach_px,
        origin=origin,
        axis_dir=axis_dir,
        half_len=half,
        half_w=half_w,
        ring_outer=ring_outer,
    )

    # Yellow "all": axis-ish AND attached to subject (drop floating road/water).
    ad = np.array(axis_dir, dtype=np.float64)
    all_segs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    n_raw = len(all_raw)
    n_detach = 0
    for p1, p2 in all_raw:
        direction, length, mid = _seg_geometry(p1, p2)
        align = abs(float(np.dot(direction, ad)))
        if length < 28 or align < 0.55:
            continue
        if not _seg_attaches_to_subject(p1, p2, mask, **attach_kw):
            n_detach += 1
            continue
        mx, my = int(np.clip(mid[0], 0, w - 1)), int(np.clip(mid[1], 0, h - 1))
        if ban is not None and ban[my, mx]:
            continue
        closable = _seg_closable_with_border(p1, p2, h=h, w=w)
        score = length * align * (1.35 if closable else 1.0)
        all_segs.append((p1, p2, score))
    all_segs.sort(key=lambda t: t[2], reverse=True)
    all_segs = all_segs[:80]

    ring_scored = _score_ring_segments(
        all_raw,
        mask=mask,
        ring=ring,
        axis_dir=axis_dir,
        origin=origin,
        half_len=half,
        half_w=half_w,
        ban=ban,
        min_len=max(36.0, 0.045 * max(h, w)),
        min_align=0.72,
        attach_px=attach_px,
        ring_outer=ring_outer,
    )
    left, right, phantom, paired = _pick_paired_edges(
        ring_scored, origin=origin, axis_dir=axis_dir, half_len=half
    )
    n_close = sum(
        1
        for p1, p2, _ in ring_scored
        if _seg_closable_with_border(p1, p2, h=h, w=w)
    )
    note = (
        f"expand_lines=3a-3d lighting={lighting}"
        f"+raw={n_raw}+detach_drop={n_detach}+all={len(all_segs)}"
        f"+ring={len(ring_scored)}+closable={n_close}"
        f"+paired={int(paired)}"
        f"+L={int(left is not None)}+R={int(right is not None)}"
        f"+phantom={int(phantom is not None)}"
        f"+attach={attach_px}+axis_half={half:.0f}"
    )
    return ExpandLinesDebug(
        work_bgr=bgr,
        mask=mask,
        axis_origin=origin,
        axis_dir=axis_dir,
        axis_half_len=half,
        all_segs=all_segs,
        ring_segs=ring_scored[:40],
        left_seg=left,
        right_seg=right,
        phantom_seg=phantom,
        paired=paired,
        note=note,
        scale=1.0,
        lighting=lighting,
        ring_inner=ring_inner,
        ring_outer=ring_outer,
    )


def debug_expand_lines_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> ExpandLinesDebug:
    """1.3 3a–3d on 1.1b grown mask (draw lines only; no mask fill)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    wr = debug_white_refine(bgr, max_side=max_side, year=year, lighting=lighting)
    dbg = _expand_lines_from_mask(
        wr.work_bgr, wr.grown, lighting=wr.lighting
    )
    dbg.scale = wr.scale
    dbg.note = f"{wr.note.split('|')[0]}|{dbg.note}"
    return dbg


def overlay_expand_lines_debug(dbg: ExpandLinesDebug) -> np.ndarray:
    """Draw 3a–3d: cyan mask · magenta axis · yellow all · orange ring · green/blue pick."""
    out = dbg.work_bgr.copy()
    mask = dbg.mask > 0
    tint = out.copy()
    tint[mask] = (255, 255, 0)  # cyan seed/mask
    out = cv2.addWeighted(tint, 0.28, out, 0.72, 0)

    # mask contour
    cnts, _ = cv2.findContours(
        (dbg.mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)

    def _draw_seg(img, seg, color, thickness=1):
        if seg is None:
            return
        p1, p2 = seg[0], seg[1]
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    for seg in dbg.all_segs[:50]:
        _draw_seg(out, seg, (0, 255, 255), 1)  # yellow
    for seg in dbg.ring_segs[:30]:
        _draw_seg(out, seg, (0, 165, 255), 2)  # orange

    _draw_seg(out, dbg.left_seg, (0, 220, 0), 3)  # green
    _draw_seg(out, dbg.right_seg, (255, 180, 0), 3)  # blue-ish

    if dbg.phantom_seg is not None:
        p1, p2 = dbg.phantom_seg
        # dashed
        pts = np.linspace(
            np.array(p1, dtype=np.float64),
            np.array(p2, dtype=np.float64),
            num=24,
        )
        for i in range(0, len(pts) - 1, 2):
            a = (int(pts[i][0]), int(pts[i][1]))
            b = (int(pts[i + 1][0]), int(pts[i + 1][1]))
            cv2.line(out, a, b, (180, 180, 180), 2, lineType=cv2.LINE_AA)

    if dbg.axis_origin is not None and dbg.axis_dir is not None:
        o = np.array(dbg.axis_origin, dtype=np.float64)
        d = np.array(dbg.axis_dir, dtype=np.float64)
        hl = max(dbg.axis_half_len, 40.0)
        a = o - d * hl
        b = o + d * hl
        cv2.line(
            out,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            (255, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            out,
            (int(o[0]), int(o[1])),
            7,
            (255, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    lines = [
        "1.3 3a-3d EXPAND LINES (draw only)",
        "CYAN=mask  MAGENTA=axis  YELLOW=attached  ORANGE=ring",
        "GREEN/BLUE=pick  GRAY=phantom  (float road/water dropped)",
        f"paired={dbg.paired}  ring={len(dbg.ring_segs)}  {dbg.note}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        col = (80, 200, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA
        )
    return out


# ---------------------------------------------------------------------------
# 1.3c expand_trap: trend trapezoid ROI + small-range edge probe (draw only)
# ---------------------------------------------------------------------------


def _axis_frame_poly(
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    *,
    along0: float,
    along1: float,
    across0: float,
    across1: float,
) -> np.ndarray:
    """Axis-aligned trapezoid/rectangle corners (4×2 float)."""
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)
    n = float(np.linalg.norm(ad)) + 1e-9
    ad = ad / n
    perp = np.array([-ad[1], ad[0]], dtype=np.float64)
    corners = [
        o + ad * along0 + perp * across0,
        o + ad * along1 + perp * across0,
        o + ad * along1 + perp * across1,
        o + ad * along0 + perp * across1,
    ]
    return np.array(corners, dtype=np.float32)


def _chain_ban_frac(
    chain: list[tuple[int, int]],
    *,
    veg: np.ndarray,
    ban: np.ndarray | None,
    n_sample: int = 16,
) -> tuple[float, float, list[tuple[int, int]]]:
    """Return (veg_frac, skywater_frac, reject_sample_pts)."""
    if len(chain) < 2:
        return 1.0, 1.0, []
    h, w = veg.shape[:2]
    idx = np.linspace(0, len(chain) - 1, num=min(n_sample, len(chain)))
    veg_hit = ban_hit = total = 0
    marks: list[tuple[int, int]] = []
    for i in idx:
        x, y = chain[int(round(i))]
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        total += 1
        if veg[y, x]:
            veg_hit += 1
            marks.append((x, y))
        elif ban is not None and ban[y, x]:
            ban_hit += 1
            marks.append((x, y))
    if total == 0:
        return 1.0, 1.0, marks
    return veg_hit / total, ban_hit / total, marks


def _chain_attach_veg_frac(
    chain: list[tuple[int, int]],
    mask: np.ndarray,
    veg: np.ndarray,
) -> float:
    """Veg fraction where the chain leaves the mask, not at the far tip.

    A real rim may cross trees once it is off the blade (0578 tip over a
    hedge). That overhang must not reject a chain whose mask-side pixels
    are still the gray edge. A line that is plants from the mask outward
    still rejects.
    """
    if len(chain) < 2 or mask.max() == 0:
        return 1.0
    h, w = mask.shape[:2]
    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    scored: list[tuple[float, int, int]] = []
    for x, y in chain:
        if 0 <= x < w and 0 <= y < h:
            scored.append((float(dist[y, x]), x, y))
    if not scored:
        return 1.0
    scored.sort(key=lambda t: t[0])
    keep = max(6, int(round(0.35 * len(scored))))
    near = scored[:keep]
    hit = sum(1 for _d, x, y in near if bool(veg[y, x]))
    return hit / len(near)


def _fit_chain_segment(
    chain: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(chain) < 4:
        return None
    pts = np.array(chain, dtype=np.float32)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    d = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(d)) + 1e-9
    d = d / n
    c = np.array([x0, y0], dtype=np.float64)
    t = (pts.astype(np.float64) - c) @ d
    a = c + d * float(t.min())
    b = c + d * float(t.max())
    return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))


def _stitch_bidir_chains(
    fwd: list[tuple[int, int]],
    back: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge forward + reverse follows that share a start."""
    if not fwd and not back:
        return []
    if not back:
        return list(fwd)
    if not fwd:
        return list(reversed(back))
    # back[0] should ≈ fwd[0]; drop duplicate start
    merged = list(reversed(back[1:])) + list(fwd)
    return merged


def _probe_trap_edges(
    bgr: np.ndarray,
    mask: np.ndarray,
    poly: np.ndarray,
    *,
    trend_dir: tuple[float, float],
    side_sign: float,
    veg: np.ndarray,
    ban: np.ndarray | None,
    max_steps: int = 120,
    start: tuple[int, int] | None = None,
    canny_lo: int = 30,
    canny_hi: int = 100,
) -> tuple[list[tuple[int, int]] | None, str, list[tuple[int, int]]]:
    """Canny + bidirectional follow inside one trapezoid; reject sky/veg/water.

    Returns (chain_or_None, reason, reject_marks).
    """
    h, w = mask.shape[:2]
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.round(poly).astype(np.int32)], 255)
    # search outside eroded mask body (keep thin mouth)
    body = cv2.erode(
        (mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    search = cv2.bitwise_and(roi, cv2.bitwise_not(body))
    if int((search > 0).sum()) < 80:
        return None, "empty_search", []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad[search == 0] = 0
    edges = cv2.Canny(blur, canny_lo, canny_hi)
    edges = cv2.bitwise_and(edges, search)

    dist = cv2.distanceTransform(
        np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
    )
    if start is not None:
        sx, sy = start
        if not (0 <= sx < w and 0 <= sy < h and search[sy, sx] > 0):
            # snap to nearest search/edge pixel
            ys, xs = np.where(search > 0)
            if len(xs) == 0:
                return None, "no_search", []
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            j = int(np.argmin(d2))
            start = (int(xs[j]), int(ys[j]))
    else:
        ys, xs = np.where((edges > 0) & (search > 0))
        if len(xs) < 8:
            return None, "no_edge", []
        dvals = dist[ys, xs]
        score = np.abs(dvals - 14.0)
        order = np.argsort(score)
        start = None
        for k in order[:40]:
            x, y = int(xs[k]), int(ys[k])
            if float(dvals[k]) > 55.0:
                continue
            start = (x, y)
            break
        if start is None:
            start = (int(xs[order[0]]), int(ys[order[0]]))

    dx, dy = float(trend_dir[0]), float(trend_dir[1])
    # Stick to Canny. Gradient-following stalls on the current pixel and
    # then walks off the rim before the physical tip (0578).
    fwd = _follow_tip_edge(
        grad,
        edges,
        search,
        start,
        dx,
        dy,
        side_sign,
        max_steps=max_steps,
        inward=0.0,
        edge_only=True,
        gap=8,
    )
    back = _follow_tip_edge(
        grad,
        edges,
        search,
        start,
        -dx,
        -dy,
        side_sign,
        max_steps=max_steps,
        inward=0.0,
        edge_only=True,
        gap=8,
    )
    chain = _stitch_bidir_chains(fwd, back)
    if len(chain) < 8:
        return None, "short", []

    veg_f, ban_f, marks = _chain_ban_frac(chain, veg=veg, ban=ban)
    attach_veg = _chain_attach_veg_frac(chain, mask, veg)
    if attach_veg > 0.22:
        return chain, f"reject_veg={attach_veg:.2f}", marks
    if ban_f > 0.18:
        return chain, f"reject_skywater={ban_f:.2f}", marks
    if veg_f > 0.22:
        # kept: only the far end crosses plants
        return chain, "ok_tip_over_veg", []
    return chain, "ok", []


def _merge_side_chains(
    chains: list[list[tuple[int, int]]],
    *,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    join_px: float = 28.0,
) -> list[list[tuple[int, int]]]:
    """Stitch same-side chains along axis into longer boundary polylines."""
    if not chains:
        return []
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)

    def _along(pt: tuple[int, int]) -> float:
        return float(np.dot(np.array(pt, dtype=np.float64) - o, ad))

    # normalize each chain tipward, sort by start along
    norms: list[list[tuple[int, int]]] = []
    for c in chains:
        if len(c) < 4:
            continue
        norms.append(_order_chain_tipward(c, origin, axis_dir))
    norms.sort(key=lambda c: _along(c[0]))

    merged: list[list[tuple[int, int]]] = []
    for c in norms:
        if not merged:
            merged.append(list(c))
            continue
        prev = merged[-1]
        gap = float(
            np.hypot(c[0][0] - prev[-1][0], c[0][1] - prev[-1][1])
        )
        # also allow overlap: if c starts before prev end along axis
        if gap <= join_px or _along(c[0]) <= _along(prev[-1]) + 12.0:
            # append points beyond prev tip
            for p in c:
                if float(np.hypot(p[0] - prev[-1][0], p[1] - prev[-1][1])) > 2.5:
                    if _along(p) >= _along(prev[-1]) - 4.0:
                        prev.append(p)
        else:
            merged.append(list(c))
    return merged


def _expand_trap_from_lines(
    bgr: np.ndarray,
    mask: np.ndarray,
    lines: ExpandLinesDebug,
) -> ExpandTrapDebug:
    """1.3c on work-size image: keep yellow/orange context; probe traps."""
    h, w = mask.shape[:2]
    empty = ExpandTrapDebug(
        work_bgr=bgr,
        mask=mask,
        axis_origin=None,
        axis_dir=None,
        axis_half_len=0.0,
        axis_half_w=0.0,
        yellow_segs=[],
        orange_segs=[],
        trap_polys=[],
        roi_mask=np.zeros((h, w), np.uint8),
        edges=np.zeros((h, w), np.uint8),
        accepted=[],
        rejected=[],
        reject_pts=[],
        note="expand_trap=empty",
        scale=1.0,
        lighting=lines.lighting,
    )
    if mask.max() == 0:
        return empty

    origin, axis_dir, half, half_w = _mask_principal_axis(mask)
    if origin is None or axis_dir is None:
        empty.note = "expand_trap=no_axis"
        return empty

    # Trend: prefer orange/yellow mean direction if aligned; else PCA axis.
    ad = np.array(axis_dir, dtype=np.float64)
    trend = ad.copy()
    for pool in (lines.ring_segs[:8], lines.all_segs[:12]):
        dirs = []
        for p1, p2, _sc in pool:
            d, length, _ = _seg_geometry(p1, p2)
            if length < 28:
                continue
            if abs(float(np.dot(d, ad))) < 0.55:
                continue
            if float(np.dot(d, ad)) < 0:
                d = -d
            dirs.append(d)
        if len(dirs) >= 2:
            mean = np.mean(np.stack(dirs, axis=0), axis=0)
            n = float(np.linalg.norm(mean)) + 1e-9
            trend = mean / n
            break

    # Orient trend toward the thinner tip end (smaller half-width samples).
    o = np.array(origin, dtype=np.float64)
    tip_a = o + trend * half * 0.85
    tip_b = o - trend * half * 0.85
    def _local_w(pt: np.ndarray) -> float:
        x, y = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
        return float(
            _perpendicular_half_width(mask, x, y, float(trend[0]), float(trend[1]))
        )

    if _local_w(tip_a) > _local_w(tip_b) + 1.0:
        trend = -trend

    veg = _vegetation_mask(bgr)
    k_veg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.dilate(veg.astype(np.uint8), k_veg, iterations=1) > 0
    ban = None
    if lines.lighting == "overcast":
        sky_cut, water_cut, _meta = _sky_water_ban_masks(bgr)
        ban = sky_cut | water_cut
        ban = ban.copy()
        ban[: max(1, int(0.08 * h))] = True

    hw = max(half_w, 12.0)
    band = max(36.0, min(70.0, 0.055 * max(h, w)))
    # (poly, side_sign, tag, optional_start)
    traps: list[
        tuple[np.ndarray, float, str, tuple[int, int] | None]
    ] = []

    # Tip-forward trap — start from mask mouth like tip_edge (not random Canny).
    tip_xy = (
        int(np.clip(round((o + trend * half * 0.92)[0]), 0, w - 1)),
        int(np.clip(round((o + trend * half * 0.92)[1]), 0, h - 1)),
    )
    tip_w = max(6.0, hw * 0.35)
    tip_poly = _tip_roi_polygon(
        tip_xy,
        float(trend[0]),
        float(trend[1]),
        tip_w,
        max_fwd=int(max(110, min(240, 0.22 * max(h, w)))),
        back=18,
    )
    mouth = _mouth_edge_points(
        mask, tip_xy, float(trend[0]), float(trend[1]), tip_w
    )
    mouth_l = mouth[0] if mouth is not None else None
    mouth_r = mouth[1] if mouth is not None else None
    traps.append((tip_poly, -1.0, "tipL", mouth_l))
    traps.append((tip_poly, 1.0, "tipR", mouth_r))

    # Side traps covering root → tip (overlapping so we can stitch)
    for along0, along1, tag in (
        (-0.75 * half, -0.15 * half, "root"),
        (-0.35 * half, 0.35 * half, "mid"),
        (0.15 * half, 0.85 * half, "fwd"),
    ):
        for sign, side in ((1.0, "L"), (-1.0, "R")):
            across0 = sign * (hw * 0.45)
            across1 = sign * (hw * 0.45 + band)
            poly = _axis_frame_poly(
                origin,
                (float(trend[0]), float(trend[1])),
                along0=float(along0),
                along1=float(along1),
                across0=float(across0),
                across1=float(across1),
            )
            traps.append((poly, sign, f"{tag}{side}", None))

    roi_all = np.zeros((h, w), np.uint8)
    edges_all = np.zeros((h, w), np.uint8)
    raw_accepted: list[list[tuple[int, int]]] = []
    rejected: list[list[tuple[int, int]]] = []
    reject_pts: list[tuple[int, int]] = []
    trap_polys: list[np.ndarray] = []
    reasons: list[str] = []
    seen_starts: set[tuple[int, int]] = set()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    steps = int(max(100, min(200, 0.18 * max(h, w))))

    for poly, side_sign, tag, start in traps:
        trap_polys.append(poly)
        roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [np.round(poly).astype(np.int32)], 255)
        roi_all = cv2.bitwise_or(roi_all, roi)
        body = cv2.erode(
            (mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        search = cv2.bitwise_and(roi, cv2.bitwise_not(body))
        ed = cv2.Canny(blur, 28, 95)
        ed = cv2.bitwise_and(ed, search)
        edges_all = cv2.bitwise_or(edges_all, ed)

        chain, reason, marks = _probe_trap_edges(
            bgr,
            mask,
            poly,
            trend_dir=(float(trend[0]), float(trend[1])),
            side_sign=side_sign,
            veg=veg,
            ban=ban,
            max_steps=steps,
            start=start,
            canny_lo=28,
            canny_hi=95,
        )
        reasons.append(f"{tag}:{reason}")
        if marks:
            reject_pts.extend(marks[:8])
        if chain is None:
            continue
        key = chain[len(chain) // 2]
        key = (key[0] // 8, key[1] // 8)
        if key in seen_starts:
            continue
        seen_starts.add(key)
        if reason.startswith("reject"):
            rejected.append(chain)
        else:
            raw_accepted.append(chain)

    # Stitch same-side fragments → longer boundary lines (complete rim first)
    left_raw: list[list[tuple[int, int]]] = []
    right_raw: list[list[tuple[int, int]]] = []
    for chain in raw_accepted:
        mid = chain[len(chain) // 2]
        if _side_of_axis(np.array(mid, dtype=np.float64), origin, (float(trend[0]), float(trend[1]))) >= 0:
            left_raw.append(chain)
        else:
            right_raw.append(chain)
    join_px = max(36.0, 0.04 * max(h, w))
    accepted = _merge_side_chains(
        left_raw,
        origin=origin,
        axis_dir=(float(trend[0]), float(trend[1])),
        join_px=join_px,
    ) + _merge_side_chains(
        right_raw,
        origin=origin,
        axis_dir=(float(trend[0]), float(trend[1])),
        join_px=join_px,
    )
    # Drop trap greens that latch onto rust / paint seams inside a 1.3a
    # outer pick on the same side (713). No blue extend from those.
    trend_t = (float(trend[0]), float(trend[1]))
    keep_acc: list[list[tuple[int, int]]] = []
    n_interior = 0
    for chain in accepted:
        mid = chain[len(chain) // 2]
        side = _side_of_axis(np.array(mid, dtype=np.float64), origin, trend_t)
        outer = None
        for seg in (lines.left_seg, lines.right_seg):
            if seg is None:
                continue
            sm = (
                0.5 * (seg[0][0] + seg[1][0]),
                0.5 * (seg[0][1] + seg[1][1]),
            )
            if _side_of_axis(np.array(sm, dtype=np.float64), origin, trend_t) * side >= 0:
                outer = seg
                break
        if outer is not None and _chain_inward_of_outer_pick(
            chain, outer, origin=origin, axis_dir=trend_t, half_w=hw
        ):
            rejected.append(chain)
            n_interior += 1
            continue
        keep_acc.append(chain)
    accepted = keep_acc
    if n_interior:
        reasons.append(f"interior_vs_1.3a={n_interior}")

    # Aligned rims that have not met a closing edge (border or the other
    # side) keep going as a straight extension. Fill stays a separate step.
    # Clip with full-frame Canny (not trap-only edges_all) so a corner stop
    # still sees the real rim outside the yellow traps.
    rim_edges = cv2.Canny(blur, 28, 95)
    extend_segs = _rim_closure_extensions(
        accepted, h, w, trend_t, edges=rim_edges
    )

    note = (
        f"expand_trap=1.3c lighting={lines.lighting}"
        f"+traps={len(traps)}+raw={len(raw_accepted)}+accept={len(accepted)}"
        f"+reject={len(rejected)}+ext={len(extend_segs)}+interior={n_interior}"
        f"+yellow={len(lines.all_segs)}+orange={len(lines.ring_segs)}"
        f"+mouth={int(mouth is not None)}+half_w={hw:.0f}+band={band:.0f}"
        f"|{';'.join(reasons[:12])}"
    )
    return ExpandTrapDebug(
        work_bgr=bgr,
        mask=mask,
        axis_origin=origin,
        axis_dir=(float(trend[0]), float(trend[1])),
        axis_half_len=half,
        axis_half_w=hw,
        yellow_segs=list(lines.all_segs[:40]),
        orange_segs=list(lines.ring_segs[:30]),
        trap_polys=trap_polys,
        roi_mask=roi_all,
        edges=edges_all,
        accepted=accepted,
        rejected=rejected,
        reject_pts=reject_pts[:40],
        note=note,
        scale=1.0,
        lighting=lines.lighting,
        n_traps=len(traps),
        n_accept=len(accepted),
        n_reject=len(rejected),
        extend_segs=extend_segs,
    )


def debug_expand_trap_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> ExpandTrapDebug:
    """1.3c: keep expand_lines yellow/orange; trapezoid probe new edges only."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    wr = debug_white_refine(bgr, max_side=max_side, year=year, lighting=lighting)
    lines = _expand_lines_from_mask(wr.work_bgr, wr.grown, lighting=wr.lighting)
    dbg = _expand_trap_from_lines(wr.work_bgr, wr.grown, lines)
    dbg.scale = wr.scale
    dbg.note = f"{wr.note.split('|')[0]}|{dbg.note}"
    return dbg


def _draw_dashed(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: float = 16.0,
    gap: float = 9.0,
) -> None:
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    v = b - a
    n = float(np.linalg.norm(v))
    if n < 1.0:
        return
    d = v / n
    t = 0.0
    while t < n:
        t2 = min(n, t + dash)
        q1 = a + d * t
        q2 = a + d * t2
        cv2.line(
            img,
            (int(round(q1[0])), int(round(q1[1]))),
            (int(round(q2[0])), int(round(q2[1]))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        t = t2 + gap


def overlay_expand_trap_debug(dbg: ExpandTrapDebug) -> np.ndarray:
    """Yellow traps · white Canny · green accept · red reject · cyan mask."""
    out = dbg.work_bgr.copy()
    h, w = out.shape[:2]
    mask = dbg.mask > 0
    tint = out.copy()
    tint[mask] = (255, 255, 0)
    out = cv2.addWeighted(tint, 0.22, out, 0.78, 0)

    # faint original yellow/orange context
    for seg in dbg.yellow_segs[:30]:
        cv2.line(out, seg[0], seg[1], (0, 200, 200), 1, lineType=cv2.LINE_AA)
    for seg in dbg.orange_segs[:20]:
        cv2.line(out, seg[0], seg[1], (0, 140, 220), 1, lineType=cv2.LINE_AA)

    # ROI wash
    if dbg.roi_mask is not None and dbg.roi_mask.max() > 0:
        wash = out.copy()
        wash[dbg.roi_mask > 0] = (40, 40, 90)
        out = cv2.addWeighted(wash, 0.22, out, 0.78, 0)

    # trap outlines
    for poly in dbg.trap_polys:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    # white canny
    if dbg.edges is not None:
        out[dbg.edges > 0] = (255, 255, 255)

    def _draw_chain(chain, color, thickness=2):
        for i in range(len(chain) - 1):
            cv2.line(out, chain[i], chain[i + 1], color, thickness, lineType=cv2.LINE_AA)

    for chain in dbg.rejected:
        _draw_chain(chain, (0, 0, 255), 2)
    for chain in dbg.accepted:
        _draw_chain(chain, (0, 220, 0), 3)
        seg = _fit_chain_segment(chain)
        if seg is not None:
            cv2.line(out, seg[0], seg[1], (0, 255, 0), 2, lineType=cv2.LINE_AA)
        if dbg.axis_origin is not None and dbg.axis_dir is not None and len(chain) >= 2:
            tip_end = _order_chain_tipward(chain, dbg.axis_origin, dbg.axis_dir)[-1]
        else:
            tip_end = chain[-1]
        cv2.circle(out, tip_end, 8, (255, 0, 255), 2, lineType=cv2.LINE_AA)

    for e1, e2 in dbg.extend_segs:
        _draw_dashed(out, e1, e2, (255, 80, 0), 2)
        cv2.circle(out, e2, 6, (255, 80, 0), 2, lineType=cv2.LINE_AA)

    for x, y in dbg.reject_pts:
        cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

    if dbg.axis_origin is not None and dbg.axis_dir is not None:
        o = np.array(dbg.axis_origin, dtype=np.float64)
        d = np.array(dbg.axis_dir, dtype=np.float64)
        hl = max(dbg.axis_half_len, 40.0)
        a = o - d * hl
        b = o + d * hl
        cv2.arrowedLine(
            out,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            (255, 0, 255),
            2,
            tipLength=0.03,
            line_type=cv2.LINE_AA,
        )

    lines = [
        "1.3c EXPAND TRAP (draw only — tip_edge ROI style)",
        "YELLOW=trap  WHITE=Canny  GREEN=accept  BLUE=extend to close  RED=reject",
        "faint teal/orange = expand_lines context (unchanged)",
        f"traps={dbg.n_traps} accept={dbg.n_accept} reject={dbg.n_reject}  {dbg.note}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        col = (80, 200, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA
        )
    return out


def _ray_hit_border(
    origin: np.ndarray, direction: np.ndarray, h: int, w: int, *, max_t: float = 5000.0
) -> tuple[int, int]:
    """March from origin along direction until photo border; return border pixel."""
    d = direction.astype(np.float64)
    n = float(np.linalg.norm(d)) + 1e-9
    d = d / n
    step = 2.0
    last = (
        int(np.clip(origin[0], 0, w - 1)),
        int(np.clip(origin[1], 0, h - 1)),
    )
    t = 0.0
    while t < max_t:
        t += step
        p = origin + d * t
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or y < 0 or x >= w or y >= h:
            return last
        last = (x, y)
        if x <= 1 or y <= 1 or x >= w - 2 or y >= h - 2:
            return (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
    return last


def _near_image_border(
    pt: tuple[int, int], h: int, w: int, *, margin: int = 72
) -> bool:
    x, y = pt
    return x <= margin or y <= margin or x >= w - 1 - margin or y >= h - 1 - margin


def _extend_seg_tipward_to_border(
    p1: tuple[int, int],
    p2: tuple[int, int],
    origin: tuple[float, float],
    h: int,
    w: int,
    *,
    max_ext: float | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Keep root-side end; extend tip-ward end to photo border when reachable.

    Extending *both* ends of every yellow line fans a cross across the frame (624).
    """
    direction, _length, _mid = _seg_geometry(p1, p2)
    o = np.array(origin, dtype=np.float64)
    d1 = float(np.linalg.norm(np.array(p1, dtype=np.float64) - o))
    d2 = float(np.linalg.norm(np.array(p2, dtype=np.float64) - o))
    if max_ext is None:
        max_ext = 0.55 * float(max(h, w))

    if d2 >= d1:
        root, tip, tip_dir = p1, p2, direction
    else:
        root, tip, tip_dir = p2, p1, -direction

    tip_arr = np.array(tip, dtype=np.float64)
    hit = _ray_hit_border(tip_arr, tip_dir, h, w, max_t=max_ext)
    hit_dist = float(np.linalg.norm(np.array(hit, dtype=np.float64) - tip_arr))
    margin = max(48, int(0.06 * max(h, w)))
    if _near_image_border(tip, h, w, margin=margin) or (
        hit_dist <= max_ext and _near_image_border(hit, h, w, margin=4)
    ):
        return root, hit
    return p1, p2


def _line_intersection(
    p1: tuple[int, int],
    p2: tuple[int, int],
    q1: tuple[int, int],
    q2: tuple[int, int],
) -> tuple[float, float] | None:
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64) - a
    c = np.array(q1, dtype=np.float64)
    d = np.array(q2, dtype=np.float64) - c
    cross = float(b[0] * d[1] - b[1] * d[0])
    if abs(cross) < 1e-8:
        return None
    t = float((c[0] - a[0]) * d[1] - (c[1] - a[1]) * d[0]) / cross
    pt = a + b * t
    return float(pt[0]), float(pt[1])


def _point_near_chain(
    pt: tuple[int, int],
    chain: list[tuple[int, int]],
    px: float,
) -> bool:
    if not chain:
        return False
    step = max(1, len(chain) // 48)
    p = np.array(pt, dtype=np.float64)
    for q in chain[::step]:
        if float(np.hypot(p[0] - q[0], p[1] - q[1])) <= px:
            return True
    return False


def _clip_ray_to_canny_rim(
    start: tuple[int, int],
    target: tuple[int, int],
    edges: np.ndarray,
    *,
    near_px: float = 7.0,
    lose_px: float = 16.0,
    lose_hits: int = 3,
) -> tuple[int, int] | None:
    """Walk start→target; stop when Canny rim is gone (corner / sky).

    Returns last on-rim point, or None if the ray never hugs an edge.
    """
    h, w = edges.shape[:2]
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dist_e = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    a = np.array(start, dtype=np.float64)
    b = np.array(target, dtype=np.float64)
    v = b - a
    length = float(np.linalg.norm(v))
    if length < 12.0:
        return None
    d = v / length
    step = 4.0
    last_ok: tuple[int, int] | None = None
    saw_near = False
    miss = 0
    t = 0.0
    while t <= length:
        p = a + d * t
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
        de = float(dist_e[y, x])
        if de <= near_px:
            saw_near = True
            miss = 0
            last_ok = (x, y)
        elif saw_near and de > lose_px:
            miss += 1
            if miss >= lose_hits:
                break
        t += step
    if last_ok is None:
        return None
    kept = float(np.hypot(last_ok[0] - start[0], last_ok[1] - start[1]))
    if kept < 12.0:
        return None
    return last_ok


def _rim_closure_extensions(
    chains: list[list[tuple[int, int]]],
    h: int,
    w: int,
    axis_dir: tuple[float, float],
    *,
    edges: np.ndarray | None = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Extend each aligned rim until a closing edge.

    Closing edge = photo border, or the other side's line where the two
    rims meet in frame. An end that already touches either is left alone.
    Only rims that run with the blade axis are extended, so a short
    cross-edge is not fired into the sky. If ``edges`` is given, truncate
    when the ray leaves a real Canny rim (corner / empty sky). Draw-only.
    """
    fits: list[tuple[list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]] = []
    for chain in chains:
        if len(chain) < 8:
            continue
        seg = _fit_chain_segment(chain)
        if seg is not None:
            fits.append((chain, seg))
    if not fits:
        return []

    ad = np.array(axis_dir, dtype=np.float64)
    ad_n = float(np.linalg.norm(ad)) + 1e-9
    ad = ad / ad_n

    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, (chain, (a, b)) in enumerate(fits):
        others = [fits[j][0] for j in range(len(fits)) if j != i]
        other_segs = [fits[j][1] for j in range(len(fits)) if j != i]
        pts = np.array(chain, dtype=np.float64)
        for fit_end, fit_back in ((a, b), (b, a)):
            i_end = int(np.argmin(np.linalg.norm(pts - np.array(fit_end, dtype=np.float64), axis=1)))
            i_back = int(np.argmin(np.linalg.norm(pts - np.array(fit_back, dtype=np.float64), axis=1)))
            end = (int(round(pts[i_end, 0])), int(round(pts[i_end, 1])))
            span = abs(i_end - i_back)
            step = max(6, span // 4)
            i_anchor = i_end - step if i_end >= i_back else i_end + step
            i_anchor = int(np.clip(i_anchor, 0, len(chain) - 1))
            direction = pts[i_end] - pts[i_anchor]
            if _near_image_border(end, h, w, margin=12):
                continue
            if any(_point_near_chain(end, oc, 26.0) for oc in others):
                continue
            nrm = float(np.linalg.norm(direction))
            if nrm < 8.0:
                continue
            direction = direction / nrm
            if abs(float(np.dot(direction, ad))) < 0.86:
                continue
            border = _ray_hit_border(np.array(end, dtype=np.float64), direction, h, w)
            target = border
            best = float(np.hypot(border[0] - end[0], border[1] - end[1]))
            for os in other_segs:
                hit = _line_intersection(a, b, os[0], os[1])
                if hit is None:
                    continue
                ix, iy = hit
                if not (2.0 <= ix < w - 2 and 2.0 <= iy < h - 2):
                    continue
                t = float(
                    np.dot(np.array([ix, iy]) - np.array(end, dtype=np.float64), direction)
                )
                if t < 16.0 or t >= best:
                    continue
                best = t
                target = (int(round(ix)), int(round(iy)))
            if best < 12.0:
                continue
            if edges is not None and edges.max() > 0:
                clipped = _clip_ray_to_canny_rim(end, target, edges)
                if clipped is None:
                    continue
                target = clipped
            out.append((end, target))
    return out


def _order_chain_tipward(
    chain: list[tuple[int, int]],
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
) -> list[tuple[int, int]]:
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)
    a = np.array(chain[0], dtype=np.float64)
    b = np.array(chain[-1], dtype=np.float64)
    if float(np.dot(b - o, ad)) >= float(np.dot(a - o, ad)):
        return list(chain)
    return list(reversed(chain))


def _on_photo_border(
    pt: tuple[int, int], h: int, w: int, *, margin: int = 3
) -> bool:
    x, y = pt
    return x <= margin or y <= margin or x >= w - 1 - margin or y >= h - 1 - margin


def _oriented_axis_dir(
    mask: np.ndarray,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
) -> tuple[float, float]:
    """Flip axis_dir so +along points toward the far tip of the seed mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return axis_dir
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)
    along = (np.stack([xs, ys], axis=1).astype(np.float64) - o) @ ad
    if float(along.max()) >= abs(float(along.min())):
        return (float(ad[0]), float(ad[1]))
    return (float(-ad[0]), float(-ad[1]))


def _seed_contour_polyline(mask: np.ndarray) -> list[tuple[int, int]] | None:
    base = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 8:
        return None
    return [(int(p[0][0]), int(p[0][1])) for p in c]


def _nearest_poly_idx(
    pts: list[tuple[int, int]], q: tuple[int, int]
) -> int:
    qx, qy = q
    best_i, best_d = 0, 1e18
    for i, (x, y) in enumerate(pts):
        d = float((x - qx) * (x - qx) + (y - qy) * (y - qy))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _seed_contour_arc(
    mask: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    origin: tuple[float, float] | None = None,
    axis_dir: tuple[float, float] | None = None,
    prefer: str = "hub",
    max_pts: int = 80,
) -> list[tuple[int, int]] | None:
    """Walk seed outer contour from a to b. prefer=hub → more negative along-axis."""
    pts = _seed_contour_polyline(mask)
    if pts is None:
        return None
    n = len(pts)
    ia = _nearest_poly_idx(pts, a)
    ib = _nearest_poly_idx(pts, b)

    def _walk(fwd: bool) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        i = ia
        for _ in range(n + 1):
            out.append(pts[i])
            if i == ib:
                break
            i = (i + 1) % n if fwd else (i - 1) % n
        return out

    arc1, arc2 = _walk(True), _walk(False)
    if len(arc1) < 2 or len(arc2) < 2:
        return None

    def _mean_along(arc: list[tuple[int, int]]) -> float:
        if origin is None or axis_dir is None:
            return float(len(arc))
        o = np.array(origin, dtype=np.float64)
        ad = np.array(axis_dir, dtype=np.float64)
        step = max(1, len(arc) // 24)
        acc = [
            float(np.dot(np.array(p, dtype=np.float64) - o, ad))
            for p in arc[::step]
        ]
        return float(np.mean(acc)) if acc else 0.0

    if prefer == "hub" and origin is not None and axis_dir is not None:
        # hub-ward = smaller along (axis oriented tipward)
        chosen = arc1 if _mean_along(arc1) <= _mean_along(arc2) else arc2
    else:
        chosen = arc1 if len(arc1) <= len(arc2) else arc2

    if len(chosen) > max_pts:
        idx = np.linspace(0, len(chosen) - 1, num=max_pts)
        chosen = [chosen[int(round(i))] for i in idx]
    if chosen[0] != a:
        chosen = [a, *chosen]
    if chosen[-1] != b:
        chosen = [*chosen, b]
    return chosen


def _close_poly_ok(
    mask: np.ndarray,
    poly: list[tuple[int, int]],
    *,
    max_seed_out_frac: float = 0.05,
    deep_cut_px: float = 30.0,
    deep_cut_run: int = 8,
    min_cut_len: float = 80.0,
) -> tuple[bool, str]:
    """G1 seed-inside + G2 no deep chord through seed. Fail → caller should OPEN."""
    if len(poly) < 4:
        return False, "short_poly"
    h, w = mask.shape[:2]
    base = mask > 0
    seed_px = int(base.sum())
    if seed_px < 50:
        return False, "no_seed"
    solid = np.zeros((h, w), np.uint8)
    cv2.fillPoly(solid, [np.array(poly, dtype=np.int32)], 255)
    seed_out = int((base & (solid == 0)).sum())
    if seed_out / float(seed_px) > max_seed_out_frac:
        return False, f"seed_out={seed_out}/{seed_px}"

    cont = np.zeros((h, w), np.uint8)
    cnts, _ = cv2.findContours(
        base.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if cnts:
        cv2.drawContours(cont, cnts, -1, 255, 1)
        d2c = cv2.distanceTransform((cont == 0).astype(np.uint8), cv2.DIST_L2, 5)
        for i in range(len(poly) - 1):
            a0, b0 = poly[i], poly[i + 1]
            length = float(np.hypot(b0[0] - a0[0], b0[1] - a0[1]))
            if length < min_cut_len:
                continue
            nsp = max(2, int(length // 3.0))
            deep = 0
            for t in np.linspace(0.0, 1.0, nsp):
                x = int(round(a0[0] + (b0[0] - a0[0]) * t))
                y = int(round(a0[1] + (b0[1] - a0[1]) * t))
                if 0 <= x < w and 0 <= y < h and base[y, x] and d2c[y, x] > deep_cut_px:
                    deep += 1
            if deep >= deep_cut_run:
                return False, f"deep_cut={i}"
    return True, "ok"


def _close_add_ok(
    mask: np.ndarray,
    added: np.ndarray,
    *,
    far_px: float = 40.0,
    max_far_frac: float = 0.25,
) -> tuple[bool, str]:
    """G3: added pixels must stay near the seed."""
    add_u8 = added > 0
    add_px = int(add_u8.sum())
    if add_px < 80:
        return True, "ok"
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    far = int((add_u8 & (dist > far_px)).sum())
    if far / float(add_px) > max_far_frac:
        return False, f"far_add={far}/{add_px}"
    return True, "ok"


def _photo_border_polyline(
    a: tuple[int, int],
    b: tuple[int, int],
    h: int,
    w: int,
    *,
    prefer: str = "short",
) -> list[tuple[int, int]]:
    """Walk image border from a to b. prefer=short|long (of the two directions)."""
    if a == b:
        return [a]

    def _clamp_border(p: tuple[int, int]) -> tuple[int, int]:
        x, y = p
        # snap to nearest border
        opts = [
            (0, int(np.clip(y, 0, h - 1))),
            (w - 1, int(np.clip(y, 0, h - 1))),
            (int(np.clip(x, 0, w - 1)), 0),
            (int(np.clip(x, 0, w - 1)), h - 1),
        ]
        return min(opts, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)

    a = _clamp_border(a)
    b = _clamp_border(b)

    def _perim(p: tuple[int, int]) -> float:
        x, y = p
        if y <= 0:
            return float(x)
        if x >= w - 1:
            return float(w - 1 + y)
        if y >= h - 1:
            return float(w - 1 + h - 1 + (w - 1 - x))
        return float(w - 1 + h - 1 + w - 1 + (h - 1 - y))

    total = 2.0 * (w - 1 + h - 1)
    pa, pb = _perim(a), _perim(b)
    d_fwd = (pb - pa) % total
    d_back = (pa - pb) % total
    if prefer == "long":
        use_fwd = d_fwd >= d_back
        dist = d_fwd if use_fwd else d_back
    else:
        use_fwd = d_fwd <= d_back
        dist = d_fwd if use_fwd else d_back
    n = max(2, int(dist / 8.0) + 1)
    out: list[tuple[int, int]] = []
    for i in range(n + 1):
        t = i / n
        s = (pa + dist * t) % total if use_fwd else (pa - dist * t) % total
        if s <= w - 1:
            out.append((int(round(s)), 0))
        elif s <= w - 1 + h - 1:
            out.append((w - 1, int(round(s - (w - 1)))))
        elif s <= w - 1 + h - 1 + w - 1:
            out.append((int(round((w - 1) - (s - (w - 1 + h - 1)))), h - 1))
        else:
            out.append((0, int(round((h - 1) - (s - (w - 1 + h - 1 + w - 1))))))
    return out


def _fill_poly_corridor(
    mask: np.ndarray,
    poly: list[tuple[int, int]],
    *,
    ban: np.ndarray | None,
    work_bgr: np.ndarray,
    allow_extra: np.ndarray | None = None,
    expand_frac: float = 0.055,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill poly ∩ dilated base corridor; ban sky/veg. Returns filled, added, add_px."""
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)
    if len(poly) < 4:
        return base, empty, 0
    solid = np.zeros_like(base)
    cv2.fillPoly(solid, [np.array(poly, dtype=np.int32)], 255)
    expand = max(40, int(expand_frac * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
    allow = cv2.dilate(base, k)
    if allow_extra is not None and allow_extra.max() > 0:
        allow = cv2.bitwise_or(allow, allow_extra)
    for i in range(len(poly) - 1):
        cv2.line(allow, poly[i], poly[i + 1], 255, thickness=max(10, expand // 5))
    solid = cv2.bitwise_and(solid, allow)
    filled = cv2.bitwise_or(base, solid)
    added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    if ban is not None:
        ban_u8 = (ban.astype(np.uint8)) * 255 if ban.dtype == bool else ban
        filled[(added > 0) & (ban_u8 > 0)] = 0
        added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    veg = _vegetation_mask(work_bgr)
    filled[(added > 0) & veg] = 0
    added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    num, labels, _stats, _ = cv2.connectedComponentsWithStats(
        (filled > 0).astype(np.uint8), 8
    )
    if num > 1:
        best_i, best_ov = 0, -1
        for i in range(1, num):
            ov = int(((labels == i) & (base > 0)).sum())
            if ov > best_ov:
                best_ov = ov
                best_i = i
        if best_i > 0:
            filled = np.where(labels == best_i, 255, 0).astype(np.uint8)
            filled = cv2.bitwise_or(filled, base)
            added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    # Veg/sky ban can punch tiny sealed holes (0752 rust disks) — fill them.
    filled = _fill_small_interior_holes(filled, max_area=2500)
    added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    return filled, added, int((added > 0).sum())


def _rim_path_to_borders(
    chain: list[tuple[int, int]],
    extend_segs: list[tuple[tuple[int, int], tuple[int, int]]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    """Chain + matching blue extends → polyline that starts and ends on photo border."""
    if len(chain) < 4:
        return None

    def _hit_for(end: tuple[int, int], used: set[int]) -> tuple[int, tuple[int, int]] | None:
        best_i, best_d, best_hit = -1, 28.0, None
        for i, (e1, e2) in enumerate(extend_segs):
            if i in used:
                continue
            d1 = float(np.hypot(e1[0] - end[0], e1[1] - end[1]))
            d2 = float(np.hypot(e2[0] - end[0], e2[1] - end[1]))
            if d1 <= d2 and d1 < best_d and _on_photo_border(e2, h, w, margin=6):
                best_i, best_d, best_hit = i, d1, e2
            elif d2 < best_d and _on_photo_border(e1, h, w, margin=6):
                best_i, best_d, best_hit = i, d2, e1
        if best_hit is not None:
            return best_i, best_hit
        if _on_photo_border(end, h, w, margin=8):
            return -1, end
        return None

    used: set[int] = set()
    a = _hit_for(chain[0], used)
    if a is None:
        return None
    if a[0] >= 0:
        used.add(a[0])
    b = _hit_for(chain[-1], used)
    if b is None:
        return None
    return [a[1], *chain, b[1]]


def _close_fill_from_trap_border(
    mask: np.ndarray,
    trap: ExpandTrapDebug,
    *,
    ban: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    str,
    str,
]:
    """B-path: 1.3c green rims + blue dashed to photo border → fill leaf strip."""
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)
    if (
        trap.axis_origin is None
        or trap.axis_dir is None
        or len(trap.accepted) < 2
        or not trap.extend_segs
    ):
        return base, empty, [], [], "expand_close=trap_border_skip", "OPEN"

    origin, axis_dir = trap.axis_origin, trap.axis_dir
    left_c: list[list[tuple[int, int]]] = []
    right_c: list[list[tuple[int, int]]] = []
    for chain in trap.accepted:
        if len(chain) < 6:
            continue
        mid = chain[len(chain) // 2]
        if _side_of_axis(np.array(mid, dtype=np.float64), origin, axis_dir) >= 0:
            left_c.append(chain)
        else:
            right_c.append(chain)
    if not left_c or not right_c:
        return base, empty, [], [], "expand_close=trap_border_one_side", "OPEN"

    left = _order_chain_tipward(max(left_c, key=len), origin, axis_dir)
    right = _order_chain_tipward(max(right_c, key=len), origin, axis_dir)
    left_path = _rim_path_to_borders(left, trap.extend_segs, h, w)
    right_path = _rim_path_to_borders(right, trap.extend_segs, h, w)
    if left_path is None or right_path is None:
        return base, empty, [], [], "expand_close=trap_border_no_hits", "OPEN"

    # both paths run border→rim→border; stitch with two border walks
    poly: list[tuple[int, int]] = []
    poly.extend(left_path)
    poly.extend(_photo_border_polyline(left_path[-1], right_path[-1], h, w)[1:])
    poly.extend(list(reversed(right_path))[1:])
    poly.extend(_photo_border_polyline(right_path[0], left_path[0], h, w)[1:])
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    ok, why = _close_poly_ok(mask, poly)
    if not ok:
        return (
            base,
            empty,
            [],
            list(trap.extend_segs),
            f"expand_close=trap_border_reject:{why}",
            "OPEN",
        )

    filled, added, add_px = _fill_poly_corridor(
        mask,
        poly,
        ban=ban,
        work_bgr=trap.work_bgr,
        allow_extra=trap.roi_mask,
    )
    if add_px < 80:
        return base, empty, [], trap.extend_segs, "expand_close=trap_border_empty", "OPEN"
    # G3 (far-from-seed) is for lines_orange only — border strip can sit
    # legitimately outside a thin 1.1b seed (624).
    note = (
        f"expand_close=trap_border gate=BORDER_NEEDED"
        f"+L={len(left)}+R={len(right)}+poly={len(poly)}+add={add_px}"
    )
    return filled, added, poly, list(trap.extend_segs), note, "BORDER_NEEDED"


def _opposite_yellow_polyline(
    yellow_segs: list[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    rim_side: float,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    """Stitch 1.3a yellow segs on the side opposite the accepted 1.3c rim."""
    if not yellow_segs or abs(rim_side) < 1e-9:
        return None
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)

    def _along(pt: tuple[int, int]) -> float:
        return float(np.dot(np.array(pt, dtype=np.float64) - o, ad))

    opp: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for a, b, sc in yellow_segs:
        mid = 0.5 * (
            np.array(a, dtype=np.float64) + np.array(b, dtype=np.float64)
        )
        if _side_of_axis(mid, origin, axis_dir) * rim_side >= 0:
            continue
        # orient each seg low-along → high-along
        if _along(a) <= _along(b):
            opp.append((a, b, sc))
        else:
            opp.append((b, a, sc))
    if not opp:
        return None
    join_px = max(56.0, 0.06 * max(h, w))
    # Grow from highest-score opposite yellow (752: long bottom edge)
    seed = max(opp, key=lambda t: t[2])
    path: list[tuple[int, int]] = [seed[0], seed[1]]
    used = {id(seed)}
    for _ in range(16):
        best: tuple[float, int, bool, tuple[int, int], tuple[int, int]] | None = None
        for i, (a, b, _sc) in enumerate(opp):
            if id(opp[i]) in used:
                continue
            for near, far, pre in (
                (a, b, True),
                (b, a, True),
                (a, b, False),
                (b, a, False),
            ):
                cur = path[0] if pre else path[-1]
                d = float(np.hypot(near[0] - cur[0], near[1] - cur[1]))
                if d >= join_px:
                    continue
                if best is None or d < best[0]:
                    best = (d, i, pre, near, far)
        if best is None:
            break
        _d, i, pre, near, far = best
        used.add(id(opp[i]))
        if pre:
            if near != path[0]:
                path.insert(0, near)
            path.insert(0, far)
        else:
            if near != path[-1]:
                path.append(near)
            path.append(far)

    clean: list[tuple[int, int]] = [path[0]]
    for p in path[1:]:
        if float(np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1])) >= 3.0:
            clean.append(p)
    return clean if len(clean) >= 2 else None


def _close_fill_from_trap_one_rim(
    mask: np.ndarray,
    trap: ExpandTrapDebug,
    *,
    ban: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    str,
    str,
]:
    """752-style: one 1.3c green rim (+ blue to borders) + opposite 1.3a yellow.

    Other side priority: opposite yellow → seed contour → photo border (last).
    Photo-border-only was a bug: it ignored existing bottom yellows.
    """
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)
    if (
        trap.axis_origin is None
        or trap.axis_dir is None
        or len(trap.accepted) < 1
        or not trap.extend_segs
    ):
        return base, empty, [], [], "expand_close=trap_one_rim_skip", "OPEN"

    origin, axis_dir = trap.axis_origin, trap.axis_dir
    axis_dir = _oriented_axis_dir(mask, origin, axis_dir)
    candidates = [c for c in trap.accepted if len(c) >= 6]
    if not candidates:
        return base, empty, [], [], "expand_close=trap_one_rim_short", "OPEN"

    best_path: list[tuple[int, int]] | None = None
    best_chain: list[tuple[int, int]] | None = None
    for chain in sorted(candidates, key=len, reverse=True):
        ordered = _order_chain_tipward(chain, origin, axis_dir)
        for orient in (ordered, list(reversed(ordered))):
            path = _rim_path_to_borders(orient, trap.extend_segs, h, w)
            if path is None:
                continue
            if not (
                _on_photo_border(path[0], h, w, margin=8)
                and _on_photo_border(path[-1], h, w, margin=8)
            ):
                continue
            best_path, best_chain = path, orient
            break
        if best_path is not None:
            break
    if best_path is None or best_chain is None:
        return base, empty, [], list(trap.extend_segs), "expand_close=trap_one_rim_no_hits", "OPEN"

    mid = best_chain[len(best_chain) // 2]
    rim_side = _side_of_axis(np.array(mid, dtype=np.float64), origin, axis_dir)

    # Nudge accepted rim slightly outward so rust on the edge falls inside.
    ys_s, xs_s = np.where(mask > 0)
    if len(xs_s):
        cx, cy = float(xs_s.mean()), float(ys_s.mean())
        out_px = max(6.0, 0.008 * max(h, w))

        def _nudge_out(pt: tuple[int, int]) -> tuple[int, int]:
            x, y = pt
            if _on_photo_border(pt, h, w, margin=6):
                return pt
            vx, vy = float(x) - cx, float(y) - cy
            nrm = float(np.hypot(vx, vy)) + 1e-6
            nx = int(round(x + out_px * vx / nrm))
            ny = int(round(y + out_px * vy / nrm))
            return (
                int(np.clip(nx, 0, w - 1)),
                int(np.clip(ny, 0, h - 1)),
            )

        best_path = [_nudge_out(p) for p in best_path]

    seed_px = int((mask > 0).sum())
    candidates_poly: list[tuple[str, list[tuple[int, int]]]] = []

    # 1) opposite 1.3a yellow (752 bottom) — do NOT invent photo-bottom
    opp = _opposite_yellow_polyline(
        list(trap.yellow_segs),
        origin=origin,
        axis_dir=axis_dir,
        rim_side=rim_side,
        h=h,
        w=w,
    )
    if opp is not None and len(opp) >= 2:
        # Orient opp so opp[0] nearer best_path[0], opp[-1] nearer best_path[-1]
        d_same = float(
            np.hypot(opp[0][0] - best_path[0][0], opp[0][1] - best_path[0][1])
            + np.hypot(opp[-1][0] - best_path[-1][0], opp[-1][1] - best_path[-1][1])
        )
        d_flip = float(
            np.hypot(opp[-1][0] - best_path[0][0], opp[-1][1] - best_path[0][1])
            + np.hypot(opp[0][0] - best_path[-1][0], opp[0][1] - best_path[-1][1])
        )
        if d_flip < d_same:
            opp = list(reversed(opp))

        def _join_ends(
            a: tuple[int, int], b: tuple[int, int]
        ) -> list[tuple[int, int]]:
            """Connect without deep-cutting seed (752: side border or short seed arc)."""
            if a == b:
                return [a]
            if _on_photo_border(a, h, w, margin=8) and _on_photo_border(
                b, h, w, margin=8
            ):
                return _photo_border_polyline(a, b, h, w, prefer="short")
            # Shorter contour arc only — hub-long arc inverts the poly (752).
            arc = _seed_contour_arc(
                mask,
                a,
                b,
                origin=origin,
                axis_dir=axis_dir,
                prefer="short",
            )
            if arc is not None and len(arc) >= 2 and len(arc) < 0.35 * (h + w):
                return arc
            # Endpoint near border: walk border to nearest border snap of the other
            if _on_photo_border(a, h, w, margin=8):
                # snap b to nearest border then short walk
                bx, by = b
                snaps = [
                    (0, int(np.clip(by, 0, h - 1))),
                    (w - 1, int(np.clip(by, 0, h - 1))),
                    (int(np.clip(bx, 0, w - 1)), 0),
                    (int(np.clip(bx, 0, w - 1)), h - 1),
                ]
                snap = min(
                    snaps, key=lambda q: (q[0] - bx) ** 2 + (q[1] - by) ** 2
                )
                walk = _photo_border_polyline(a, snap, h, w, prefer="short")
                return walk + [b]
            if _on_photo_border(b, h, w, margin=8):
                ax, ay = a
                snaps = [
                    (0, int(np.clip(ay, 0, h - 1))),
                    (w - 1, int(np.clip(ay, 0, h - 1))),
                    (int(np.clip(ax, 0, w - 1)), 0),
                    (int(np.clip(ax, 0, w - 1)), h - 1),
                ]
                snap = min(
                    snaps, key=lambda q: (q[0] - ax) ** 2 + (q[1] - ay) ** 2
                )
                return [a] + _photo_border_polyline(snap, b, h, w, prefer="short")[1:]
            return [a, b]
        poly = list(best_path)
        join_r = _join_ends(best_path[-1], opp[-1])
        poly.extend(join_r[1:])
        poly.extend(list(reversed(opp))[1:])
        join_l = _join_ends(opp[0], best_path[0])
        poly.extend(join_l[1:])
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        candidates_poly.append(("yellow_opp", poly))

    # 2) seed contour between rim ends (hub/blade body, not photo bottom)
    arc = _seed_contour_arc(
        mask,
        best_path[-1],
        best_path[0],
        origin=origin,
        axis_dir=axis_dir,
        prefer="hub",
    )
    # For a top rim, hub-ward arc may be wrong; also try the other preference
    # by flipping prefer via mean-along: pick arc with MORE opposite-side mass.
    if arc is not None and len(arc) >= 3:
        poly = list(best_path)
        poly.extend(arc[1:])
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        candidates_poly.append(("seed_arc", poly))

    # 3) last resort: short photo-border only (never long — that was the 752 bug)
    poly = list(best_path)
    poly.extend(
        _photo_border_polyline(
            best_path[-1], best_path[0], h, w, prefer="short"
        )[1:]
    )
    if poly[0] != poly[-1]:
        poly.append(poly[0])
    candidates_poly.append(("border_short", poly))

    best_poly: list[tuple[int, int]] | None = None
    best_cov = -1.0
    best_tag = ""
    # Prefer yellow_opp hard; reject photo-bottom dumps
    rank = {"yellow_opp": 5, "seed_arc": 2, "border_short": 0}
    best_score = -1.0
    for tag, poly in candidates_poly:
        ok, _why = _close_poly_ok(mask, poly, max_seed_out_frac=0.08)
        if not ok:
            continue
        solid = np.zeros((h, w), np.uint8)
        cv2.fillPoly(solid, [np.array(poly, dtype=np.int32)], 255)
        cov = float(((mask > 0) & (solid > 0)).sum()) / float(max(seed_px, 1))
        ys_a = np.where(solid > 0)[0]
        far_pen = 0.0
        if len(ys_a):
            seed_ymax = int(np.where(mask > 0)[0].max()) if seed_px else h
            below = int((ys_a > seed_ymax + 8).sum())
            far_pen = below / float(max(len(ys_a), 1))
        if far_pen > 0.08 and tag.startswith("border"):
            continue
        score = cov + 0.08 * rank.get(tag, 0) - 0.9 * far_pen
        if score > best_score:
            best_score = score
            best_cov = cov
            best_poly = poly
            best_tag = tag

    if best_poly is None:
        return (
            base,
            empty,
            [],
            list(trap.extend_segs),
            "expand_close=trap_one_rim_reject:seed_gate",
            "OPEN",
        )

    solid_extra = np.zeros_like(base)
    cv2.fillPoly(solid_extra, [np.array(best_poly, dtype=np.int32)], 255)
    filled, added, add_px = _fill_poly_corridor(
        mask,
        best_poly,
        ban=ban,
        work_bgr=trap.work_bgr,
        allow_extra=solid_extra,
        expand_frac=0.08,
    )
    if add_px < 40:
        return (
            base,
            empty,
            [],
            list(trap.extend_segs),
            "expand_close=trap_one_rim_empty",
            "OPEN",
        )

    note = (
        f"expand_close=trap_one_rim gate=BORDER_NEEDED"
        f"+chain={len(best_chain)}+poly={len(best_poly)}+add={add_px}"
        f"+seal={best_tag}+seed_cov={best_cov:.3f}"
    )
    # draw yellow opp as debug extended if used
    ext = list(trap.extend_segs)
    if best_tag == "yellow_opp" and opp is not None and len(opp) >= 2:
        ext.append((opp[0], opp[-1]))
    return filled, added, best_poly, ext, note, "BORDER_NEEDED"


def _attach_oranges_to_pick(
    pick: tuple[tuple[int, int], tuple[int, int], float],
    oranges: list[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    origin: tuple[float, float],
    axis_dir: tuple[float, float],
    h: int,
    w: int,
    join_px: float = 36.0,
) -> list[tuple[int, int]]:
    """Grow 1.3a pick with same-side orange segs from tip and root (bridge to border)."""
    p1, p2, _ = pick
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)
    mid = 0.5 * (
        np.array(p1, dtype=np.float64) + np.array(p2, dtype=np.float64)
    )
    pick_side = _side_of_axis(mid, origin, axis_dir)
    if abs(pick_side) < 1e-6:
        pick_side = 1.0

    def _along(pt: tuple[int, int]) -> float:
        return float(np.dot(np.array(pt, dtype=np.float64) - o, ad))

    # tip = higher along-axis (axis already tip-oriented by caller)
    if _along(p2) >= _along(p1):
        root, tip = p1, p2
    else:
        root, tip = p2, p1
    path: list[tuple[int, int]] = [root, tip]

    same_idx: list[int] = []
    used: set[int] = set()
    for i, (a, b, _sc) in enumerate(oranges):
        sm = 0.5 * (
            np.array(a, dtype=np.float64) + np.array(b, dtype=np.float64)
        )
        if _side_of_axis(sm, origin, axis_dir) * pick_side < 0:
            continue
        same_idx.append(i)
        if {a, b} == {p1, p2}:
            used.add(i)

    bridge_px = max(join_px, 0.12 * max(h, w))

    def _grow(*, at_tip: bool) -> None:
        # Root already on photo border → do not climb hub-ward onto other borders (713).
        if (not at_tip) and _on_photo_border(path[0], h, w, margin=8):
            return
        for _ in range(8):
            cur = path[-1] if at_tip else path[0]
            best_i, best_d, best_next = -1, bridge_px + 1.0, None
            for i in same_idx:
                if i in used:
                    continue
                a, b, _sc = oranges[i]
                for x, y in ((a, b), (b, a)):
                    d = float(np.hypot(x[0] - cur[0], x[1] - cur[1]))
                    touches_border = _on_photo_border(
                        x, h, w, margin=8
                    ) or _on_photo_border(y, h, w, margin=8)
                    lim = bridge_px if touches_border else join_px
                    if d >= lim or d >= best_d:
                        continue
                    # tip growth: only tipward along-axis
                    if at_tip and _along(y) + 4.0 < _along(cur):
                        continue
                    # root growth: never climb hub-ward; bridge tipward gaps only
                    if (not at_tip) and _along(y) + 4.0 < _along(cur):
                        continue
                    best_i, best_d, best_next = i, d, y
            if best_next is None:
                break
            used.add(best_i)
            if at_tip:
                path.append(best_next)
            else:
                path.insert(0, best_next)

    _grow(at_tip=True)
    _grow(at_tip=False)

    # Orient root→tip by along-axis
    if _along(path[0]) > _along(path[-1]):
        path = list(reversed(path))
    return path


def _close_fill_from_lines_orange(
    mask: np.ndarray,
    lines: ExpandLinesDebug,
    *,
    ban: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    str,
    str,
]:
    """713-style: 1.3a green/blue pick + orange ring segs that reach photo border."""
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)
    if (
        lines.axis_origin is None
        or lines.axis_dir is None
        or lines.left_seg is None
        or lines.right_seg is None
    ):
        return base, empty, [], [], "expand_close=lines_orange_skip", "OPEN"

    axis_dir = _oriented_axis_dir(mask, lines.axis_origin, lines.axis_dir)
    oranges = list(lines.ring_segs)
    left_path = _attach_oranges_to_pick(
        lines.left_seg,
        oranges,
        origin=lines.axis_origin,
        axis_dir=axis_dir,
        h=h,
        w=w,
    )
    right_path = _attach_oranges_to_pick(
        lines.right_seg,
        oranges,
        origin=lines.axis_origin,
        axis_dir=axis_dir,
        h=h,
        w=w,
    )
    edge_segs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for path in (left_path, right_path):
        if len(path) >= 2:
            edge_segs.append((path[0], path[-1]))

    def _ensure_border(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) < 2:
            return None
        tip = path[-1]
        if _on_photo_border(tip, h, w, margin=8):
            return path
        # short tip-ward ray if already close to border
        root = path[0]
        d = np.array(tip, dtype=np.float64) - np.array(root, dtype=np.float64)
        n = float(np.linalg.norm(d)) + 1e-9
        hit = _ray_hit_border(np.array(tip, dtype=np.float64), d / n, h, w, max_t=120.0)
        if _on_photo_border(hit, h, w, margin=4) and float(
            np.hypot(hit[0] - tip[0], hit[1] - tip[1])
        ) <= 140.0:
            return path + [hit]
        return None

    left_b = _ensure_border(left_path)
    right_b = _ensure_border(right_path)
    if left_b is None or right_b is None:
        return base, empty, [], edge_segs, "expand_close=lines_orange_open", "OPEN"

    poly: list[tuple[int, int]] = []
    poly.extend(left_b)
    poly.extend(_photo_border_polyline(left_b[-1], right_b[-1], h, w)[1:])
    poly.extend(list(reversed(right_b))[1:])
    # root side: photo border, else seed-contour arc (never a deep straight chord)
    if _on_photo_border(right_b[0], h, w, margin=8) and _on_photo_border(
        left_b[0], h, w, margin=8
    ):
        poly.extend(_photo_border_polyline(right_b[0], left_b[0], h, w)[1:])
    else:
        arc = _seed_contour_arc(
            mask,
            right_b[0],
            left_b[0],
            origin=lines.axis_origin,
            axis_dir=axis_dir,
            prefer="hub",
        )
        if arc is None or len(arc) < 3:
            return (
                base,
                empty,
                [],
                edge_segs,
                "expand_close=lines_orange_open_root",
                "OPEN",
            )
        poly.extend(arc[1:])
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    ok, why = _close_poly_ok(mask, poly)
    if not ok:
        return (
            base,
            empty,
            [],
            edge_segs,
            f"expand_close=lines_orange_reject:{why}",
            "OPEN",
        )

    filled, added, add_px = _fill_poly_corridor(
        mask, poly, ban=ban, work_bgr=lines.work_bgr
    )
    if add_px < 80:
        return base, empty, [], edge_segs, "expand_close=lines_orange_empty", "OPEN"
    ok_add, why_add = _close_add_ok(mask, added)
    if not ok_add:
        return (
            base,
            empty,
            [],
            edge_segs,
            f"expand_close=lines_orange_reject:{why_add}",
            "OPEN",
        )
    note = (
        f"expand_close=lines_orange gate=BORDER_NEEDED"
        f"+L={len(left_b)}+R={len(right_b)}+poly={len(poly)}+add={add_px}"
    )
    return filled, added, poly, edge_segs, note, "BORDER_NEEDED"


def _close_fill_from_trap(
    mask: np.ndarray,
    trap: ExpandTrapDebug,
    *,
    ban: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    str,
    str,
]:
    """Close using 1.3c accepted edge chains (tip-in-frame polygon; no yellow hull).

    Returns (filled, added, poly_pts, edge_segs, note, gate).
    """
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)
    if (
        base.max() == 0
        or trap.axis_origin is None
        or trap.axis_dir is None
        or len(trap.accepted) < 2
    ):
        return base, empty, [], [], "expand_close=trap_skip", "UNSURE"

    origin = trap.axis_origin
    axis_dir = trap.axis_dir
    o = np.array(origin, dtype=np.float64)
    ad = np.array(axis_dir, dtype=np.float64)

    left_c: list[list[tuple[int, int]]] = []
    right_c: list[list[tuple[int, int]]] = []
    for chain in trap.accepted:
        if len(chain) < 6:
            continue
        mid = chain[len(chain) // 2]
        side = _side_of_axis(
            np.array(mid, dtype=np.float64), origin, axis_dir
        )
        if side >= 0:
            left_c.append(chain)
        else:
            right_c.append(chain)
    if not left_c or not right_c:
        return base, empty, [], [], "expand_close=trap_one_side", "UNSURE"

    left = _order_chain_tipward(max(left_c, key=len), origin, axis_dir)
    right = _order_chain_tipward(max(right_c, key=len), origin, axis_dir)
    seg_l = _fit_chain_segment(left)
    seg_r = _fit_chain_segment(right)
    if seg_l is None or seg_r is None:
        return base, empty, [], [], "expand_close=trap_no_fit", "UNSURE"

    # Close only from real ROI edges — no long invented tip rays (risky).
    left_tip, right_tip = left[-1], right[-1]
    tip_sep = float(
        np.hypot(left_tip[0] - right_tip[0], left_tip[1] - right_tip[1])
    )
    hit = _line_intersection(seg_l[0], seg_l[1], seg_r[0], seg_r[1])
    pad = 0.02 * min(h, w)
    tip: tuple[int, int] | None = None
    if hit is not None:
        ix, iy = hit
        in_frame = pad <= ix < w - pad and pad <= iy < h - pad
        tip_along = float(np.dot(np.array([ix, iy]) - o, ad))
        l_along = float(np.dot(np.array(left_tip, dtype=np.float64) - o, ad))
        r_along = float(np.dot(np.array(right_tip, dtype=np.float64) - o, ad))
        # intersection must sit near the tip ends of both chains
        near = (
            abs(tip_along - l_along) < 0.25 * max(trap.axis_half_len, 1.0)
            and abs(tip_along - r_along) < 0.25 * max(trap.axis_half_len, 1.0)
        )
        if in_frame and near and tip_along >= min(l_along, r_along) - 8.0:
            tip = (int(round(ix)), int(round(iy)))
    if tip is None and tip_sep <= max(18.0, 2.2 * trap.axis_half_w):
        tip = (
            int(round(0.5 * (left_tip[0] + right_tip[0]))),
            int(round(0.5 * (left_tip[1] + right_tip[1]))),
        )

    tip_dist = 0.0
    if tip is not None:
        tip_dist = float(min(tip[0], tip[1], w - 1 - tip[0], h - 1 - tip[1]))
    gate = "TIP_IN_FRAME" if tip is not None and tip_dist > 0.08 * min(h, w) else (
        "BORDER_NEEDED" if tip is not None else "OPEN"
    )

    def _sub(chain: list[tuple[int, int]], n: int = 24) -> list[tuple[int, int]]:
        if len(chain) <= n:
            return list(chain)
        idx = np.linspace(0, len(chain) - 1, num=n)
        return [chain[int(round(i))] for i in idx]

    # Incomplete rim: draw L/R only, do not invent a closed fill.
    if tip is None:
        note = (
            f"expand_close=trap_open sep={tip_sep:.0f}"
            f"+L={len(left)}+R={len(right)} (ROI rim incomplete — no fake tip)"
        )
        return base, empty, [], [seg_l, seg_r], note, "OPEN"

    axis_dir = _oriented_axis_dir(mask, origin, axis_dir)

    poly: list[tuple[int, int]] = []
    poly.extend(_sub(left))
    poly.append(tip)
    poly.extend(reversed(_sub(right)))
    # Root seal along seed contour (hub-ward) — never chop validated seed
    if poly[0] != poly[-1]:
        arc = _seed_contour_arc(
            mask,
            poly[-1],
            poly[0],
            origin=origin,
            axis_dir=axis_dir,
            prefer="hub",
        )
        if arc is not None and len(arc) >= 3:
            poly.extend(arc[1:])
        else:
            poly.append(poly[0])
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    if len(poly) < 4:
        return base, empty, [], [seg_l, seg_r], "expand_close=trap_short_poly", gate

    ok, why = _close_poly_ok(mask, poly)
    if not ok:
        return (
            base,
            empty,
            [],
            [seg_l, seg_r],
            f"expand_close=trap_reject:{why}",
            "OPEN",
        )

    filled, added, add_px = _fill_poly_corridor(
        mask,
        poly,
        ban=ban,
        work_bgr=trap.work_bgr,
        allow_extra=trap.roi_mask,
    )
    # G3 skipped here — tip-in-frame add is corridor-clipped to seed/ROI.

    edge_segs = [seg_l, seg_r]
    note = (
        f"expand_close=trap_poly gate={gate}"
        f"+L={len(left)}+R={len(right)}+poly={len(poly)}+add={add_px}"
    )
    return filled, added, poly, edge_segs, note, gate


def _close_fill_from_yellow_lines(
    mask: np.ndarray,
    yellow_segs: list[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    origin: tuple[float, float] | None = None,
    ban: np.ndarray | None = None,
    max_expand_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], str]:
    """Close attached yellow lines with photo border; convex fill (no deep concavity).

    Returns (filled, added, hull_pts, extended_segs, note).
    """
    h, w = mask.shape[:2]
    base = (mask > 0).astype(np.uint8) * 255
    if base.max() == 0 or not yellow_segs:
        empty = np.zeros_like(base)
        return base, empty, [], [], "expand_close=skip_no_yellow"

    if origin is None:
        ys, xs = np.where(base > 0)
        origin = (float(xs.mean()), float(ys.mean()))

    pts: list[list[int]] = []
    extended: list[tuple[tuple[int, int], tuple[int, int]]] = []

    # mask contour samples (blade body — already near-convex)
    cnts, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        step = max(1, len(c) // 180)
        for p in c[::step]:
            pts.append([int(p[0][0]), int(p[0][1])])

    # top yellow only — tip-ward extend to border (not both ways)
    use_segs = yellow_segs[:8]
    for p1, p2, _sc in use_segs:
        e1, e2 = _extend_seg_tipward_to_border(p1, p2, origin, h, w)
        extended.append((e1, e2))
        for x, y in _sample_seg_points(e1, e2, n=14):
            pts.append([int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))])

    if len(pts) < 8:
        empty = np.zeros_like(base)
        return base, empty, [], extended, "expand_close=too_few_pts"

    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(arr)
    hull_pts = [(int(p[0][0]), int(p[0][1])) for p in hull]

    solid = np.zeros_like(base)
    cv2.drawContours(solid, [hull], -1, 255, thickness=cv2.FILLED)

    # Safety corridor: don't swallow sky wedges far from blade / yellow
    expand = max_expand_px
    if expand is None:
        expand = max(48, int(0.07 * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
    allow = cv2.dilate(base, k)
    for e1, e2 in extended:
        cv2.line(allow, e1, e2, 255, thickness=max(14, expand // 4), lineType=cv2.LINE_AA)
    solid = cv2.bitwise_and(solid, allow)

    filled = cv2.bitwise_or(base, solid)
    added = cv2.bitwise_and(filled, cv2.bitwise_not(base))
    if ban is not None:
        ban_u8 = (ban.astype(np.uint8)) * 255 if ban.dtype == bool else ban
        filled[(added > 0) & (ban_u8 > 0)] = 0
        added = cv2.bitwise_and(filled, cv2.bitwise_not(base))

    # keep only the component overlapping the original subject
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (filled > 0).astype(np.uint8), 8
    )
    if num > 1:
        best_i, best_ov = 0, -1
        for i in range(1, num):
            ov = int(((labels == i) & (base > 0)).sum())
            if ov > best_ov:
                best_ov = ov
                best_i = i
        if best_i > 0:
            filled = np.where(labels == best_i, 255, 0).astype(np.uint8)
            filled = cv2.bitwise_or(filled, base)
            added = cv2.bitwise_and(filled, cv2.bitwise_not(base))

    add_px = int((added > 0).sum())
    note = (
        f"expand_close=hull+tip_border yellow={len(use_segs)}"
        f"+ext={len(extended)}+hull={len(hull_pts)}+add={add_px}"
        f"+expand={expand}"
    )
    return filled, added, hull_pts, extended, note


def _close_fill_overcast_priority(
    base_mask: np.ndarray,
    lines: ExpandLinesDebug,
    trap: ExpandTrapDebug,
    *,
    ban: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    str,
    str,
    str,
]:
    """Overcast 1.3 close pick order (shared by debug view + final mask).

    1) trap_poly  2) trap_border  3) trap_one_rim  4) lines_orange  5) OPEN
    Never falls back to yellow convex hull.
    Returns (filled, added, poly, extended, note, mode, gate).
    """
    base = (base_mask > 0).astype(np.uint8) * 255
    empty = np.zeros_like(base)

    filled, added, poly, extended, cnote, gate = _close_fill_from_trap(
        base, trap, ban=ban
    )
    if poly and gate == "TIP_IN_FRAME" and "trap_poly" in cnote:
        return filled, added, poly, extended, cnote, "trap_poly", gate

    filled, added, poly, extended, cnote, gate = _close_fill_from_trap_border(
        base, trap, ban=ban
    )
    if poly and gate == "BORDER_NEEDED" and "trap_border" in cnote:
        return filled, added, poly, extended, cnote, "trap_border", gate

    filled, added, poly, extended, cnote, gate = _close_fill_from_trap_one_rim(
        base, trap, ban=ban
    )
    if poly and gate == "BORDER_NEEDED" and "trap_one_rim" in cnote:
        return filled, added, poly, extended, cnote, "trap_one_rim", gate

    filled, added, poly, extended, cnote, gate = _close_fill_from_lines_orange(
        base, lines, ban=ban
    )
    if poly and "lines_orange" in cnote and "open" not in cnote and "skip" not in cnote:
        return filled, added, poly, extended, cnote, "lines_orange", gate

    return (
        base,
        empty,
        [],
        [],
        "expand_close=OPEN (no closable 1.3c/1.3a rim — no yellow hull)",
        "OPEN",
        "OPEN",
    )


def debug_expand_close_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> ExpandCloseDebug:
    """1.3 close by closability — not yellow-hull-first.

    Overcast pick order: see ``_close_fill_overcast_priority``.
    Sunny keeps legacy yellow hull.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    lines = debug_expand_lines_pass(
        bgr, max_side=max_side, year=year, lighting=lighting
    )
    ban = None
    if lines.lighting == "overcast":
        sky_cut, water_cut, _m = _sky_water_ban_masks(lines.work_bgr)
        ban = sky_cut | water_cut

    if lines.lighting == "overcast":
        trap = debug_expand_trap_pass(
            bgr, max_side=max_side, year=year, lighting=lines.lighting
        )
        trap_extra = (
            trap.note.split("|")[-1] + "|" if "|" in trap.note else trap.note + "|"
        )
        filled, added, poly, extended, cnote, mode, gate = (
            _close_fill_overcast_priority(lines.mask, lines, trap, ban=ban)
        )
        return ExpandCloseDebug(
            work_bgr=lines.work_bgr,
            base_mask=lines.mask,
            filled=filled,
            added=added,
            hull_pts=poly,
            extended_segs=extended,
            note=f"{lines.note}|{trap_extra}{cnote}",
            scale=lines.scale,
            lighting=lines.lighting,
            n_yellow=len(lines.all_segs),
            add_px=int((added > 0).sum()),
            mode=mode,
            close_poly=poly if poly else None,
            gate=gate,
        )

    # Sunny: legacy yellow hull
    segs = list(lines.all_segs)
    for seg in (lines.left_seg, lines.right_seg):
        if seg is not None:
            segs.append(seg)
    filled, added, hull_pts, extended, cnote = _close_fill_from_yellow_lines(
        lines.mask, segs, origin=lines.axis_origin, ban=ban
    )
    return ExpandCloseDebug(
        work_bgr=lines.work_bgr,
        base_mask=lines.mask,
        filled=filled,
        added=added,
        hull_pts=hull_pts,
        extended_segs=extended,
        note=f"{lines.note}|{cnote}",
        scale=lines.scale,
        lighting=lines.lighting,
        n_yellow=len(segs),
        add_px=int((added > 0).sum()),
        mode="yellow_hull",
        close_poly=None,
        gate="",
    )


def overlay_expand_close_debug(dbg: ExpandCloseDebug) -> np.ndarray:
    """Cyan=base · red=closed fill added · yellow=edges · magenta=close poly."""
    out = dbg.work_bgr.copy()
    base = dbg.base_mask > 0
    added = dbg.added > 0
    tint = out.copy()
    tint[base] = (255, 255, 0)
    tint[added] = (0, 0, 255)
    out = cv2.addWeighted(tint, 0.42, out, 0.58, 0)

    for e1, e2 in dbg.extended_segs:
        cv2.line(out, e1, e2, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    poly = dbg.close_poly if dbg.close_poly else dbg.hull_pts
    if len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    if dbg.mode == "trap_poly":
        title = "1.3 EXPAND CLOSE (1.3c green → tip poly)"
        legend = "CYAN=1.1b  RED=add  YELLOW=L/R  MAGENTA=close poly"
    elif dbg.mode == "trap_border":
        title = "1.3 EXPAND CLOSE (1.3c green+blue dashed → border)"
        legend = "CYAN=1.1b  RED=add  YELLOW=blue extends  MAGENTA=close poly"
    elif dbg.mode == "trap_one_rim":
        title = "1.3 EXPAND CLOSE (1.3c one rim+blue → border)"
        legend = "CYAN=1.1b  RED=add  YELLOW=blue extends  MAGENTA=close poly"
    elif dbg.mode == "lines_orange":
        title = "1.3 EXPAND CLOSE (1.3a green+orange → border)"
        legend = "CYAN=1.1b  RED=add  YELLOW=pick/orange  MAGENTA=close poly"
    elif dbg.mode == "OPEN":
        title = "1.3 EXPAND CLOSE (OPEN — no closable rim, no yellow hull)"
        legend = "CYAN=1.1b  RED=add(0)  MAGENTA=none"
    else:
        title = "1.3 EXPAND CLOSE (yellow × photo border → hull)"
        legend = "CYAN=1.1b  RED=add  YELLOW=ext to border  MAGENTA=hull"
    lines = [
        title,
        legend,
        f"mode={dbg.mode} gate={dbg.gate or '-'} add_px={dbg.add_px}",
        f"{dbg.note}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        col = (80, 200, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA
        )
    return out


def debug_white_refine(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> WhiteRefineDebug:
    """Run geometry seed + white grow only; for step inspection (no tip yet)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    from blade.lighting import estimate_lighting

    small, scale = _resize_max_side_bgr(bgr, max_side)
    lit = lighting or estimate_lighting(small, year=year).lighting
    seed, hub_xy, _n, geo = _geometry_seed(small, lit)
    grow_iters = 160  # unused when overcast (skip path)
    grown, note, dbg = _refine_mask_white_rules_ex(
        small, seed, max_grow_iters=grow_iters, lighting=lit
    )
    hub2, _ = _hub_from_distance(seed)
    if hub2 is not None:
        hub_xy = hub2
    rejected = dbg.get("rejected")
    if rejected is None:
        rejected = np.zeros_like(dbg["seed"])
    return WhiteRefineDebug(
        work_bgr=small,
        seed=dbg["seed"],
        core=dbg["core"],
        domain=dbg["domain"],
        grown=grown,
        added=dbg["added"],
        hub_xy=hub_xy,
        white_frac=float(dbg["white_frac"]),
        max_dist=float(dbg["max_dist"]),
        note=f"lighting={lit}|{geo}|{note}",
        scale=scale,
        lighting=lit,
        seed_detail=geo,
        rejected=rejected,
    )


def debug_geo2_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> Geo2Debug:
    """Seed → white grow → pass-2 geometry trim (before tip complete)."""
    wr = debug_white_refine(bgr, max_side=max_side, year=year, lighting=lighting)
    cleaned, note, dbg = _second_geometry_trim(wr.seed, wr.grown, wr.hub_xy)
    return Geo2Debug(
        work_bgr=wr.work_bgr,
        seed=wr.seed,
        grown=wr.grown,
        cleaned=cleaned,
        kept_added=dbg["kept_added"],
        cut_added=dbg["cut_added"],
        hub_xy=wr.hub_xy,
        arm_samples=list(dbg["arm_samples"]),
        tip_end=dbg["tip_end"],
        tip_dir=dbg["tip_dir"],
        allow_pts=list(dbg["allow_pts"]),
        mode=str(dbg["mode"]),
        note=note,
        scale=wr.scale,
    )


def overlay_geo2_debug(dbg: Geo2Debug) -> np.ndarray:
    """Helper overlay for pass-2 geometry.

    - Cyan: geometric seed
    - Green: white-grow kept by geo2
    - Red: white-grow cut (obvious expansion)
    - Magenta: seed centerline + tip ray
    - Yellow dots: tip corridor envelope
    """
    out = dbg.work_bgr.copy()
    seed = dbg.seed > 0
    kept = dbg.kept_added > 0
    cut = dbg.cut_added > 0

    tint = out.copy()
    tint[seed] = (255, 255, 0)
    tint[kept] = (0, 220, 0)
    tint[cut] = (0, 0, 255)
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    for m, color, th in (
        (seed, (255, 255, 0), 2),
        (kept, (0, 220, 0), 1),
        (cut, (0, 0, 255), 2),
    ):
        u8 = (m.astype(np.uint8)) * 255
        cs, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, color, th, lineType=cv2.LINE_AA)

    # centerline
    if len(dbg.arm_samples) >= 2:
        pts = np.array(dbg.arm_samples, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (255, 0, 255), 2, lineType=cv2.LINE_AA)
    if dbg.tip_end is not None and dbg.tip_dir is not None:
        tx, ty = dbg.tip_end
        dx, dy = dbg.tip_dir
        cv2.circle(out, (tx, ty), 10, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        x1 = int(round(tx + dx * 140))
        y1 = int(round(ty + dy * 140))
        cv2.line(out, (tx, ty), (x1, y1), (255, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "SEED_TIP",
            (tx - 40, ty - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for px, py in dbg.allow_pts:
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.circle(out, (px, py), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    if dbg.hub_xy is not None:
        cv2.circle(out, dbg.hub_xy, 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

    lines = [
        "CYAN=seed  GREEN=geo2 kept  RED=geo2 cut",
        "MAGENTA=centerline/tip ray  YELLOW=tip envelope",
        f"mode={dbg.mode}  {dbg.note}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def debug_tip_need_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> TipNeedDebug:
    """geo2 mask → tip-need gate only (1.1d gate view)."""
    wr = debug_white_refine(bgr, max_side=max_side, year=year, lighting=lighting)
    cleaned, _g2note, _g2 = _second_geometry_trim(wr.seed, wr.grown, wr.hub_xy)
    needed, reason, dbg = _judge_tip_needed(wr.work_bgr, cleaned, wr.hub_xy)
    return TipNeedDebug(
        work_bgr=wr.work_bgr,
        base_mask=cleaned,
        hub_xy=wr.hub_xy,
        tip_end=dbg.get("tip_end"),
        tip_dir=dbg.get("tip_dir"),
        needed=bool(needed),
        reason=reason,
        tip_w=float(dbg.get("tip_w", 0.0)),
        mid_w=float(dbg.get("mid_w", 0.0)),
        elong=float(dbg.get("elong", 1.0)),
        paint_gap=int(dbg.get("paint_gap", 0)),
        scale=wr.scale,
    )


def overlay_tip_need_debug(dbg: TipNeedDebug) -> np.ndarray:
    """Show tip-need gate: cyan mask, magenta mouth, green/red banner."""
    out = dbg.work_bgr.copy()
    base = dbg.base_mask > 0
    tint = out.copy()
    tint[base] = (255, 255, 0)
    out = cv2.addWeighted(tint, 0.40, out, 0.60, 0)

    if dbg.tip_end is not None:
        cv2.circle(out, dbg.tip_end, 10, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "MOUTH",
            (dbg.tip_end[0] - 28, dbg.tip_end[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if dbg.tip_dir is not None:
            tx, ty = dbg.tip_end
            dx, dy = dbg.tip_dir
            # short probe ray (not a full tip ROI)
            x1 = int(round(tx + dx * 80))
            y1 = int(round(ty + dy * 80))
            color = (0, 220, 0) if dbg.needed else (80, 80, 220)
            cv2.line(out, (tx, ty), (x1, y1), color, 2, lineType=cv2.LINE_AA)

    if dbg.hub_xy is not None:
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

    verdict = "TIP NEED = YES" if dbg.needed else "TIP NEED = NO"
    vcol = (0, 220, 0) if dbg.needed else (0, 0, 255)
    lines = [
        verdict,
        dbg.reason,
        f"elong={dbg.elong:.1f} tip_w={dbg.tip_w:.1f} mid_w={dbg.mid_w:.1f} paint_gap={dbg.paint_gap}",
        "YES → run tip edge; NO → leave geo2 mask alone",
    ]
    y0 = 32
    for i, text in enumerate(lines):
        col = vcol if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA
        )
    return out


def debug_tip_edge_pass(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> TipEdgeDebug:
    """geo2 cleaned mask → tip-need gate → optional tip ROI edge follow."""
    wr = debug_white_refine(bgr, max_side=max_side, year=year, lighting=lighting)
    cleaned, _g2note, _g2 = _second_geometry_trim(wr.seed, wr.grown, wr.hub_xy)
    needed, need_note, ndbg = _judge_tip_needed(wr.work_bgr, cleaned, wr.hub_xy)
    h, w = cleaned.shape[:2]
    empty_fill = np.zeros((h, w), np.uint8)
    if not needed:
        return TipEdgeDebug(
            work_bgr=wr.work_bgr,
            base_mask=cleaned,
            roi_mask=np.zeros((h, w), np.uint8),
            edges=np.zeros((h, w), np.uint8),
            left_poly=[],
            right_poly=[],
            tip_point=None,
            tip_fill=empty_fill,
            hub_xy=wr.hub_xy,
            tip_end=ndbg.get("tip_end"),
            tip_dir=ndbg.get("tip_dir"),
            needed=False,
            need_note=need_note,
            note=f"tip_edge=gated_off|{need_note}",
            scale=wr.scale,
        )

    _out, note, dbg = _complete_tip_by_edges(wr.work_bgr, cleaned, wr.hub_xy)
    tip_fill = dbg["tip_fill"]
    if tip_fill is None or tip_fill.size == 0:
        tip_fill = empty_fill
    return TipEdgeDebug(
        work_bgr=wr.work_bgr,
        base_mask=cleaned,
        roi_mask=dbg["roi_mask"],
        edges=dbg["edges"],
        left_poly=list(dbg["left_poly"]),
        right_poly=list(dbg["right_poly"]),
        tip_point=dbg["tip_point"],
        tip_fill=tip_fill,
        hub_xy=wr.hub_xy,
        tip_end=dbg["tip_end"],
        tip_dir=dbg["tip_dir"],
        needed=True,
        need_note=need_note,
        note=note,
        scale=wr.scale,
    )


def overlay_tip_edge_debug(dbg: TipEdgeDebug) -> np.ndarray:
    """Tip-edge helpers:

    - Dim ROI outline (yellow)
    - Cyan: geo2 base mask
    - Green: proposed tip fill
    - White: Canny edges in ROI
    - Orange / blue: left / right followed edges
    - Magenta: tip point
    """
    out = dbg.work_bgr.copy()
    base = dbg.base_mask > 0
    fill = dbg.tip_fill > 0
    edges = dbg.edges > 0

    tint = out.copy()
    tint[base] = (255, 255, 0)
    tint[fill & (~base)] = (0, 220, 0)
    out = cv2.addWeighted(tint, 0.40, out, 0.60, 0)

    if not dbg.needed:
        if dbg.tip_end is not None:
            cv2.circle(out, dbg.tip_end, 10, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        lines = [
            "TIP NEED = NO  (tip edge not run)",
            dbg.need_note,
            "cyan=geo2 base — leave tip alone",
        ]
        y0 = 28
        for i, text in enumerate(lines):
            col = (0, 0, 255) if i == 0 else (255, 255, 255)
            cv2.putText(
                out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA
            )
        return out

    # ROI contour
    ru = (dbg.roi_mask > 0).astype(np.uint8) * 255
    cr, _ = cv2.findContours(ru, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cr, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    # canny edges in white
    out[edges] = (255, 255, 255)

    if len(dbg.left_poly) >= 2:
        pts = np.array(dbg.left_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 140, 255), 2, lineType=cv2.LINE_AA)
    if len(dbg.right_poly) >= 2:
        pts = np.array(dbg.right_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (255, 180, 0), 2, lineType=cv2.LINE_AA)

    if dbg.tip_end is not None:
        cv2.circle(out, dbg.tip_end, 8, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "MOUTH",
            (dbg.tip_end[0] - 30, dbg.tip_end[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if dbg.tip_point is not None:
        cv2.circle(out, dbg.tip_point, 10, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out,
            dbg.tip_point,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            out,
            "TIP",
            (dbg.tip_point[0] + 10, dbg.tip_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if dbg.tip_end is not None and dbg.tip_dir is not None:
        tx, ty = dbg.tip_end
        dx, dy = dbg.tip_dir
        cv2.line(
            out,
            (tx, ty),
            (int(round(tx + dx * 120)), int(round(ty + dy * 120))),
            (255, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    if dbg.hub_xy is not None:
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

    add_px = int(((dbg.tip_fill > 0) & (dbg.base_mask == 0)).sum())
    lines = [
        "TIP NEED = YES",
        "YELLOW=tip ROI  WHITE=Canny  ORANGE=left  BLUE=right",
        "CYAN=geo2 base  GREEN=tip fill  RED cross=tip point",
        f"{dbg.need_note}",
        f"{dbg.note}  add_px={add_px}",
    ]
    y0 = 28
    for i, text in enumerate(lines):
        col = (0, 220, 0) if i == 0 else (255, 255, 255)
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, text, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA
        )
    return out


def overlay_white_refine_debug(dbg: WhiteRefineDebug) -> np.ndarray:
    """Color key for 1.1b (sunny white-grow / overcast interior fill).

    - Cyan: geometric seed
    - Red: added (sunny white-grow, or overcast interior fill / close)
    - Orange: overcast holes rejected by fill gate
    - Yellow: dmax (sunny only)
    - Green: core = seed ∩ white
    """
    out = dbg.work_bgr.copy()
    seed = dbg.seed > 0
    added = dbg.added > 0
    core = dbg.core > 0
    rejected = (
        dbg.rejected > 0
        if dbg.rejected is not None and dbg.rejected.size
        else np.zeros(dbg.seed.shape[:2], dtype=bool)
    )
    interior_mode = (
        dbg.lighting == "overcast"
        or "overcast_interior_fill" in dbg.note
        or "overcast_skip_white_grow" in dbg.note
    )

    tint = out.copy()
    tint[seed] = (255, 255, 0)  # cyan BGR: seed = start
    tint[rejected] = (0, 140, 255)  # orange: refused holes
    tint[added] = (0, 0, 255)  # red: filled / white-grow added
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    # seed / added / core / rejected contours
    seed_u8 = (seed.astype(np.uint8)) * 255
    added_u8 = (added.astype(np.uint8)) * 255
    core_u8 = (core.astype(np.uint8)) * 255
    rej_u8 = (rejected.astype(np.uint8)) * 255
    cs, _ = cv2.findContours(seed_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ca, _ = cv2.findContours(added_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc, _ = cv2.findContours(core_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cr, _ = cv2.findContours(rej_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.drawContours(out, ca, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.drawContours(out, cc, -1, (0, 220, 0), 1, lineType=cv2.LINE_AA)
    cv2.drawContours(out, cr, -1, (0, 140, 255), 1, lineType=cv2.LINE_AA)

    # dmax iso-line around seed (where white grow is allowed to reach)
    if dbg.max_dist > 1.0 and seed_u8.max() > 0:
        inv = np.where(seed_u8 > 0, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        band = ((dist > dbg.max_dist - 1.5) & (dist <= dbg.max_dist + 1.5)).astype(
            np.uint8
        ) * 255
        cb, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cb, -1, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    if dbg.hub_xy is not None:
        cv2.circle(out, dbg.hub_xy, 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out,
            dbg.hub_xy,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

    # legend
    if interior_mode:
        lines = [
            "1.1b OVERCAST interior fill  (no far white-grow)",
            "CYAN=seed  RED=filled inside contour  ORANGE=rejected hole",
            f"added_px={int(added.sum())}  reject_px={int(rejected.sum())}  "
            f"wf={dbg.white_frac:.2f}  {dbg.note}",
        ]
    else:
        lines = [
            "CYAN=seed start  RED=white-grow added  GREEN=core(seed∩white)",
            f"YELLOW=dmax({dbg.max_dist:.0f}px)  white_frac={dbg.white_frac:.2f}",
            f"added_px={int(added.sum())}  {dbg.note}",
        ]
    y0 = 28
    for i, text in enumerate(lines):
        col = (80, 200, 255) if (interior_mode and i == 0) else (255, 255, 255)
        cv2.putText(
            out,
            text,
            (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (12, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA,
        )
    return out


def overlay_seed_only(dbg: WhiteRefineDebug) -> np.ndarray:
    """Just the geometric seed before white grow."""
    out = overlay_mask_with_hub(dbg.work_bgr, dbg.seed, dbg.hub_xy)
    note = (dbg.note or "")[:120]
    if note:
        cv2.putText(
            out, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
    return out


def estimate_blade_mask(
    bgr: np.ndarray,
    *,
    max_side: int = 1280,
    year: str | None = None,
    lighting: str | None = None,
) -> MaskResult:
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    from blade.lighting import estimate_lighting

    full_h, full_w = bgr.shape[:2]
    small, scale = _resize_max_side_bgr(bgr, max_side)
    sh, sw = small.shape[:2]

    lit = lighting or estimate_lighting(small, year=year).lighting
    seed, hub_xy, n_kept, geo = _geometry_seed(small, lit)
    grown, refine_note = _refine_mask_white_rules(
        small, seed, max_grow_iters=160, lighting=lit
    )
    hub2, _ = _hub_from_distance(seed)
    if hub2 is not None:
        hub_xy = hub2
    # 1.3: overcast trap/lines close (same priority as expand_delta); sunny skip
    expand_note = "expand_close=skip"
    if lit == "overcast":
        lines_dbg = _expand_lines_from_mask(small, grown, lighting=lit)
        sky_cut, water_cut, _bm = _sky_water_ban_masks(small)
        ban = sky_cut | water_cut
        trap_dbg = _expand_trap_from_lines(small, grown, lines_dbg)
        grown, _added, _poly, _ext, cnote, mode, _gate = (
            _close_fill_overcast_priority(
                grown, lines_dbg, trap_dbg, ban=ban
            )
        )
        # cnote already starts with expand_close=…; keep mode for step-log skim
        expand_note = f"expand_close={mode}"
        if cnote and not cnote.startswith(f"expand_close={mode}"):
            expand_note = f"{expand_note}|{cnote}"
    kept, geo2_note, _geo2 = _second_geometry_trim(seed, grown, hub_xy)
    tip_need_note = "tip_need=skip"
    tip_edge_note = "tip_edge=skip"
    tip_note = "tip=skip"
    if hub_xy is not None:
        needed, tip_need_note, _nd = _judge_tip_needed(small, kept, hub_xy)
        if needed:
            kept, tip_edge_note, _te = _complete_tip_by_edges(small, kept, hub_xy)
            kept, tip_note = _complete_tip_taper(small, kept, hub_xy)
        else:
            tip_edge_note = "tip_edge=gated_off"
            tip_note = "tip=gated_off"
        hub3, _ = _hub_from_distance(kept)
        if hub3 is not None:
            hub_xy = hub3

    # Rust / paint pits leave tiny sealed holes — fill both sunny & overcast.
    before_px = int((kept > 0).sum())
    kept = _fill_small_interior_holes(kept, max_area=2500)
    hole_note = f"holes_small=+{int((kept > 0).sum()) - before_px}"

    if scale == 1.0:
        mask = kept
        hub_full = hub_xy
    else:
        mask = cv2.resize(kept, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        hub_full = None
        if hub_xy is not None:
            hub_full = (int(round(hub_xy[0] * scale)), int(round(hub_xy[1] * scale)))

    n_kept = 1 if mask.max() > 0 else 0
    area = int((mask > 0).sum())
    coverage = area / float(full_h * full_w)
    from blade.usable import assess_mask_edge_usable, assess_multi_blade_hub

    is_multi, multi_meta = assess_multi_blade_hub(bgr)
    if is_multi:
        usable, reject_reason = False, "multi_blade_hub"
        use_meta = multi_meta
    else:
        usable, reject_reason, use_meta = assess_mask_edge_usable(
            small, kept, lighting=lit
        )
    detail = (
        f"method=edge_hub lighting={lit} work={sw}x{sh} components={n_kept} "
        f"area={area} coverage={coverage:.3f} usable={int(usable)} "
        f"reject={reject_reason} rim_sobel={use_meta.get('rim_sobel')} "
        f"rim_canny={use_meta.get('rim_canny')} "
        f"mb_arms={use_meta.get('n_arms', use_meta.get('mb_n_arms'))} "
        f"{geo} {refine_note} {expand_note} "
        f"{geo2_note} {tip_need_note} {tip_edge_note} {tip_note} {hole_note}"
    )
    return MaskResult(
        mask=mask,
        work_size=(sw, sh),
        area_px=area,
        coverage=coverage,
        num_components_kept=n_kept,
        detail=detail,
        hub_xy=hub_full,
        usable=usable,
        reject_reason=reject_reason,
    )


def blade_mask(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        bgr = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray_or_bgr
    return estimate_blade_mask(bgr).mask


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    tint = out.copy()
    tint[mask > 0] = (0, 220, 0)
    cv2.addWeighted(tint, alpha, out, 1.0 - alpha, 0, out)
    return out


def overlay_mask_with_hub(
    bgr: np.ndarray,
    mask: np.ndarray,
    hub_xy: tuple[int, int] | None,
    alpha: float = 0.45,
) -> np.ndarray:
    out = overlay_mask(bgr, mask, alpha=alpha)
    if hub_xy is not None:
        cv2.circle(out, hub_xy, 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out, hub_xy, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2
        )
    return out


def apply_mask_bgr(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(bgr)
    out[mask > 0] = bgr[mask > 0]
    return out
