"""Blade-frame texture atlas: rim/axis define coords, interior is the texture.

Gold-pair direction (2026-10-08): query overcast 2017 → match sunny 2018
(rust reads clearer under overcast).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from blade.blade_mask import (
    _mask_principal_axis,
    _oriented_axis_dir,
    estimate_blade_mask,
)


@dataclass
class BladeAtlas:
    image_id: str
    atlas_gray: np.ndarray  # HxW uint8
    atlas_mask: np.ndarray  # HxW 0/255
    origin: tuple[float, float]
    axis_dir: tuple[float, float]  # tipward +along
    half_len: float
    half_w: float
    atlas_w: int
    atlas_h: int
    mask_detail: str
    source_mask: np.ndarray | None = None  # full-res blade mask


@dataclass
class RustSpot:
    """One rust locus (a cluster of interior keypoints / density peak)."""

    xy: tuple[float, float]
    along_n: float  # -1..1, tipward +
    across_n: float  # -1..1
    n_support: int


@dataclass
class RustConstellation:
    spots: list[RustSpot]
    method: str
    n_keypoints: int
    n_interior: int


def _across_dir(axis_dir: tuple[float, float]) -> tuple[float, float]:
    ax, ay = axis_dir
    return (-ay, ax)


def xy_to_blade_n(
    xy: tuple[float, float],
    atlas: BladeAtlas,
) -> tuple[float, float]:
    """Image pixel → normalized (along_n, across_n) in blade frame."""
    ox, oy = atlas.origin
    ax, ay = atlas.axis_dir
    bx, by = _across_dir(atlas.axis_dir)
    dx = float(xy[0]) - ox
    dy = float(xy[1]) - oy
    along = dx * ax + dy * ay
    across = dx * bx + dy * by
    return (
        float(along / (atlas.half_len + 1e-9)),
        float(across / (atlas.half_w + 1e-9)),
    )


def blade_n_to_atlas_uv(
    along_n: float,
    across_n: float,
    atlas: BladeAtlas,
) -> tuple[int, int]:
    u = int(round((along_n * 0.5 + 0.5) * (atlas.atlas_w - 1)))
    v = int(round((across_n * 0.5 + 0.5) * (atlas.atlas_h - 1)))
    u = int(np.clip(u, 0, atlas.atlas_w - 1))
    v = int(np.clip(v, 0, atlas.atlas_h - 1))
    return u, v


def build_blade_atlas(
    image_id: str,
    bgr: np.ndarray,
    *,
    year: str | None = None,
    atlas_w: int = 512,
    atlas_h: int = 160,
    clahe: bool = True,
) -> BladeAtlas | None:
    """Warp blade mask interior into a fixed (along × across) rectangle."""
    mr = estimate_blade_mask(bgr, year=year)
    mask = mr.mask
    if mask.max() == 0:
        return None
    origin, axis_dir, half_len, half_w = _mask_principal_axis(mask)
    if origin is None or axis_dir is None or half_len < 8 or half_w < 4:
        return None
    axis_dir = _oriented_axis_dir(mask, origin, axis_dir)
    bx, by = _across_dir(axis_dir)
    ax, ay = axis_dir
    ox, oy = origin

    # Destination (u,v): u along tipward, v across
    uu, vv = np.meshgrid(
        np.linspace(-1.0, 1.0, atlas_w, dtype=np.float32),
        np.linspace(-1.0, 1.0, atlas_h, dtype=np.float32),
    )
    along = uu * half_len
    across = vv * half_w
    map_x = (ox + along * ax + across * bx).astype(np.float32)
    map_y = (oy + along * ay + across * by).astype(np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    atlas = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    atlas_m = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    atlas = np.where(atlas_m > 0, atlas, 0).astype(np.uint8)
    return BladeAtlas(
        image_id=image_id,
        atlas_gray=atlas,
        atlas_mask=atlas_m,
        origin=origin,
        axis_dir=axis_dir,
        half_len=half_len,
        half_w=half_w,
        atlas_w=atlas_w,
        atlas_h=atlas_h,
        mask_detail=mr.detail,
        source_mask=mask,
    )


def detect_rust_constellation(
    bgr: np.ndarray,
    atlas: BladeAtlas,
    *,
    n_spots: int = 4,
    erode_px: int = 18,
) -> RustConstellation:
    """Find the obvious 4 rust loci via interior keypoint density peaks.

    Human sees four brown patches; ORB already fires there (your kps_masked
    screenshots). Code path: mask → drop rim → ORB → density peaks → top-K.
    """
    mask = atlas.source_mask
    if mask is None or mask.max() == 0:
        return RustConstellation([], "no_mask", 0, 0)

    # Interior only — rim keypoints are the "outer frame", not the rust signature.
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    interior = cv2.erode(mask, k)
    if interior.max() == 0:
        interior = mask

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Prefer denser ORB so sunny 0190 still has enough hits on faint rust.
    det = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8)
    kps = det.detect(gray, interior)
    kps = list(kps or [])
    n_all = len(kps)
    if n_all == 0:
        return RustConstellation([], "no_kps", 0, 0)

    # Distance-to-edge filter (extra rim reject beyond erode).
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for kp in kps:
        x, y = kp.pt
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or ix < 0 or iy >= dist.shape[0] or ix >= dist.shape[1]:
            continue
        if dist[iy, ix] < max(6.0, erode_px * 0.35):
            continue
        pts.append((x, y))
    n_int = len(pts)
    if n_int < 3:
        pts = [(kp.pt[0], kp.pt[1]) for kp in kps]
        n_int = len(pts)

    peaks = _density_peaks(pts, gray.shape[:2], n_peaks=n_spots, sigma=28.0)
    method = "density_peaks"
    if len(peaks) < min(3, n_spots) and n_int >= n_spots:
        peaks = _kmeans_peaks(pts, n_spots)
        method = "kmeans"

    spots: list[RustSpot] = []
    for (x, y), support in peaks:
        an, cn = xy_to_blade_n((x, y), atlas)
        spots.append(
            RustSpot(xy=(float(x), float(y)), along_n=an, across_n=cn, n_support=support)
        )
    # Stable order: tipward along, then across
    spots.sort(key=lambda s: (s.along_n, s.across_n))
    return RustConstellation(
        spots=spots[:n_spots],
        method=method,
        n_keypoints=n_all,
        n_interior=n_int,
    )


def _density_peaks(
    pts: list[tuple[float, float]],
    shape_hw: tuple[int, int],
    *,
    n_peaks: int,
    sigma: float,
) -> list[tuple[tuple[float, float], int]]:
    """Gaussian density on a coarse grid → NMS local maxima."""
    h, w = shape_hw
    scale = 4  # work smaller
    hh, ww = max(8, h // scale), max(8, w // scale)
    heat = np.zeros((hh, ww), dtype=np.float32)
    for x, y in pts:
        ix = int(np.clip(round(x / scale), 0, ww - 1))
        iy = int(np.clip(round(y / scale), 0, hh - 1))
        heat[iy, ix] += 1.0
    sig = max(1.0, sigma / scale)
    heat = cv2.GaussianBlur(heat, (0, 0), sig)
    if heat.max() <= 0:
        return []

    # NMS: keep local max vs dilated
    rad = max(3, int(round(40 / scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
    dil = cv2.dilate(heat, kernel)
    peaks_mask = (heat >= dil - 1e-6) & (heat > heat.max() * 0.12)
    ys, xs = np.where(peaks_mask)
    cands = [(float(heat[y, x]), x, y) for y, x in zip(ys, xs)]
    cands.sort(reverse=True)
    picked: list[tuple[tuple[float, float], int]] = []
    for val, x, y in cands:
        cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
        if any((cx - px) ** 2 + (cy - py) ** 2 < (55.0**2) for (px, py), _ in picked):
            continue
        # support ≈ nearby keypoints
        support = sum(
            1 for px, py in pts if (px - cx) ** 2 + (py - cy) ** 2 < (50.0**2)
        )
        picked.append(((cx, cy), support))
        if len(picked) >= n_peaks:
            break
    return picked


def _kmeans_peaks(
    pts: list[tuple[float, float]], n_spots: int
) -> list[tuple[tuple[float, float], int]]:
    arr = np.array(pts, dtype=np.float32)
    k = min(n_spots, len(pts))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _c, labels, centers = cv2.kmeans(
        arr, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS
    )
    out: list[tuple[tuple[float, float], int]] = []
    labels = labels.ravel()
    for i, c in enumerate(centers):
        support = int((labels == i).sum())
        out.append(((float(c[0]), float(c[1])), support))
    out.sort(key=lambda t: -t[1])
    return out


def flip_constellation_n(
    const: RustConstellation,
    *,
    flip_along: bool,
    flip_across: bool,
) -> RustConstellation:
    spots = []
    for s in const.spots:
        an = -s.along_n if flip_along else s.along_n
        cn = -s.across_n if flip_across else s.across_n
        spots.append(
            RustSpot(xy=s.xy, along_n=an, across_n=cn, n_support=s.n_support)
        )
    spots.sort(key=lambda s: (s.along_n, s.across_n))
    return RustConstellation(
        spots=spots,
        method=const.method + "+flip",
        n_keypoints=const.n_keypoints,
        n_interior=const.n_interior,
    )


def constellation_rms(
    a: RustConstellation,
    b: RustConstellation,
) -> float | None:
    """RMS distance in normalized blade coords (equal count, sorted)."""
    if not a.spots or not b.spots or len(a.spots) != len(b.spots):
        return None
    d2 = 0.0
    for sa, sb in zip(a.spots, b.spots):
        d2 += (sa.along_n - sb.along_n) ** 2 + (sa.across_n - sb.across_n) ** 2
    return float(np.sqrt(d2 / len(a.spots)))


def best_constellation_alignment(
    query: RustConstellation,
    ref: RustConstellation,
) -> dict:
    best = {
        "rms": None,
        "flip_along": False,
        "flip_across": False,
        "ref_view": ref,
    }
    if len(query.spots) < 3 or len(ref.spots) < 3:
        return best
    # Use min count
    n = min(len(query.spots), len(ref.spots), 4)
    q = RustConstellation(query.spots[:n], query.method, query.n_keypoints, query.n_interior)
    for fa in (False, True):
        for fc in (False, True):
            r0 = flip_constellation_n(ref, flip_along=fa, flip_across=fc)
            r = RustConstellation(r0.spots[:n], r0.method, r0.n_keypoints, r0.n_interior)
            rms = constellation_rms(q, r)
            if rms is None:
                continue
            if best["rms"] is None or rms < best["rms"]:
                best = {
                    "rms": rms,
                    "flip_along": fa,
                    "flip_across": fc,
                    "ref_view": r,
                    "query_view": q,
                }
    return best


def extract_rust_patch(
    bgr: np.ndarray,
    atlas: BladeAtlas,
    const: RustConstellation,
    *,
    out_w: int = 256,
    out_h: int = 256,
    margin: float = 2.2,
    clahe: bool = True,
) -> dict | None:
    """Crop a blade-frame patch around the 4 rust loci only (not whole blade).

    Whole-blade atlas misleads: half_w differs year-to-year, and a compact
    constellation collapses on a 512-wide strip. Rust↔rust compares this patch.
    """
    if len(const.spots) < 3:
        return None
    ans = np.array([s.along_n for s in const.spots], dtype=np.float64)
    cns = np.array([s.across_n for s in const.spots], dtype=np.float64)
    an0, cn0 = float(ans.mean()), float(cns.mean())
    span_a = float(max(ans.max() - ans.min(), 0.02))
    span_c = float(max(cns.max() - cns.min(), 0.02))
    # local window in normalized blade coords
    half_a = 0.5 * span_a * margin
    half_c = 0.5 * span_c * margin

    ox, oy = atlas.origin
    ax, ay = atlas.axis_dir
    bx, by = _across_dir(atlas.axis_dir)
    uu, vv = np.meshgrid(
        np.linspace(-1.0, 1.0, out_w, dtype=np.float32),
        np.linspace(-1.0, 1.0, out_h, dtype=np.float32),
    )
    along = (an0 + uu * half_a) * atlas.half_len
    across = (cn0 + vv * half_c) * atlas.half_w
    map_x = (ox + along * ax + across * bx).astype(np.float32)
    map_y = (oy + along * ay + across * by).astype(np.float32)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    patch = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = atlas.source_mask
    pmask = None
    if mask is not None:
        pmask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        patch = np.where(pmask > 0, patch, 0).astype(np.uint8)

    # spot positions inside the patch (for overlay)
    spots_uv: list[tuple[int, int]] = []
    for s in const.spots:
        u = int(round(((s.along_n - an0) / (half_a + 1e-9) * 0.5 + 0.5) * (out_w - 1)))
        v = int(round(((s.across_n - cn0) / (half_c + 1e-9) * 0.5 + 0.5) * (out_h - 1)))
        spots_uv.append(
            (int(np.clip(u, 0, out_w - 1)), int(np.clip(v, 0, out_h - 1)))
        )

    return {
        "gray": patch,
        "mask": pmask,
        "spots_uv": spots_uv,
        "center_n": (an0, cn0),
        "half_n": (half_a, half_c),
    }


def flip_patch(gray: np.ndarray, *, flip_along: bool, flip_across: bool) -> np.ndarray:
    out = gray
    if flip_along:
        out = np.fliplr(out)
    if flip_across:
        out = np.flipud(out)
    return out


def rust_patch_ncc(
    a: np.ndarray,
    b: np.ndarray,
    ma: np.ndarray | None = None,
    mb: np.ndarray | None = None,
) -> float | None:
    if a.shape != b.shape:
        return None
    if ma is not None and mb is not None:
        valid = (ma > 0) & (mb > 0)
    else:
        valid = (a > 0) & (b > 0)
    if int(valid.sum()) < 80:
        return None
    aa = a[valid].astype(np.float64)
    bb = b[valid].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-9
    return float((aa * bb).sum() / denom)


@dataclass
class RustGeometry:
    """Proportions of the rust cluster vs blade edges (normalized, deformation-tolerant)."""

    tip_frac: float  # 0..1 along PCA extent toward tip (+along)
    d_plus: float  # distance to +across rim (in half_w units, 0 at rim)
    d_minus: float  # distance to -across rim
    edge_ratio: float  # min(d+,d-) / max(d+,d-) — ~1 if mid-chord
    aspect: float  # constellation along_span / across_span
    center_n: tuple[float, float]
    span_n: tuple[float, float]


def measure_rust_geometry(const: RustConstellation) -> RustGeometry | None:
    if len(const.spots) < 3:
        return None
    ans = np.array([s.along_n for s in const.spots], dtype=np.float64)
    cns = np.array([s.across_n for s in const.spots], dtype=np.float64)
    an0, cn0 = float(ans.mean()), float(cns.mean())
    span_a = float(max(ans.max() - ans.min(), 1e-6))
    span_c = float(max(cns.max() - cns.min(), 1e-6))
    # +along = tipward → tip_frac in [0,1] from -1 hub side to +1 tip
    tip_frac = float(np.clip((an0 + 1.0) * 0.5, 0.0, 1.0))
    d_plus = float(max(0.0, 1.0 - cn0))
    d_minus = float(max(0.0, 1.0 + cn0))
    edge_ratio = float(min(d_plus, d_minus) / (max(d_plus, d_minus) + 1e-9))
    return RustGeometry(
        tip_frac=tip_frac,
        d_plus=d_plus,
        d_minus=d_minus,
        edge_ratio=edge_ratio,
        aspect=float(span_a / span_c),
        center_n=(an0, cn0),
        span_n=(span_a, span_c),
    )


def compare_rust_geometry(
    gq: RustGeometry,
    gr: RustGeometry,
    *,
    flip_along: bool = False,
    flip_across: bool = False,
) -> dict:
    """Compare proportions after optional flips on the ref side."""
    tip_r = (1.0 - gr.tip_frac) if flip_along else gr.tip_frac
    if flip_across:
        d_plus_r, d_minus_r = gr.d_minus, gr.d_plus
    else:
        d_plus_r, d_minus_r = gr.d_plus, gr.d_minus
    edge_r = float(min(d_plus_r, d_minus_r) / (max(d_plus_r, d_minus_r) + 1e-9))
    return {
        "tip_frac_q": gq.tip_frac,
        "tip_frac_r": tip_r,
        "tip_delta": abs(gq.tip_frac - tip_r),
        "edge_ratio_q": gq.edge_ratio,
        "edge_ratio_r": edge_r,
        "edge_delta": abs(gq.edge_ratio - edge_r),
        "aspect_q": gq.aspect,
        "aspect_r": gr.aspect,
        "aspect_delta": abs(gq.aspect - gr.aspect),
        "d_plus_q": gq.d_plus,
        "d_minus_q": gq.d_minus,
        "d_plus_r": d_plus_r,
        "d_minus_r": d_minus_r,
    }


def _rotate_gray(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def best_rust_patch_match(
    q_patch: dict,
    r_patch: dict,
    *,
    angles_deg: tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0),
) -> dict:
    """Rust↔rust: flips + small rotations; return best NCC."""
    qg = q_patch["gray"]
    qm = q_patch.get("mask")
    best: dict = {
        "ncc": None,
        "flip_along": False,
        "flip_across": False,
        "angle_deg": 0.0,
        "ref_gray": r_patch["gray"],
    }
    for fa in (False, True):
        for fc in (False, True):
            rg0 = flip_patch(r_patch["gray"], flip_along=fa, flip_across=fc)
            rm0 = r_patch.get("mask")
            if rm0 is not None:
                rm0 = flip_patch(rm0, flip_along=fa, flip_across=fc)
            for ang in angles_deg:
                rg = _rotate_gray(rg0, ang)
                rm = _rotate_gray(rm0, ang) if rm0 is not None else None
                score = rust_patch_ncc(qg, rg, qm, rm)
                if score is None:
                    continue
                if best["ncc"] is None or score > best["ncc"]:
                    best = {
                        "ncc": score,
                        "flip_along": fa,
                        "flip_across": fc,
                        "angle_deg": float(ang),
                        "ref_gray": rg,
                    }
    return best


def compose_rust_pair_preview(
    q_patch: dict,
    r_gray_aligned: np.ndarray,
    *,
    q_title: str,
    r_title: str,
    ncc: float | None,
    geom: dict | None,
    angle_deg: float = 0.0,
    flip_along: bool = False,
    flip_across: bool = False,
    q_locus: np.ndarray | None = None,
    r_locus: np.ndarray | None = None,
) -> np.ndarray:
    """Rust crops + optional original-frame provenance (where the crop came from)."""
    def _panel(gray: np.ndarray, title: str, spots: list | None, accent: tuple[int, int, int]) -> np.ndarray:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = gray.shape[:2]
        # mid lines = blade frame after straighten
        cv2.line(bgr, (0, h // 2), (w - 1, h // 2), (60, 60, 200), 1, cv2.LINE_AA)
        cv2.line(bgr, (w // 2, 0), (w // 2, h - 1), (60, 60, 200), 1, cv2.LINE_AA)
        if spots:
            for i, (u, v) in enumerate(spots, 1):
                cv2.circle(bgr, (int(u), int(v)), 6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    bgr, str(i), (int(u) + 5, int(v) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA,
                )
        cv2.putText(
            bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA
        )
        return bgr

    def _fit(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(tw / max(w, 1), th / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 28)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = small
        return canvas

    # scale up crops for eyeballing
    scale = 2
    qg = cv2.resize(q_patch["gray"], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    rg = cv2.resize(r_gray_aligned, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    q_spots = [
        (u * scale, v * scale) for u, v in (q_patch.get("spots_uv") or [])
    ]
    left = _panel(qg, q_title + " · crop", q_spots, (0, 220, 255))
    right = _panel(rg, r_title + " · crop", None, (0, 255, 120))

    gap = 12
    footer_h = 72
    cell_w = max(left.shape[1], right.shape[1])
    cell_h = max(left.shape[0], right.shape[0])
    # pad crop panels to equal size
    def _pad(panel: np.ndarray) -> np.ndarray:
        out = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        out[:] = (24, 24, 28)
        out[: panel.shape[0], : panel.shape[1]] = panel
        return out

    left, right = _pad(left), _pad(right)
    row_h = cell_h
    if q_locus is not None and r_locus is not None:
        loc_h = max(220, cell_h)
        q_loc = _fit(q_locus, cell_w, loc_h)
        r_loc = _fit(r_locus, cell_w, loc_h)
        cv2.putText(
            q_loc, "Q source (spots+crop box)", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            r_loc, "R source (spots+crop box)", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA,
        )
        top = np.zeros((loc_h, cell_w * 2 + gap, 3), dtype=np.uint8)
        top[:] = (24, 24, 28)
        top[:, :cell_w] = q_loc
        top[:, cell_w + gap :] = r_loc
        bot = np.zeros((row_h, cell_w * 2 + gap, 3), dtype=np.uint8)
        bot[:] = (24, 24, 28)
        bot[:, :cell_w] = left
        bot[:, cell_w + gap :] = right
        body = np.vstack([top, bot])
    else:
        body = np.zeros((row_h, cell_w * 2 + gap, 3), dtype=np.uint8)
        body[:] = (24, 24, 28)
        body[:, :cell_w] = left
        body[:, cell_w + gap :] = right

    canvas = np.zeros((body.shape[0] + footer_h, body.shape[1], 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    canvas[: body.shape[0]] = body

    ncc_s = f"{ncc:.3f}" if ncc is not None else "n/a"
    line1 = (
        f"rust↔rust NCC={ncc_s}  rot={angle_deg:+.0f}deg  "
        f"flip_along={flip_along} flip_across={flip_across}"
    )
    cv2.putText(
        canvas, line1, (8, canvas.shape[0] - 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    if geom:
        line2 = (
            f"edge_ratio Q={geom['edge_ratio_q']:.3f} R={geom['edge_ratio_r']:.3f} "
            f"Δ={geom['edge_delta']:.3f}  |  tip_frac Q={geom['tip_frac_q']:.3f} "
            f"R={geom['tip_frac_r']:.3f} Δ={geom['tip_delta']:.3f}  |  "
            f"aspect Q={geom['aspect_q']:.2f} R={geom['aspect_r']:.2f}"
        )
        cv2.putText(
            canvas, line2, (8, canvas.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA,
        )
    return canvas


def rust_patch_quad_xy(atlas: BladeAtlas, patch: dict) -> np.ndarray | None:
    """Four corners of the rust crop window in full-image xy (float32 Nx2)."""
    if not patch or "center_n" not in patch or "half_n" not in patch:
        return None
    an0, cn0 = float(patch["center_n"][0]), float(patch["center_n"][1])
    half_a, half_c = float(patch["half_n"][0]), float(patch["half_n"][1])
    ox, oy = atlas.origin
    ax, ay = atlas.axis_dir
    bx, by = _across_dir(atlas.axis_dir)
    corners: list[list[float]] = []
    for su, sv in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        along = (an0 + su * half_a) * atlas.half_len
        across = (cn0 + sv * half_c) * atlas.half_w
        corners.append([ox + along * ax + across * bx, oy + along * ay + across * by])
    return np.asarray(corners, dtype=np.float32)


def draw_constellation_on_bgr(
    bgr: np.ndarray,
    const: RustConstellation,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    out = bgr.copy()
    for i, s in enumerate(const.spots, 1):
        x, y = int(round(s.xy[0])), int(round(s.xy[1]))
        cv2.circle(out, (x, y), 28, color, 3, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + 16, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2,
            cv2.LINE_AA,
        )
    # link as polygon to show constellation shape
    if len(const.spots) >= 3:
        poly = np.array(
            [[int(s.xy[0]), int(s.xy[1])] for s in const.spots], dtype=np.int32
        )
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    return out


def draw_rust_provenance(
    bgr: np.ndarray,
    const: RustConstellation,
    *,
    atlas: BladeAtlas | None = None,
    patch: dict | None = None,
    color: tuple[int, int, int] = (0, 255, 255),
    box_color: tuple[int, int, int] = (0, 140, 255),
    label: str = "",
) -> np.ndarray:
    """Original-frame provenance: numbered rust loci + crop window (orange)."""
    out = draw_constellation_on_bgr(bgr, const, color=color)
    if atlas is not None and patch is not None:
        quad = rust_patch_quad_xy(atlas, patch)
        if quad is not None:
            cv2.polylines(
                out,
                [quad.astype(np.int32)],
                isClosed=True,
                color=box_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
    banner = label or f"rust locus ×{len(const.spots)}"
    cv2.putText(
        out, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )
    return out


def draw_constellation_on_atlas(
    panel_bgr: np.ndarray,
    atlas: BladeAtlas,
    const: RustConstellation,
    *,
    color: tuple[int, int, int],
    flip_along: bool = False,
    flip_across: bool = False,
) -> np.ndarray:
    out = panel_bgr.copy()
    for i, s in enumerate(const.spots, 1):
        an = -s.along_n if flip_along else s.along_n
        cn = -s.across_n if flip_across else s.across_n
        u, v = blade_n_to_atlas_uv(an, cn, atlas)
        cv2.circle(out, (u, v), 7, color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (u + 6, v - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _flip_atlas(atlas: BladeAtlas, *, flip_along: bool, flip_across: bool) -> BladeAtlas:
    g = atlas.atlas_gray
    m = atlas.atlas_mask
    if flip_along:
        g = np.fliplr(g)
        m = np.fliplr(m)
    if flip_across:
        g = np.flipud(g)
        m = np.flipud(m)
    return BladeAtlas(
        image_id=atlas.image_id,
        atlas_gray=g,
        atlas_mask=m,
        origin=atlas.origin,
        axis_dir=atlas.axis_dir,
        half_len=atlas.half_len,
        half_w=atlas.half_w,
        atlas_w=atlas.atlas_w,
        atlas_h=atlas.atlas_h,
        mask_detail=atlas.mask_detail,
        source_mask=atlas.source_mask,
    )


def atlas_ncc(
    query: BladeAtlas,
    ref: BladeAtlas,
) -> float | None:
    """Normalized correlation on overlapping mask pixels (same atlas size)."""
    if query.atlas_gray.shape != ref.atlas_gray.shape:
        return None
    valid = (query.atlas_mask > 0) & (ref.atlas_mask > 0)
    n = int(valid.sum())
    if n < 200:
        return None
    a = query.atlas_gray[valid].astype(np.float64)
    b = ref.atlas_gray[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a * b).sum() / denom)


def best_atlas_alignment(
    query: BladeAtlas,
    ref: BladeAtlas,
) -> dict:
    """Try axis/across flips; return best NCC (tip/left-right ambiguity)."""
    best = {
        "ncc": None,
        "flip_along": False,
        "flip_across": False,
        "ref_view": ref,
    }
    for fa in (False, True):
        for fc in (False, True):
            view = _flip_atlas(ref, flip_along=fa, flip_across=fc)
            score = atlas_ncc(query, view)
            if score is None:
                continue
            if best["ncc"] is None or score > best["ncc"]:
                best = {
                    "ncc": score,
                    "flip_along": fa,
                    "flip_across": fc,
                    "ref_view": view,
                }
    return best


def rust_patch_search(
    query: BladeAtlas,
    ref: BladeAtlas,
    *,
    patch_frac: float = 0.35,
) -> dict:
    """Take high-variance patch from query atlas; matchTemplate on ref."""
    q = query.atlas_gray
    qm = query.atlas_mask > 0
    # variance map (coarse) to find rusty / textured band
    blur = cv2.GaussianBlur(q, (0, 0), 3)
    local = cv2.blur((q.astype(np.float32) - blur.astype(np.float32)) ** 2, (31, 15))
    local = np.where(qm, local, 0)
    ys, xs = np.where(local >= np.percentile(local[qm], 85) if qm.any() else 0)
    if len(xs) < 20:
        return {"ok": False, "reason": "no_texture_peak"}
    cy, cx = int(np.median(ys)), int(np.median(xs))
    ph = max(24, int(query.atlas_h * 0.55))
    pw = max(48, int(query.atlas_w * patch_frac))
    y0 = max(0, cy - ph // 2)
    x0 = max(0, cx - pw // 2)
    y1 = min(query.atlas_h, y0 + ph)
    x1 = min(query.atlas_w, x0 + pw)
    patch = q[y0:y1, x0:x1]
    pm = qm[y0:y1, x0:x1]
    if pm.mean() < 0.4 or patch.size < 100:
        return {"ok": False, "reason": "thin_patch"}

    r = ref.atlas_gray
    if r.shape[0] < patch.shape[0] or r.shape[1] < patch.shape[1]:
        return {"ok": False, "reason": "ref_too_small"}
    res = cv2.matchTemplate(r, patch, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
    return {
        "ok": True,
        "score": float(max_v),
        "loc": (int(max_loc[0]), int(max_loc[1])),
        "patch_box": (x0, y0, x1, y1),
        "patch": patch,
    }


def draw_atlas_panel(atlas: BladeAtlas, *, title: str, accent: tuple[int, int, int]) -> np.ndarray:
    """BGR panel: atlas + rim bands + axis midline."""
    g = atlas.atlas_gray
    m = atlas.atlas_mask
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    # tint outside mask
    outside = m == 0
    bgr[outside] = (30, 30, 30)
    # mark outer bands (high |across|) as "rim frame"
    h, w = g.shape
    rim = np.zeros_like(m)
    band = max(2, h // 12)
    rim[:band, :] = 255
    rim[-band:, :] = 255
    rim = (rim > 0) & (m > 0)
    bgr[rim] = (
        (0.55 * bgr[rim] + 0.45 * np.array(accent, dtype=np.float32))
        .clip(0, 255)
        .astype(np.uint8)
    )
    cv2.line(bgr, (0, h // 2), (w - 1, h // 2), (80, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(
        bgr,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        accent,
        1,
        cv2.LINE_AA,
    )
    return bgr


def compose_gold_preview(
    query: BladeAtlas,
    ref_aligned: BladeAtlas,
    *,
    ncc: float | None,
    patch_info: dict | None,
) -> np.ndarray:
    left = draw_atlas_panel(
        query, title=f"Q {query.image_id}", accent=(0, 220, 255)
    )
    right = draw_atlas_panel(
        ref_aligned, title=f"R {ref_aligned.image_id}", accent=(0, 255, 120)
    )
    if patch_info and patch_info.get("ok"):
        x0, y0, x1, y1 = patch_info["patch_box"]
        cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
        lx, ly = patch_info["loc"]
        pw, ph = x1 - x0, y1 - y0
        cv2.rectangle(right, (lx, ly), (lx + pw - 1, ly + ph - 1), (0, 0, 255), 2)
    gap = 8
    canvas = np.zeros(
        (max(left.shape[0], right.shape[0]) + 36, left.shape[1] + right.shape[1] + gap, 3),
        dtype=np.uint8,
    )
    canvas[:] = (24, 24, 28)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    ncc_s = f"{ncc:.3f}" if ncc is not None else "n/a"
    patch_s = (
        f"patch={patch_info['score']:.3f}"
        if patch_info and patch_info.get("ok")
        else "patch=n/a"
    )
    cv2.putText(
        canvas,
        f"atlas NCC={ncc_s}  {patch_s}  (0752→0190 · rim=frame · interior=texture)",
        (8, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas
