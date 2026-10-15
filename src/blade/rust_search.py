"""2017 overcast rust signature → search 2018 usable gallery.

Direction is intentional: overcast rust has higher local contrast, so the
query constellation is denser; sunny gallery still carries the same loci.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from blade.blade_atlas import (
    RustGeometry,
    best_rust_patch_match,
    build_blade_atlas,
    compare_rust_geometry,
    compose_rust_pair_preview,
    detect_rust_constellation,
    draw_rust_provenance,
    extract_rust_patch,
    measure_rust_geometry,
)
from blade.usable import FILTER_VERSION, list_usable_near_images

CACHE_TAG = "rust_sig_v3"
_DEFAULT_CACHE_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "gallery_features" / CACHE_TAG
)

# Blade-normalized constellation diameter hypot(span_a, span_c).
# Local rust (0190≈0.40) keeps; cross-part blobs (0253≈1.47, 0352≈2.2) reject.
# Not photo aspect — span is in atlas (along, across) units.
SPAN_DIAM_MAX = 1.0

# Fraction of rust crop that lands on blade mask. Hub↔blade-root spreads
# (0349≈0.07) leave a mostly-empty remap; gold 0190≈0.84 / 0752≈0.79.
CROP_FILL_MIN = 0.20

# Process-lifetime gallery cache: "year/name" -> sig dict (incl. reject stubs)
_RUST_MEM: dict[str, dict] = {}


def span_diam(geom: RustGeometry | dict) -> float:
    """Constellation extent in blade-normalized coords."""
    if isinstance(geom, RustGeometry):
        sa, sc = geom.span_n
    else:
        sa, sc = geom["span_n"]
    return float(np.hypot(float(sa), float(sc)))


def cache_path(year: str, name: str, cache_root: Path | None = None) -> Path:
    root = cache_root or _DEFAULT_CACHE_ROOT
    return root / year / f"{Path(name).stem}.npz"


def _mem_key(year: str, name: str) -> str:
    return f"{year}/{Path(name).name}"


def save_sig(path: Path, *, name: str, geom: RustGeometry, patch: dict, spots: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = patch.get("mask")
    if mask is None:
        mask = np.zeros_like(patch["gray"])
    spots_xy = np.array([s.xy for s in spots], dtype=np.float32)
    spots_n = np.array([[s.along_n, s.across_n] for s in spots], dtype=np.float32)
    np.savez_compressed(
        path,
        name=np.array(name),
        tag=np.array(CACHE_TAG),
        usable_v=np.int32(FILTER_VERSION),
        reject=np.array(""),
        gray=patch["gray"],
        mask=mask,
        tip_frac=np.float64(geom.tip_frac),
        edge_ratio=np.float64(geom.edge_ratio),
        d_plus=np.float64(geom.d_plus),
        d_minus=np.float64(geom.d_minus),
        aspect=np.float64(geom.aspect),
        center_n=np.array(geom.center_n, dtype=np.float64),
        span_n=np.array(geom.span_n, dtype=np.float64),
        span_diam=np.float64(span_diam(geom)),
        crop_fill=np.float64(
            float((mask > 0).mean()) if mask.size else 0.0
        ),
        spots_xy=spots_xy,
        spots_n=spots_n,
    )


def save_reject_sig(
    path: Path,
    *,
    name: str,
    reject: str,
    n_spots: int = 0,
    span_diam_v: float | None = None,
    crop_fill_v: float | None = None,
) -> None:
    """Persist few_spots / patch_fail / span_wide / crop_leak stubs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "name": np.array(name),
        "tag": np.array(CACHE_TAG),
        "usable_v": np.int32(FILTER_VERSION),
        "reject": np.array(reject or "reject"),
        "n_spots": np.int32(n_spots),
    }
    if span_diam_v is not None:
        kwargs["span_diam"] = np.float64(span_diam_v)
    if crop_fill_v is not None:
        kwargs["crop_fill"] = np.float64(crop_fill_v)
    np.savez_compressed(path, **kwargs)


def load_sig(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        if str(data["tag"]) != CACHE_TAG or int(data["usable_v"]) != FILTER_VERSION:
            return None
        reject = ""
        if "reject" in data.files:
            reject = str(data["reject"])
        if reject:
            n_spots = int(data["n_spots"]) if "n_spots" in data.files else 0
            out = {
                "name": str(data["name"]),
                "patch": None,
                "geom": None,
                "n_spots": n_spots,
                "reject": reject,
            }
            if "span_diam" in data.files:
                out["span_diam"] = float(data["span_diam"])
            if "crop_fill" in data.files:
                out["crop_fill"] = float(data["crop_fill"])
            return out
        spots_xy = (
            np.array(data["spots_xy"], copy=True)
            if "spots_xy" in data.files
            else None
        )
        span_n = tuple(np.array(data["span_n"], copy=True).tolist())
        diam = (
            float(data["span_diam"])
            if "span_diam" in data.files
            else float(np.hypot(span_n[0], span_n[1]))
        )
        mask = np.array(data["mask"], copy=True)
        fill = (
            float(data["crop_fill"])
            if "crop_fill" in data.files
            else float((mask > 0).mean()) if mask.size else 0.0
        )
        return {
            "name": str(data["name"]),
            "patch": {
                "gray": np.array(data["gray"], copy=True),
                "mask": mask,
                "spots_uv": None,
            },
            "geom": {
                "tip_frac": float(data["tip_frac"]),
                "edge_ratio": float(data["edge_ratio"]),
                "d_plus": float(data["d_plus"]),
                "d_minus": float(data["d_minus"]),
                "aspect": float(data["aspect"]),
                "center_n": tuple(np.array(data["center_n"], copy=True).tolist()),
                "span_n": span_n,
                "span_diam": diam,
                "crop_fill": fill,
            },
            "n_spots": int(len(spots_xy)) if spots_xy is not None else 0,
            "spots_xy": spots_xy,
            "crop_fill": fill,
            "reject": None,
        }
    except (KeyError, ValueError, TypeError):
        return None
    finally:
        data.close()


def resolve_gallery_sig(
    path: Path,
    *,
    year: str,
    use_cache: bool = True,
    cache_root: Path | None = None,
) -> tuple[dict | None, str]:
    """Return (sig, source) where source is mem|disk|built|skip."""
    key = _mem_key(year, path.name)
    if key in _RUST_MEM:
        return _RUST_MEM[key], "mem"
    if use_cache:
        cached = load_sig(cache_path(year, path.name, cache_root))
        if cached is not None:
            _RUST_MEM[key] = cached
            return cached, "disk"
    built = build_sig_from_image(path, year=year)
    if built is None:
        return None, "skip"
    if use_cache:
        if (
            built.get("patch") is not None
            and built.get("geom_obj") is not None
            and built.get("const") is not None
        ):
            save_sig(
                cache_path(year, path.name, cache_root),
                name=path.name,
                geom=built["geom_obj"],
                patch=built["patch"],
                spots=built["const"].spots,
            )
        else:
            g = built.get("geom") or {}
            diam_v = g.get("span_diam")
            fill_v = built.get("crop_fill")
            if fill_v is None and g.get("crop_fill") is not None:
                fill_v = g.get("crop_fill")
            save_reject_sig(
                cache_path(year, path.name, cache_root),
                name=path.name,
                reject=str(built.get("reject") or "reject"),
                n_spots=int(built.get("n_spots") or 0),
                span_diam_v=float(diam_v) if diam_v is not None else None,
                crop_fill_v=float(fill_v) if fill_v is not None else None,
            )
    sig = {
        "name": built["name"],
        "patch": built["patch"],
        "geom": built["geom"],
        "n_spots": built["n_spots"],
        "reject": built.get("reject"),
        "geom_obj": built.get("geom_obj"),
    }
    _RUST_MEM[key] = {
        "name": sig["name"],
        "patch": sig["patch"],
        "geom": sig["geom"],
        "n_spots": sig["n_spots"],
        "reject": sig.get("reject"),
    }
    return sig, "built"


def build_sig_from_image(path: Path, *, year: str) -> dict | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    atlas = build_blade_atlas(path.name, bgr, year=year)
    if atlas is None:
        return None
    const = detect_rust_constellation(bgr, atlas)
    if len(const.spots) < 3:
        return {
            "name": path.name,
            "patch": None,
            "geom": None,
            "n_spots": len(const.spots),
            "reject": "few_spots",
            "bgr": bgr,
            "atlas": atlas,
            "const": const,
        }
    geom = measure_rust_geometry(const)
    if geom is None:
        return {
            "name": path.name,
            "patch": None,
            "geom": None,
            "n_spots": len(const.spots),
            "reject": "patch_fail",
            "bgr": bgr,
            "atlas": atlas,
            "const": const,
        }
    geom_dict = {
        "tip_frac": geom.tip_frac,
        "edge_ratio": geom.edge_ratio,
        "d_plus": geom.d_plus,
        "d_minus": geom.d_minus,
        "aspect": geom.aspect,
        "center_n": geom.center_n,
        "span_n": geom.span_n,
        "span_diam": span_diam(geom),
    }
    # Obvious cross-component constellation (nacelle↔hub, full-mask spread).
    if geom_dict["span_diam"] > SPAN_DIAM_MAX:
        return {
            "name": path.name,
            "patch": None,
            "geom": geom_dict,
            "n_spots": len(const.spots),
            "const": const,
            "geom_obj": geom,
            "bgr": bgr,
            "atlas": atlas,
            "reject": "span_wide",
        }
    patch = extract_rust_patch(bgr, atlas, const, out_w=280, out_h=280, margin=2.4)
    if patch is None:
        return {
            "name": path.name,
            "patch": None,
            "geom": geom_dict,
            "n_spots": len(const.spots),
            "reject": "patch_fail",
            "bgr": bgr,
            "atlas": atlas,
            "const": const,
        }
    pmask = patch.get("mask")
    crop_fill = float((pmask > 0).mean()) if pmask is not None and pmask.size else 0.0
    geom_dict["crop_fill"] = crop_fill
    # Crop window mostly off-paint (0349 hub↔blade-root: fill≈0.07).
    if crop_fill < CROP_FILL_MIN:
        return {
            "name": path.name,
            "patch": None,
            "geom": geom_dict,
            "n_spots": len(const.spots),
            "const": const,
            "geom_obj": geom,
            "bgr": bgr,
            "atlas": atlas,
            "crop_fill": crop_fill,
            "reject": "crop_leak",
        }
    return {
        "name": path.name,
        "patch": patch,
        "geom": geom_dict,
        "n_spots": len(const.spots),
        "const": const,
        "geom_obj": geom,
        "bgr": bgr,
        "atlas": atlas,
        "crop_fill": crop_fill,
        "reject": None,
    }


def locus_preview_name(year: str, name: str) -> str:
    return f"rust_locus_{year}_{Path(name).stem}.jpg"


def pair_preview_name(query_name: str, ref_name: str) -> str:
    return f"rust_{Path(query_name).stem}__{Path(ref_name).stem}.jpg"


def _reject_pair_panel(
    *,
    q_title: str,
    r_title: str,
    reason: str,
    q_locus: np.ndarray | None = None,
    r_locus: np.ndarray | None = None,
) -> np.ndarray:
    """Dark stub so review UI does not 404 on gated / unscored pairs."""
    tw, th = 480, 320
    gap = 12
    footer_h = 64

    def _fit(img: np.ndarray | None, label: str) -> np.ndarray:
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 32)
        if img is not None:
            h, w = img.shape[:2]
            scale = min(tw / max(w, 1), th / max(h, 1))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            y0, x0 = (th - nh) // 2, (tw - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = small
        else:
            cv2.putText(
                canvas,
                "no locus",
                (tw // 2 - 60, th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 140),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    left = _fit(q_locus, q_title)
    right = _fit(r_locus, r_title)
    body = np.zeros((th, tw * 2 + gap, 3), dtype=np.uint8)
    body[:] = (24, 24, 28)
    body[:, :tw] = left
    body[:, tw + gap :] = right
    canvas = np.zeros((th + footer_h, body.shape[1], 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    canvas[:th] = body
    cv2.putText(
        canvas,
        f"NO CROP PANEL · reject={reason}",
        (12, th + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_rust_eval_pair_preview(
    *,
    data_root: Path,
    query_name: str,
    ref_name: str,
    query_year: str = "2017",
    gallery_year: str = "2018",
    out_dir: Path,
    force: bool = False,
    use_cache: bool = True,
    cache_root: Path | None = None,
    max_side: int = 960,
    quality: int = 80,
) -> dict:
    """Write rust_{q}__{r}.jpg for one eval pair (match panel or reject stub)."""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = pair_preview_name(query_name, ref_name)
    out = out_dir / out_name
    if out.is_file() and not force:
        return {"preview": out_name, "status": "skip", "path": str(out)}

    q_path = data_root / query_year / Path(query_name).name
    r_path = data_root / gallery_year / Path(ref_name).name
    if not q_path.is_file():
        return {"preview": out_name, "status": "fail", "reject": "query_missing"}
    if not r_path.is_file():
        return {"preview": out_name, "status": "fail", "reject": "ref_missing"}

    q_sig, _ = resolve_gallery_sig(
        q_path, year=query_year, use_cache=use_cache, cache_root=cache_root
    )
    r_sig, _ = resolve_gallery_sig(
        r_path, year=gallery_year, use_cache=use_cache, cache_root=cache_root
    )
    locus_q = write_rust_locus(
        q_path,
        year=query_year,
        out_dir=out_dir,
        color=(0, 220, 255),
        force=False,
        max_side=max_side,
        quality=quality,
    )
    locus_r = write_rust_locus(
        r_path,
        year=gallery_year,
        out_dir=out_dir,
        color=(0, 255, 120),
        force=False,
        max_side=max_side,
        quality=quality,
    )
    q_locus = (
        cv2.imread(str(out_dir / locus_q))
        if locus_q and (out_dir / locus_q).is_file()
        else None
    )
    r_locus = (
        cv2.imread(str(out_dir / locus_r))
        if locus_r and (out_dir / locus_r).is_file()
        else None
    )

    q_title = f"Q {Path(query_name).name}"
    r_title = f"R {Path(ref_name).name}"
    if q_sig is None or q_sig.get("patch") is None:
        reason = (q_sig or {}).get("reject") or "query_no_patch"
        panel = _reject_pair_panel(
            q_title=q_title,
            r_title=r_title,
            reason=str(reason),
            q_locus=q_locus,
            r_locus=r_locus,
        )
        cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return {
            "preview": out_name,
            "status": "reject_stub",
            "reject": reason,
            "path": str(out),
        }

    if r_sig is None:
        panel = _reject_pair_panel(
            q_title=q_title,
            r_title=r_title,
            reason="ref_no_sig",
            q_locus=q_locus,
            r_locus=r_locus,
        )
        cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return {
            "preview": out_name,
            "status": "reject_stub",
            "reject": "ref_no_sig",
            "path": str(out),
        }

    sc = score_pair(q_sig, r_sig)
    match = sc.get("match")
    if match is None or match.get("ref_gray") is None:
        reason = sc.get("reject") or r_sig.get("reject") or "no_match"
        panel = _reject_pair_panel(
            q_title=q_title,
            r_title=r_title,
            reason=str(reason),
            q_locus=q_locus,
            r_locus=r_locus,
        )
        cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return {
            "preview": out_name,
            "status": "reject_stub",
            "reject": reason,
            "score": sc.get("score"),
            "path": str(out),
        }

    panel = compose_rust_pair_preview(
        q_sig["patch"],
        match["ref_gray"],
        q_title=q_title,
        r_title=r_title,
        ncc=sc.get("ncc"),
        geom=sc.get("geom"),
        angle_deg=float(match.get("angle_deg") or 0),
        flip_along=bool(match.get("flip_along")),
        flip_across=bool(match.get("flip_across")),
        q_locus=q_locus,
        r_locus=r_locus,
    )
    cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return {
        "preview": out_name,
        "status": "wrote",
        "score": sc.get("score"),
        "ncc": sc.get("ncc"),
        "path": str(out),
    }


def _draw_spots_xy_box(
    bgr: np.ndarray,
    spots_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    label: str,
    pad: float = 2.4,
) -> np.ndarray:
    """Fast provenance from cached spot xy (axis-aligned crop box ≈ patch)."""
    out = bgr.copy()
    pts = np.asarray(spots_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return out
    for i, (x, y) in enumerate(pts, 1):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(out, (ix, iy), 28, color, 3, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (ix + 16, iy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2,
            cv2.LINE_AA,
        )
    if len(pts) >= 3:
        cv2.polylines(
            out,
            [pts.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # padded AABB around constellation ≈ rust crop window
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        hw, hh = max((x1 - x0) * 0.5 * pad, 40.0), max((y1 - y0) * 0.5 * pad, 40.0)
        box = np.array(
            [
                [cx - hw, cy - hh],
                [cx + hw, cy - hh],
                [cx + hw, cy + hh],
                [cx - hw, cy + hh],
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [box], True, (0, 140, 255), 3, cv2.LINE_AA)
    banner = label or f"rust locus ×{len(pts)}"
    cv2.putText(
        out, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )
    return out


def render_rust_locus(
    path: Path,
    *,
    year: str,
    color: tuple[int, int, int] = (0, 255, 255),
    max_side: int = 960,
    use_cache: bool = True,
    cache_root: Path | None = None,
    precise: bool = False,
) -> tuple[np.ndarray | None, dict]:
    """Annotate where the rust crop sits on the original frame.

    Default: cached spots_xy + AABB (fast). `precise=True` rebuilds atlas crop quad.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, {"ok": False, "reject": "read_fail"}

    if not precise and use_cache:
        cached = load_sig(cache_path(year, path.name, cache_root))
        spots = None
        if cached and cached.get("spots_xy") is not None:
            spots = cached["spots_xy"]
        if spots is not None and len(spots) >= 3:
            vis = _draw_spots_xy_box(
                bgr,
                spots,
                color=color,
                label=f"{year}/{path.name}  spots={len(spots)}",
            )
            h, w = vis.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                vis = cv2.resize(
                    vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            return vis, {"ok": True, "n_spots": len(spots), "has_box": True, "mode": "cache"}

    sig = build_sig_from_image(path, year=year)
    if sig is None:
        return None, {"ok": False, "reject": "read_fail"}
    const = sig.get("const")
    atlas = sig.get("atlas")
    if const is None:
        return None, {"ok": False, "reject": sig.get("reject") or "no_const"}
    rej = sig.get("reject")
    diam = fill = None
    if sig.get("geom"):
        diam = sig["geom"].get("span_diam")
        fill = sig["geom"].get("crop_fill")
    if fill is None:
        fill = sig.get("crop_fill")
    label = f"{year}/{path.name}  spots={len(const.spots)}"
    if diam is not None:
        label += f"  diam={float(diam):.2f}"
    if fill is not None:
        label += f"  fill={float(fill):.2f}"
    if rej:
        label += f"  REJECT={rej}"
    vis = draw_rust_provenance(
        bgr,
        const,
        atlas=atlas,
        patch=sig.get("patch"),
        color=color,
        label=label,
    )
    h, w = vis.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        vis = cv2.resize(
            vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return vis, {
        "ok": True,
        "n_spots": len(const.spots),
        "reject": sig.get("reject"),
        "has_box": bool(sig.get("patch") is not None),
        "mode": "precise",
    }


def write_rust_locus(
    path: Path,
    *,
    year: str,
    out_dir: Path,
    color: tuple[int, int, int] = (0, 255, 255),
    force: bool = False,
    precise: bool = False,
    max_side: int = 960,
    quality: int = 80,
) -> str | None:
    """Write rust_locus_{year}_{stem}.jpg → basename or None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = locus_preview_name(year, path.name)
    out = out_dir / name
    if out.is_file() and not force:
        return name
    vis, info = render_rust_locus(
        path, year=year, color=color, precise=precise, max_side=max_side
    )
    if vis is None:
        return None
    cv2.imwrite(str(out), vis, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return name if info.get("ok") else name


def _geom_obj_from_dict(d: dict) -> RustGeometry:
    return RustGeometry(
        tip_frac=d["tip_frac"],
        d_plus=d["d_plus"],
        d_minus=d["d_minus"],
        edge_ratio=d["edge_ratio"],
        aspect=d["aspect"],
        center_n=tuple(d["center_n"]),
        span_n=tuple(d["span_n"]),
    )


def score_pair(q_sig: dict, r_sig: dict) -> dict:
    if r_sig.get("patch") is None or r_sig.get("geom") is None:
        return {
            "score": -1.0,
            "ncc": None,
            "reject": r_sig.get("reject", "no_sig"),
        }
    match = best_rust_patch_match(q_sig["patch"], r_sig["patch"])
    gq = q_sig.get("geom_obj") or _geom_obj_from_dict(q_sig["geom"])
    gr = _geom_obj_from_dict(r_sig["geom"])
    geom = compare_rust_geometry(
        gq,
        gr,
        flip_along=bool(match["flip_along"]),
        flip_across=bool(match["flip_across"]),
    )
    ncc = match["ncc"]
    if ncc is None:
        return {
            "score": -1.0,
            "ncc": None,
            "reject": "ncc_fail",
            "match": match,
            "geom": geom,
        }
    score = (
        float(ncc)
        - 0.35 * geom["tip_delta"]
        - 0.25 * geom["edge_delta"]
        - 0.10 * min(geom["aspect_delta"], 1.0)
    )
    return {
        "score": score,
        "ncc": ncc,
        "reject": None,
        "match": match,
        "geom": geom,
    }


def search_rust_gallery(
    *,
    data_root: Path,
    query_name: str,
    query_year: str = "2017",
    gallery_year: str = "2018",
    top_k: int = 8,
    limit: int = 0,
    use_cache: bool = True,
    cache_root: Path | None = None,
    preview_dir: Path | None = None,
    gold_name: str = "DJI_0190.JPG",
    log_print: bool = False,
) -> dict:
    """Run rust↔rust search. Returns JSON-serializable summary (+ preview paths)."""
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
    q_sig = build_sig_from_image(q_path, year=query_year)
    if q_sig is None or q_sig.get("patch") is None:
        why = (q_sig or {}).get("reject") or "few_spots/mask"
        raise RuntimeError(f"query rust signature failed ({why})")

    if log_print:
        print(
            f"query {query_year}/{q_path.name} → {gallery_year} usable×{len(refs)} "
            f"(pool≈{filt.get('usable')}) spots={q_sig['n_spots']}"
        )

    ranked: list[dict] = []
    hit = miss = skip = mem_hit = 0
    for i, path in enumerate(refs, 1):
        r_sig, src = resolve_gallery_sig(
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

        sc = score_pair(q_sig, r_sig)
        ranked.append(
            {
                "name": r_sig["name"],
                "score": sc["score"],
                "ncc": sc.get("ncc"),
                "reject": sc.get("reject"),
                "geom": sc.get("geom"),
                "n_spots": r_sig.get("n_spots"),
                "match": {
                    "flip_along": sc["match"]["flip_along"],
                    "flip_across": sc["match"]["flip_across"],
                    "angle_deg": sc["match"]["angle_deg"],
                }
                if sc.get("match")
                else None,
                "_ref_gray": sc["match"]["ref_gray"] if sc.get("match") else None,
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
    locus_q_name: str | None = None
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        q_stem = Path(query_name).stem
        # query provenance on original frame
        locus_q_name = write_rust_locus(
            q_path,
            year=query_year,
            out_dir=preview_dir,
            color=(0, 220, 255),
            force=True,
            precise=True,
        )
        q_locus_bgr = None
        if locus_q_name and (preview_dir / locus_q_name).is_file():
            q_locus_bgr = cv2.imread(str(preview_dir / locus_q_name))
        elif q_sig.get("bgr") is not None and q_sig.get("const") is not None:
            q_locus_bgr = draw_rust_provenance(
                q_sig["bgr"],
                q_sig["const"],
                atlas=q_sig.get("atlas"),
                patch=q_sig.get("patch"),
                color=(0, 220, 255),
                label=f"{query_year}/{Path(query_name).name}",
            )

        for i, row in enumerate(top, 1):
            if row.get("_ref_gray") is None:
                continue
            r_path = data_root / gallery_year / Path(row["name"]).name
            locus_r_name = write_rust_locus(
                r_path, year=gallery_year, out_dir=preview_dir, color=(0, 255, 120)
            )
            r_locus_bgr = None
            if locus_r_name and (preview_dir / locus_r_name).is_file():
                r_locus_bgr = cv2.imread(str(preview_dir / locus_r_name))
            panel = compose_rust_pair_preview(
                q_sig["patch"],
                row["_ref_gray"],
                q_title=f"Q {Path(query_name).name}",
                r_title=f"R#{i} {row['name']}",
                ncc=row["ncc"],
                geom=row.get("geom"),
                angle_deg=float((row.get("match") or {}).get("angle_deg") or 0),
                flip_along=bool((row.get("match") or {}).get("flip_along")),
                flip_across=bool((row.get("match") or {}).get("flip_across")),
                q_locus=q_locus_bgr,
                r_locus=r_locus_bgr,
            )
            out = preview_dir / f"rust_{q_stem}__{Path(row['name']).stem}.jpg"
            cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            row["preview"] = out.name
            row["locus_q"] = locus_q_name
            row["locus_r"] = locus_r_name
            preview_paths[row["name"]] = str(out)
        # convenience aliases
        if top and top[0].get("preview"):
            alias = preview_dir / "rust_search_top1.jpg"
            src = preview_dir / top[0]["preview"]
            if src.is_file():
                alias.write_bytes(src.read_bytes())
                preview_paths["top1"] = str(alias)

    summary = {
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
        "query_geom": q_sig.get("geom"),
        "query_spots": q_sig.get("n_spots"),
        "locus_q": locus_q_name,
        "direction_note": (
            "overcast query → sunny gallery: rust contrast higher under overcast"
        ),
        "top": [
            {
                "rank": i,
                "name": r["name"],
                "ref_id": r["name"],
                "query_id": Path(query_name).name,
                "score": r["score"],
                "ncc": r["ncc"],
                "reject": r["reject"],
                "geom": r["geom"],
                "match": r["match"],
                "n_spots": r.get("n_spots"),
                "preview": r.get("preview"),
                "locus_q": r.get("locus_q") or locus_q_name,
                "locus_r": r.get("locus_r"),
            }
            for i, r in enumerate(top, 1)
        ],
    }
    return summary
