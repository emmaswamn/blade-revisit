#!/usr/bin/env python3
"""Gold-pair rust↔rust: crop four-spot neighborhood, straighten, compare.

Main evidence (not whole-blade atlas):
  1) interior ORB density → 4 rust loci
  2) local blade-frame crop around them (axis horizontal)
  3) flips + small rotation → patch NCC
  4) edge_ratio / tip_frac / aspect — proportions to both rims

Writes:
  data/debug_preview/rust_vs_rust_0752_0190.jpg   ← look here first
  data/debug_preview/rust4_0752.jpg
  data/debug_preview/rust4_0190.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.blade_atlas import (  # noqa: E402
    best_rust_patch_match,
    build_blade_atlas,
    compare_rust_geometry,
    compose_rust_pair_preview,
    detect_rust_constellation,
    draw_constellation_on_bgr,
    extract_rust_patch,
    measure_rust_geometry,
)


def _thumb(bgr, max_side: int = 1280):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    s = max_side / m
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main() -> None:
    data = Path.home() / "data" / "wind-turbine-dtu"
    out_dir = ROOT / "data" / "debug_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = data / "2017" / "DJI_0752.JPG"
    r_path = data / "2018" / "DJI_0190.JPG"
    qb = cv2.imread(str(q_path))
    rb = cv2.imread(str(r_path))
    if qb is None or rb is None:
        raise SystemExit(f"missing images under {data}")

    print("0752 → 0190  rust↔rust + edge proportions")
    q_atlas = build_blade_atlas(q_path.name, qb, year="2017")
    r_atlas = build_blade_atlas(r_path.name, rb, year="2018")
    if q_atlas is None or r_atlas is None:
        raise SystemExit("atlas/mask failed")

    q_c = detect_rust_constellation(qb, q_atlas)
    r_c = detect_rust_constellation(rb, r_atlas)
    print(f"0752 spots={len(q_c.spots)} via {q_c.method} (kps={q_c.n_interior})")
    print(f"0190 spots={len(r_c.spots)} via {r_c.method} (kps={r_c.n_interior})")
    if len(q_c.spots) < 3 or len(r_c.spots) < 3:
        raise SystemExit("need ≥3 rust loci each")

    gq = measure_rust_geometry(q_c)
    gr = measure_rust_geometry(r_c)
    assert gq is not None and gr is not None
    print(
        f"0752 geom tip_frac={gq.tip_frac:.3f} edge_ratio={gq.edge_ratio:.3f} "
        f"d+/d-={gq.d_plus:.3f}/{gq.d_minus:.3f} aspect={gq.aspect:.2f}"
    )
    print(
        f"0190 geom tip_frac={gr.tip_frac:.3f} edge_ratio={gr.edge_ratio:.3f} "
        f"d+/d-={gr.d_plus:.3f}/{gr.d_minus:.3f} aspect={gr.aspect:.2f}"
    )

    q_patch = extract_rust_patch(qb, q_atlas, q_c, out_w=280, out_h=280, margin=2.4)
    r_patch = extract_rust_patch(rb, r_atlas, r_c, out_w=280, out_h=280, margin=2.4)
    if q_patch is None or r_patch is None:
        raise SystemExit("rust patch extract failed")

    match = best_rust_patch_match(q_patch, r_patch)
    geom = compare_rust_geometry(
        gq,
        gr,
        flip_along=bool(match["flip_along"]),
        flip_across=bool(match["flip_across"]),
    )
    print(
        f"rust NCC={match['ncc']} rot={match['angle_deg']} "
        f"fa={match['flip_along']} fc={match['flip_across']}"
    )
    print(
        f"geom Δ tip={geom['tip_delta']:.3f} edge={geom['edge_delta']:.3f} "
        f"aspect={geom['aspect_delta']:.3f}"
    )

    panel = compose_rust_pair_preview(
        q_patch,
        match["ref_gray"],
        q_title="Q 0752 rust crop",
        r_title="R 0190 rust crop (aligned)",
        ncc=match["ncc"],
        geom=geom,
        angle_deg=float(match["angle_deg"]),
        flip_along=bool(match["flip_along"]),
        flip_across=bool(match["flip_across"]),
    )
    main_out = out_dir / "rust_vs_rust_0752_0190.jpg"
    cv2.imwrite(str(main_out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    cv2.imwrite(
        str(out_dir / "rust4_0752.jpg"),
        _thumb(draw_constellation_on_bgr(qb, q_c, color=(0, 255, 255))),
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )
    cv2.imwrite(
        str(out_dir / "rust4_0190.jpg"),
        _thumb(draw_constellation_on_bgr(rb, r_c, color=(0, 255, 120))),
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )
    print(f"wrote {main_out}")
    print(f"wrote {out_dir / 'rust4_0752.jpg'}")
    print(f"wrote {out_dir / 'rust4_0190.jpg'}")


if __name__ == "__main__":
    main()
