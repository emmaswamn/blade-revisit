#!/usr/bin/env python3
"""v2 bg blocks debug: color+edge regions on 0190 / 0752.

Writes data/debug_preview/bg_blocks_*.jpg for eyeball check.
See docs/background-match.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.background import draw_bg_blocks_debug, extract_bg_blocks  # noqa: E402


def _run_one(path: Path, year: str, out_dir: Path) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    res = extract_bg_blocks(path.name, bgr, year=year)
    vis = draw_bg_blocks_debug(res)
    stem = Path(path.name).stem
    out = out_dir / f"bg_blocks_{stem}.jpg"
    cv2.imwrite(str(out), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    anchors = [
        {
            "idx": b.idx,
            "label": b.label,
            "zone": f"{'U' if b.cy < 0.33 else ('D' if b.cy > 0.66 else 'M')}"
            f"{'L' if b.cx < 0.33 else ('R' if b.cx > 0.66 else 'M')}",
            "area": b.area,
            "cx": round(b.cx, 3),
            "cy": round(b.cy, 3),
            "edge": round(b.edge_mean, 1),
            "hsv": [round(b.mean_h, 1), round(b.mean_s, 1), round(b.mean_v, 1)],
        }
        for b in res.anchors
    ]
    print(f"{year}/{path.name}  blocks={len(res.blocks)} anchors={len(res.anchors)}")
    for a in anchors:
        print(
            f"  #{a['idx']} {a['label']:9s} {a['zone']}  "
            f"area={a['area']} edge={a['edge']} hsv={a['hsv']}"
        )
    print(f"  -> {out}")
    return {
        "image": f"{year}/{path.name}",
        "bg_frac": round(res.bg_frac, 3),
        "n_blocks": len(res.blocks),
        "anchors": anchors,
        "preview": out.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="bg color+edge blocks debug")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "wind-turbine-dtu")
    args = ap.parse_args()
    out_dir = ROOT / "data" / "debug_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        (args.data_root / "2018" / "DJI_0190.JPG", "2018"),
        (args.data_root / "2017" / "DJI_0752.JPG", "2017"),
    ]
    summary = [_run_one(p, y, out_dir) for p, y in pairs]
    meta = out_dir / "bg_blocks_0190_0752.json"
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {meta}")


if __name__ == "__main__":
    main()
