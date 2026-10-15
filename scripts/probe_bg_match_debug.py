#!/usr/bin/env python3
"""Debug: annotate matched blocks + texture crops for a query/ref pair.

Example:
  .venv/bin/python scripts/probe_bg_match_debug.py \\
      --query DJI_0190.JPG --ref DJI_0593.JPG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.background import build_bg_match_texture_debug  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="bg match texture debug panel")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "wind-turbine-dtu")
    ap.add_argument("--query", default="DJI_0190.JPG")
    ap.add_argument("--ref", default="DJI_0593.JPG")
    ap.add_argument("--query-year", default="2018")
    ap.add_argument("--gallery-year", default="2017")
    args = ap.parse_args()

    panel, vote = build_bg_match_texture_debug(
        data_root=args.data_root,
        query_name=args.query,
        ref_name=args.ref,
        query_year=args.query_year,
        gallery_year=args.gallery_year,
    )
    out_dir = ROOT / "data" / "debug_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_q = Path(args.query).stem
    stem_r = Path(args.ref).stem
    out = out_dir / f"bg_match_tex_{stem_q}_vs_{stem_r}.jpg"
    cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    meta = {
        "query": f"{args.query_year}/{args.query}",
        "ref": f"{args.gallery_year}/{args.ref}",
        "score": vote.get("score"),
        "hits": vote.get("hits"),
        "adj_score": vote.get("adj_score"),
        "pairs": vote.get("pairs"),
        "adj_pairs": vote.get("adj_pairs"),
        "preview": out.name,
    }
    meta_path = out_dir / f"bg_match_tex_{stem_q}_vs_{stem_r}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
