#!/usr/bin/env python3
"""CLI wrapper: 2017 rust query → 2018 usable gallery (see blade.rust_search)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.rust_search import search_rust_gallery  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="0752 rust → search 2018 usable")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "wind-turbine-dtu")
    ap.add_argument("--query", default="DJI_0752.JPG")
    ap.add_argument("--query-year", default="2017")
    ap.add_argument("--gallery-year", default="2018")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=all usable")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "debug_preview"
    summary = search_rust_gallery(
        data_root=args.data_root,
        query_name=args.query,
        query_year=args.query_year,
        gallery_year=args.gallery_year,
        top_k=args.top_k,
        limit=args.limit,
        use_cache=not args.no_cache,
        preview_dir=out_dir,
        log_print=True,
    )
    sum_path = out_dir / "rust_search_0752_in_2018.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTop-K:")
    gold = summary.get("gold_name")
    for row in summary["top"]:
        mark = " ← GOLD" if row["name"] == gold else ""
        ncc_s = f"{row['ncc']:.3f}" if row["ncc"] is not None else "n/a"
        print(f"  #{row['rank']} {row['name']} score={row['score']:.3f} ncc={ncc_s}{mark}")
    print(f"gold {gold} rank: {summary.get('gold_rank')}")
    print(f"wrote {sum_path}  ({summary.get('secs')}s)")


if __name__ == "__main__":
    main()
