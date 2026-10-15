#!/usr/bin/env python3
"""Background station prior: 2018 sunny query → 2017 usable gallery.

See docs/background-match.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.background import (  # noqa: E402
    search_bg_blocks_gallery,
    search_bg_gallery,
    search_bg_layout_gallery,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="2018 bg → search 2017 usable")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "wind-turbine-dtu")
    ap.add_argument("--query", default="DJI_0190.JPG")
    ap.add_argument("--query-year", default="2018")
    ap.add_argument("--gallery-year", default="2017")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=all usable")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--gold", default="DJI_0752.JPG")
    ap.add_argument(
        "--method",
        choices=("blocks", "hsv", "layout"),
        default="blocks",
        help="blocks=v2 vote; layout=v3 land lines; hsv=v1 grid",
    )
    args = ap.parse_args()

    out_dir = ROOT / "data" / "debug_preview"
    fn = {
        "blocks": search_bg_blocks_gallery,
        "hsv": search_bg_gallery,
        "layout": search_bg_layout_gallery,
    }[args.method]
    summary = fn(
        data_root=args.data_root,
        query_name=args.query,
        query_year=args.query_year,
        gallery_year=args.gallery_year,
        top_k=args.top_k,
        limit=args.limit,
        use_cache=not args.no_cache,
        preview_dir=out_dir,
        gold_name=args.gold,
        log_print=True,
    )
    stem = Path(args.query).stem
    sum_path = out_dir / f"bg_search_{stem}_in_{args.gallery_year}_{args.method}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTop-K:")
    gold = summary.get("gold_name")
    for row in summary["top"]:
        mark = " ← GOLD" if row["name"] == gold else ""
        if args.method == "blocks":
            print(
                f"  #{row['rank']} {row['name']} "
                f"hits={row.get('hits')} score={row['score']:.2f}{mark}"
            )
        elif args.method == "layout":
            print(
                f"  #{row['rank']} {row['name']} "
                f"score={row['score']:.3f} ch={row.get('chamfer')} "
                f"ori={row.get('ori')}{mark}"
            )
        else:
            print(f"  #{row['rank']} {row['name']} cos={row['score']:.3f}{mark}")
    print(f"gold {gold} rank: {summary.get('gold_rank')}")
    print(f"method={summary.get('method')} wrote {sum_path}  ({summary.get('secs')}s)")
    if summary.get("preview"):
        print(f"previews: {summary['preview']}")


if __name__ == "__main__":
    main()
