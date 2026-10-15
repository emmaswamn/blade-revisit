#!/usr/bin/env python3
"""Local retrieval: 2018 query → search 2017 gallery.

Example:
  python worker/run_local.py --query DJI_0004.JPG
  python worker/run_local.py --query-limit 3 --top-k 5 --ref-limit 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.config import load_thresholds
from blade.retrieve import build_gallery, list_images, search_query_in_2017
from blade.store import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="2018→2017 Top-k retrieval (local)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "data" / "wind-turbine-dtu",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "thresholds.yaml")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "results.db")
    parser.add_argument("--overlay-dir", type=Path, default=ROOT / "data" / "overlays")
    parser.add_argument("--query", type=str, default=None, help="Single 2018 filename")
    parser.add_argument("--query-limit", type=int, default=1, help="If no --query, take first N")
    parser.add_argument("--top-k", type=int, default=5, help="Store top-k 2017 refs per query")
    parser.add_argument(
        "--ref-limit",
        type=int,
        default=0,
        help="Limit within 2017 usable pool (0=all usable ≈77)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Use all 2017 files instead of usable near-shot pool",
    )
    args = parser.parse_args()

    year2017 = args.data_root / "2017"
    year2018 = args.data_root / "2018"
    if not year2017.is_dir() or not year2018.is_dir():
        raise SystemExit(f"Expected {year2017} and {year2018}")

    thresholds = load_thresholds(args.config)
    store = ResultStore(args.db)

    queries = list_images(year2018)
    if args.query:
        queries = [year2018 / args.query]
        if not queries[0].is_file():
            raise SystemExit(f"Query not found: {queries[0]}")
    else:
        queries = queries[: args.query_limit]

    print(f"OpenCV {cv2.__version__} | detector={thresholds.detector} | max_side={thresholds.max_side}")
    print(f"blur_thr={thresholds.blur_laplacian_min} | min_query_keypoints={thresholds.min_query_keypoints}")
    print(f"Queries 2018: {[q.name for q in queries]}")

    gallery = build_gallery(
        year2017,
        thresholds,
        ref_limit=args.ref_limit,
        usable_only=not args.all_files,
        log_print=True,
    )

    for qpath in queries:
        search_query_in_2017(
            query_path=qpath,
            gallery=gallery,
            thresholds=thresholds,
            store=store,
            overlay_dir=args.overlay_dir,
            top_k=args.top_k,
        )

    store.close()
    print(f"\nDB → {args.db}")
    print("Debug UI: uvicorn api.main:app --app-dir . --reload --port 8765")


if __name__ == "__main__":
    main()
