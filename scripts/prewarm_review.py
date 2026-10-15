#!/usr/bin/env python3
"""Regenerate all review-page images for eval/pairs_15.json.

Covers:
  - stage1: original / lighting / overlay / kps_masked
  - rust locus: rust_locus_{year}_{stem}.jpg
  - rust pair panel: rust_{qStem}__{rStem}.jpg (match or reject stub)

Examples:
  python scripts/prewarm_review.py --force
  docker compose exec api python scripts/prewarm_review.py --force
  python scripts/prewarm_review.py --only pairs   # only rust_*__*.jpg
  python scripts/prewarm_review.py --only stage1,locus
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.rust_search import write_rust_eval_pair_preview  # noqa: E402

DEFAULT_PAIRS = ROOT / "eval" / "pairs_15.json"
DEFAULT_OUT = ROOT / "data" / "debug_preview"


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prewarm all /review images for 15 pairs")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "BLADE_DATA_ROOT", Path.home() / "data" / "wind-turbine-dtu"
            )
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument(
        "--only",
        default="stage1,locus,pairs",
        help="Comma list: stage1,locus,pairs (default all)",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=int(os.environ.get("BLADE_VIEW_MAX_SIDE", "960")),
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=int(os.environ.get("BLADE_VIEW_JPEG_QUALITY", "80")),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    parts = {p.strip() for p in args.only.split(",") if p.strip()}
    bad = parts - {"stage1", "locus", "pairs"}
    if bad:
        raise SystemExit(f"unknown --only parts: {bad}")
    if not args.pairs.is_file():
        raise SystemExit(f"pairs missing: {args.pairs}")

    py = sys.executable
    t0 = time.time()

    if "stage1" in parts:
        cmd = [
            py,
            str(ROOT / "scripts" / "prewarm_stage1.py"),
            "--data-root",
            str(args.data_root),
            "--out-dir",
            str(args.out_dir),
            "--pairs",
            str(args.pairs),
            "--max-side",
            str(args.max_side),
            "--quality",
            str(args.quality),
            "--views",
            "review",
        ]
        if args.force:
            cmd.append("--force")
        if _run(cmd) != 0:
            raise SystemExit("stage1 prewarm failed")

    if "locus" in parts:
        cmd = [
            py,
            str(ROOT / "scripts" / "prewarm_rust_locus.py"),
            "--data-root",
            str(args.data_root),
            "--out-dir",
            str(args.out_dir),
            "--pairs",
            str(args.pairs),
            "--max-side",
            str(args.max_side),
            "--quality",
            str(args.quality),
        ]
        if args.force:
            cmd.append("--force")
        if _run(cmd) != 0:
            raise SystemExit("locus prewarm failed")

    if "pairs" in parts:
        data = json.loads(args.pairs.read_text(encoding="utf-8"))
        pairs = data.get("pairs") or []
        print(
            f"prewarm pair panels: {len(pairs)} "
            f"(max_side={args.max_side} q={args.quality}) → {args.out_dir}"
        )
        ok = stub = skip = fail = 0
        for i, p in enumerate(pairs, 1):
            q = Path(p["query_2017"]).name
            r = Path(p["ref_2018"]).name
            info = write_rust_eval_pair_preview(
                data_root=args.data_root,
                query_name=q,
                ref_name=r,
                out_dir=args.out_dir,
                force=args.force,
                max_side=args.max_side,
                quality=args.quality,
            )
            st = info.get("status")
            if st == "skip":
                skip += 1
            elif st == "wrote":
                ok += 1
            elif st == "reject_stub":
                stub += 1
            else:
                fail += 1
            size = ""
            path = info.get("path")
            if path and Path(path).is_file():
                size = f"  ({Path(path).stat().st_size / 1024:.0f} KiB)"
            print(
                f"  [{i}/{len(pairs)}] {p.get('id')} {st} "
                f"{info.get('preview')}{size}"
                + (f" reject={info.get('reject')}" if info.get("reject") else "")
            )
        print(f"pairs done wrote={ok} stub={stub} skip={skip} fail={fail}")

    print(f"prewarm_review finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
