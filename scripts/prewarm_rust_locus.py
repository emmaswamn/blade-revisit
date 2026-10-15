#!/usr/bin/env python3
"""Pre-render rust_locus_*.jpg into data/debug_preview/ for GET /api/rust_locus.

Default: unique images from eval/pairs_15.json (2017 query + 2018 ref).
Uses BLADE_VIEW_MAX_SIDE / BLADE_VIEW_JPEG_QUALITY (default 960 / 80).
Existing larger cache files are kept until --force.

Examples:
  python scripts/prewarm_rust_locus.py
  python scripts/prewarm_rust_locus.py --year 2018 --name DJI_0253.JPG --force
  python scripts/prewarm_rust_locus.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.rust_search import locus_preview_name, write_rust_locus  # noqa: E402

DEFAULT_PAIRS = ROOT / "eval" / "pairs_15.json"
DEFAULT_OUT = ROOT / "data" / "debug_preview"


def _jobs_from_pairs(pairs_path: Path) -> list[tuple[str, str]]:
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    jobs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for p in data.get("pairs") or []:
        for year, key in (("2017", "query_2017"), ("2018", "ref_2018")):
            name = Path(p[key]).name
            item = (year, name)
            if item not in seen:
                seen.add(item)
                jobs.append(item)
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Prewarm rust locus preview JPEGs")
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
    ap.add_argument("--year", choices=("2017", "2018"), help="Single image mode")
    ap.add_argument("--name", help="Single image filename (with --year)")
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
    ap.add_argument(
        "--precise",
        action="store_true",
        help="Rebuild atlas crop (slow). Default uses rust_sig spots_xy.",
    )
    ap.add_argument("--force", action="store_true", help="Rewrite even if cache exists")
    args = ap.parse_args()

    if args.year or args.name:
        if not (args.year and args.name):
            raise SystemExit("--year and --name must be used together")
        jobs = [(args.year, Path(args.name).name)]
    else:
        if not args.pairs.is_file():
            raise SystemExit(f"pairs file missing: {args.pairs}")
        jobs = _jobs_from_pairs(args.pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    t0 = time.time()
    print(
        f"prewarm rust_locus: {len(jobs)} images "
        f"(max_side={args.max_side} q={args.quality} precise={args.precise}) "
        f"→ {args.out_dir}"
    )
    for i, (year, name) in enumerate(jobs, 1):
        src = args.data_root / year / name
        out_name = locus_preview_name(year, name)
        out = args.out_dir / out_name
        if not src.is_file():
            fail += 1
            print(f"  [{i}/{len(jobs)}] MISS {year}/{name}")
            continue
        fresh = out.is_file() and not args.force
        if fresh:
            skip += 1
            print(f"  [{i}/{len(jobs)}] skip {out_name}")
            continue
        written = write_rust_locus(
            src,
            year=year,
            out_dir=args.out_dir,
            color=(0, 220, 255) if year == "2017" else (0, 255, 120),
            force=True,
            precise=args.precise,
            max_side=args.max_side,
            quality=args.quality,
        )
        if not written or not (args.out_dir / written).is_file():
            fail += 1
            print(f"  [{i}/{len(jobs)}] FAIL {year}/{name}")
            continue
        ok += 1
        size_kb = (args.out_dir / written).stat().st_size / 1024
        print(f"  [{i}/{len(jobs)}] wrote {written}  ({size_kb:.0f} KiB)")

    print(
        f"done wrote={ok} skip={skip} fail={fail} "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
