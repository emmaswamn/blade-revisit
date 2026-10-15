#!/usr/bin/env python3
"""Pre-render stage1 display JPEGs into data/debug_preview/ for the review API.

Default: 15 eval pairs × (original, lighting, overlay, kps_masked) on both years.
API /api/stage1/view and /api/image then serve disk cache without recompute.

Examples:
  python scripts/prewarm_stage1.py
  python scripts/prewarm_stage1.py --views lighting,overlay --force
  python scripts/prewarm_stage1.py --year 2017 --name DJI_0752.JPG --views all
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

from blade.config import load_thresholds  # noqa: E402
from blade.stage1_views import (  # noqa: E402
    REVIEW_PREWARM_VIEWS,
    STAGE1_VIEWS,
    write_stage1_view,
)

DEFAULT_PAIRS = ROOT / "eval" / "pairs_15.json"
DEFAULT_OUT = ROOT / "data" / "debug_preview"
DEFAULT_CONFIG = ROOT / "configs" / "thresholds.yaml"


def _parse_views(raw: str) -> list[str]:
    text = (raw or "review").strip().lower()
    if text in ("review", "default"):
        return list(REVIEW_PREWARM_VIEWS)
    if text == "all":
        return sorted(STAGE1_VIEWS)
    views = [v.strip() for v in text.split(",") if v.strip()]
    bad = [v for v in views if v not in STAGE1_VIEWS]
    if bad:
        raise SystemExit(f"unknown views: {bad}; choose from {sorted(STAGE1_VIEWS)}")
    return views


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
    ap = argparse.ArgumentParser(description="Prewarm stage1 preview JPEGs")
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
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--views",
        default="review",
        help="review | all | comma list (default: review set)",
    )
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
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    views = _parse_views(args.views)
    if args.year or args.name:
        if not (args.year and args.name):
            raise SystemExit("--year and --name must be used together")
        jobs = [(args.year, Path(args.name).name)]
    else:
        if not args.pairs.is_file():
            raise SystemExit(f"pairs file missing: {args.pairs}")
        jobs = _jobs_from_pairs(args.pairs)

    thresholds = load_thresholds(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    t0 = time.time()
    total = len(jobs) * len(views)
    n = 0
    print(
        f"prewarm stage1: {len(jobs)} images × {len(views)} views "
        f"(max_side={args.max_side} q={args.quality}) → {args.out_dir}"
    )
    for year, name in jobs:
        src = args.data_root / year / name
        if not src.is_file():
            print(f"  MISS {year}/{name}")
            fail += len(views)
            n += len(views)
            continue
        for view in views:
            n += 1
            out = args.out_dir / f"stage1_{year}_{Path(name).stem}_{view}.jpg"
            fresh = (
                not args.force
                and out.is_file()
                and out.stat().st_mtime >= src.stat().st_mtime
            )
            try:
                path = write_stage1_view(
                    src,
                    year=year,
                    view=view,
                    out_dir=args.out_dir,
                    thresholds=thresholds,
                    max_side=args.max_side,
                    quality=args.quality,
                    force=args.force,
                )
            except Exception as exc:  # noqa: BLE001 — batch continue
                fail += 1
                print(f"  FAIL {year}/{name} view={view}: {exc}")
                continue
            if fresh:
                skip += 1
                tag = "skip"
            else:
                ok += 1
                tag = "wrote"
            if n % 10 == 0 or n == total or tag == "wrote":
                print(f"  [{n}/{total}] {tag} {path.name}")

    print(
        f"done wrote={ok} skip={skip} fail={fail} "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
