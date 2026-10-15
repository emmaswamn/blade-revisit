#!/usr/bin/env python3
"""S3-backed retrieval: 2018 query → search 2017 gallery → DB + overlays on S3.

I/O only — same blade.retrieve / ResultStore as run_local.py.

Examples:
  export BLADE_S3_BUCKET=blade-revisit-xxx
  export BLADE_DATA_ROOT=/data/wind-turbine-dtu   # EBS cache preferred
  python worker/run_s3.py --query DJI_0190.JPG --ref-limit 30
  python worker/run_s3.py --list-only             # smoke: list S3 keys
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.config import load_thresholds
from blade.retrieve import build_gallery, list_images, search_query_in_2017
from blade.s3_io import (
    bucket_from_env,
    client,
    download_file,
    list_keys,
    overlay_key,
    s3_uri,
    upload_file,
    year_key,
)
from blade.store import ResultStore


def _ensure_local(
    s3,
    *,
    bucket: str,
    year: str,
    name: str,
    data_root: Path,
    cache_dir: Path,
) -> Path:
    """Prefer EBS/data_root; else download once into cache_dir."""
    local = data_root / year / name
    if local.is_file():
        return local
    dest = cache_dir / year / name
    if dest.is_file():
        return dest
    key = year_key(year, name)
    print(f"s3 get {key}")
    return download_file(s3, bucket=bucket, key=key, dest=dest)


def _upload_overlays(
    s3,
    *,
    bucket: str,
    overlay_dir: Path,
    dry_run: bool = False,
) -> int:
    n = 0
    if not overlay_dir.is_dir():
        return 0
    for p in sorted(overlay_dir.glob("*.jpg")):
        key = overlay_key(p.name)
        if dry_run:
            print(f"would upload overlay {p.name} -> {s3_uri(bucket, key)}")
        else:
            uri = upload_file(
                s3, bucket=bucket, key=key, path=p, content_type="image/jpeg"
            )
            print(f"overlay {uri}")
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="2018→2017 Top-k via S3 + local cache")
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("BLADE_DATA_ROOT", Path.home() / "data" / "wind-turbine-dtu")
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get("BLADE_S3_CACHE", str(ROOT / "data" / "s3_cache"))
        ),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "thresholds.yaml")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("BLADE_DB", str(ROOT / "data" / "results.db"))),
    )
    parser.add_argument("--overlay-dir", type=Path, default=ROOT / "data" / "overlays")
    parser.add_argument("--query", type=str, default=None, help="2018 filename")
    parser.add_argument("--query-limit", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ref-limit", type=int, default=0)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Gallery = all 2017 files (not usable-only)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List S3 2017/2018 keys and exit (connectivity smoke)",
    )
    parser.add_argument(
        "--skip-overlay-upload",
        action="store_true",
        help="Keep overlays only on local disk",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bucket = bucket_from_env(args.bucket)
    s3 = client(args.region)

    print(f"OpenCV {cv2.__version__}")
    print(f"bucket={bucket} data_root={args.data_root}")

    if args.list_only:
        for year in ("2017", "2018", "overlays"):
            keys = list_keys(s3, bucket=bucket, prefix=year)
            print(f"{year}/ ×{len(keys)}")
            for k in keys[:5]:
                print(f"  {k}")
            if len(keys) > 5:
                print(f"  … +{len(keys) - 5}")
        return

    # Prefer EBS tree; materialize missing query/refs from S3 into cache.
    year2017 = args.data_root / "2017"
    year2018 = args.data_root / "2018"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not year2018.is_dir() or not any(year2018.glob("*.JPG")):
        # Bootstrap year lists from S3 into cache, then point data_root at cache.
        print("local 2018 empty → using S3 cache layout under", args.cache_dir)
        for year in ("2017", "2018"):
            keys = list_keys(s3, bucket=bucket, prefix=year)
            for key in keys:
                name = Path(key).name
                dest = args.cache_dir / year / name
                if not dest.is_file():
                    download_file(s3, bucket=bucket, key=key, dest=dest)
        data_root = args.cache_dir
        year2017 = data_root / "2017"
        year2018 = data_root / "2018"
    else:
        data_root = args.data_root

    thresholds = load_thresholds(args.config)
    store = ResultStore(args.db)

    queries = list_images(year2018)
    if args.query:
        qpath = year2018 / args.query
        if not qpath.is_file():
            qpath = _ensure_local(
                s3,
                bucket=bucket,
                year="2018",
                name=args.query,
                data_root=data_root,
                cache_dir=args.cache_dir,
            )
        queries = [qpath]
    else:
        queries = queries[: args.query_limit]

    print(
        f"detector={thresholds.detector} max_side={thresholds.max_side} "
        f"queries={[q.name for q in queries]}"
    )

    if args.dry_run:
        print("dry-run: skip gallery build / search")
        store.close()
        return

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

    if not args.skip_overlay_upload:
        n = _upload_overlays(
            s3, bucket=bucket, overlay_dir=args.overlay_dir, dry_run=False
        )
        print(f"overlays uploaded ×{n}")

    print(f"\nDB → {args.db}")
    print("UI: uvicorn api.main:app --app-dir . --host 0.0.0.0 --port 8765")


if __name__ == "__main__":
    main()
