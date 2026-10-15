#!/usr/bin/env python3
"""Sync local DTU years to S3 prefixes 2017/ and 2018/.

Examples:
  export BLADE_S3_BUCKET=blade-revisit-xxx
  python scripts/sync_dtu_to_s3.py --dry-run
  python scripts/sync_dtu_to_s3.py
  python scripts/sync_dtu_to_s3.py --usable-only   # only usable near shots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.s3_io import bucket_from_env, client, sync_dir_to_prefix  # noqa: E402


def _usable_names(year: str, cache_path: Path) -> set[str] | None:
    if not cache_path.is_file():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = data.get(year)
    if not isinstance(entry, dict) or not isinstance(entry.get("usable"), list):
        return None
    return set(entry["usable"])


def main() -> None:
    p = argparse.ArgumentParser(description="Upload DTU 2017/2018 JPGs to S3")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "data" / "wind-turbine-dtu",
    )
    p.add_argument("--bucket", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument(
        "--years",
        nargs="+",
        default=["2017", "2018"],
        help="Year folder names / S3 prefixes",
    )
    p.add_argument(
        "--usable-only",
        action="store_true",
        help="Only upload names listed in data/usable_shots_cache.json",
    )
    p.add_argument(
        "--usable-cache",
        type=Path,
        default=ROOT / "data" / "usable_shots_cache.json",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    bucket = bucket_from_env(args.bucket)
    s3 = client(args.region)
    total = 0
    for year in args.years:
        local = args.data_root / year
        names = None
        if args.usable_only:
            names = _usable_names(year, args.usable_cache)
            if names is None:
                raise SystemExit(
                    f"usable cache missing for {year}: {args.usable_cache} "
                    "(run usable filter once locally, or drop --usable-only)"
                )
            print(f"{year}: usable-only ×{len(names)}")
        n = sync_dir_to_prefix(
            s3,
            bucket=bucket,
            local_dir=local,
            prefix=year,
            names=names,
            dry_run=args.dry_run,
        )
        print(f"{year}: {n} files")
        total += n
    print(f"done total={total} bucket={bucket}")


if __name__ == "__main__":
    main()
