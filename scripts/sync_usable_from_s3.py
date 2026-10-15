#!/usr/bin/env python3
"""Pull usable near-shot JPGs from S3 onto local/EBS for low-latency API reads.

Uses data/usable_shots_cache.json name lists when present; otherwise downloads
the whole year prefix.

Examples:
  export BLADE_S3_BUCKET=blade-revisit-xxx
  python scripts/sync_usable_from_s3.py --data-root /data/wind-turbine-dtu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.s3_io import bucket_from_env, client, sync_prefix_to_dir  # noqa: E402


def _usable_names(year: str, cache_path: Path) -> set[str] | None:
    if not cache_path.is_file():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = data.get(year)
    if not isinstance(entry, dict) or not isinstance(entry.get("usable"), list):
        return None
    return set(entry["usable"])


def main() -> None:
    p = argparse.ArgumentParser(description="Download usable JPGs from S3 to EBS/local")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os_environ_data_root()),
    )
    p.add_argument("--bucket", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--years", nargs="+", default=["2017", "2018"])
    p.add_argument(
        "--usable-cache",
        type=Path,
        default=ROOT / "data" / "usable_shots_cache.json",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download entire year prefix (ignore usable cache)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    bucket = bucket_from_env(args.bucket)
    s3 = client(args.region)
    total = 0
    for year in args.years:
        names = None if args.all else _usable_names(year, args.usable_cache)
        if names is None and not args.all:
            print(
                f"{year}: no usable cache → downloading full prefix "
                f"(pass --all to silence, or ship usable_shots_cache.json)"
            )
        dest = args.data_root / year
        n = sync_prefix_to_dir(
            s3,
            bucket=bucket,
            prefix=year,
            local_dir=dest,
            names=names,
            dry_run=args.dry_run,
        )
        print(f"{year}: {n} files → {dest}")
        total += n
    print(f"done total={total} data_root={args.data_root}")


def os_environ_data_root() -> str:
    import os

    return os.environ.get(
        "BLADE_DATA_ROOT", str(Path.home() / "data" / "wind-turbine-dtu")
    )


if __name__ == "__main__":
    main()
