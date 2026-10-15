#!/usr/bin/env python3
"""Download DTU Nordtank drone JPGs from Mendeley (public API, file-by-file).

Dataset: https://data.mendeley.com/datasets/hd96prn3nc/1
Licence: CC BY 4.0 · DOI 10.17632/hd96prn3nc.1

Mendeley does not offer one clean zip for this set; this script lists each
folder via the public API and downloads with resume-friendly skip-by-size.

Examples:
  python scripts/download_dtu_mendeley.py
  python scripts/download_dtu_mendeley.py --out ~/data/wind-turbine-dtu
  python scripts/download_dtu_mendeley.py --year 2017
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

socket.setdefaulttimeout(40)

BASE = "https://data.mendeley.com/public-api/datasets/hd96prn3nc"
API_HEADERS = {
    "User-Agent": "blade-revisit/1.0 (research; +https://github.com/)",
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
}
DL_HEADERS = {"User-Agent": "blade-revisit/1.0 (research)"}

# Folder UUIDs under dataset version 1 (2017 / 2018 year folders).
FOLDERS = {
    "2017": "0b65f309-ff9c-48da-9f52-687990a5483e",
    "2018": "ec0873c4-56b9-4795-a94a-f0d48686de7d",
}


def get_json(url: str, *, retries: int = 5):
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=API_HEADERS)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * (i + 1))
    assert last is not None
    raise last


def download(url: str, dest: Path, *, retries: int = 4) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=DL_HEADERS)
            with urllib.request.urlopen(req, timeout=40) as r, part.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            part.unlink(missing_ok=True)
            time.sleep(2 * (i + 1))
    assert last is not None
    raise last


def main() -> None:
    default_out = Path(
        os.environ.get(
            "BLADE_DATA_ROOT", Path.home() / "data" / "wind-turbine-dtu"
        )
    )
    ap = argparse.ArgumentParser(
        description="Download DTU 2017/2018 inspection JPGs from Mendeley"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output root with 2017/ and 2018/ (default: {default_out})",
    )
    ap.add_argument(
        "--year",
        choices=("2017", "2018"),
        help="Only one year (default: both)",
    )
    ap.add_argument(
        "--version",
        type=int,
        default=1,
        help="Mendeley dataset version query param (default: 1)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="API page size (default: 1000, enough for each year)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List counts only, do not download",
    )
    args = ap.parse_args()

    years = (args.year,) if args.year else ("2017", "2018")
    ok = skip = bad = 0
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    for year in years:
        folder_id = FOLDERS[year]
        dest_dir = args.out / year
        url = (
            f"{BASE}/files?folder_id={folder_id}"
            f"&version={args.version}&$start=0&$limit={args.limit}"
        )
        files = get_json(url)
        if not isinstance(files, list):
            raise SystemExit(f"unexpected API payload for {year}: {type(files)}")
        print(f"{year}: {len(files)} files → {dest_dir}", flush=True)
        if args.dry_run:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        for n, item in enumerate(files, 1):
            name = item["filename"]
            dest = dest_dir / name
            size = int(item.get("size") or 0)
            if dest.is_file() and (size <= 0 or dest.stat().st_size == size):
                skip += 1
            else:
                if dest.is_file():
                    # Same filename, different size → keep both.
                    dest = dest_dir / (
                        f"{Path(name).stem}-{str(item.get('id', 'x'))[:8]}"
                        f"{Path(name).suffix}"
                    )
                try:
                    dl = item["content_details"]["download_url"]
                    download(dl, dest)
                    ok += 1
                except Exception as e:  # noqa: BLE001 — batch continue
                    bad += 1
                    print(f"FAIL {year}/{name}: {e}", flush=True)

            if n % 20 == 0 or n == len(files):
                print(
                    f"  {year} {n}/{len(files)}  new={ok} skip={skip} fail={bad}",
                    flush=True,
                )

    print(
        f"finished new={ok} skip={skip} fail={bad} dir={args.out}",
        flush=True,
    )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
