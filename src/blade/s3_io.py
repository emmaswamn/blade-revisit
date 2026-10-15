"""Thin S3 helpers for Worker / sync scripts. Decision core stays S3-unaware."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PREFIXES = ("2017", "2018", "overlays")


def bucket_from_env(explicit: str | None = None) -> str:
    b = explicit or os.environ.get("BLADE_S3_BUCKET") or ""
    if not b.strip():
        raise SystemExit("Set BLADE_S3_BUCKET or pass --bucket")
    return b.strip()


def region_from_env(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
        "AWS_REGION"
    )


def client(region: str | None = None):
    import boto3

    kwargs = {}
    r = region_from_env(region)
    if r:
        kwargs["region_name"] = r
    return boto3.client("s3", **kwargs)


def s3_uri(bucket: str, key: str) -> str:
    key = key.lstrip("/")
    return f"s3://{bucket}/{key}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"bad s3 uri: {uri}")
    return bucket, key


def year_key(year: str, name: str) -> str:
    return f"{year}/{Path(name).name}"


def overlay_key(name: str) -> str:
    return f"overlays/{Path(name).name}"


def upload_file(
    s3,
    *,
    bucket: str,
    key: str,
    path: Path,
    content_type: str | None = None,
) -> str:
    extra = {}
    if content_type:
        extra["ExtraArgs"] = {"ContentType": content_type}
    s3.upload_file(str(path), bucket, key, **extra)
    return s3_uri(bucket, key)


def download_file(s3, *, bucket: str, key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))
    return dest


def list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
    out: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            k = obj["Key"]
            if k.endswith("/"):
                continue
            out.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def sync_dir_to_prefix(
    s3,
    *,
    bucket: str,
    local_dir: Path,
    prefix: str,
    suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG"),
    names: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Upload files under local_dir to s3://bucket/prefix/. Returns count."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)
    prefix = prefix.strip("/")
    n = 0
    for p in sorted(local_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix not in suffixes:
            continue
        if names is not None and p.name not in names:
            continue
        key = f"{prefix}/{p.name}"
        if dry_run:
            print(f"would upload {p} -> s3://{bucket}/{key}")
        else:
            upload_file(s3, bucket=bucket, key=key, path=p, content_type="image/jpeg")
            print(f"upload {p.name} -> s3://{bucket}/{key}")
        n += 1
    return n


def sync_prefix_to_dir(
    s3,
    *,
    bucket: str,
    prefix: str,
    local_dir: Path,
    names: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Download s3://bucket/prefix/ into local_dir. Returns count."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.strip("/")
    keys = list_keys(s3, bucket=bucket, prefix=prefix)
    n = 0
    for key in keys:
        name = Path(key).name
        if names is not None and name not in names:
            continue
        dest = local_dir / name
        if dry_run:
            print(f"would download s3://{bucket}/{key} -> {dest}")
        else:
            download_file(s3, bucket=bucket, key=key, dest=dest)
            print(f"download {name}")
        n += 1
    return n
