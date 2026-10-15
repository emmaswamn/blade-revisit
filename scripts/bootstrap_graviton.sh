#!/usr/bin/env bash
# Bootstrap a Graviton (or any) host: deps, usable sync from S3, start API.
# Run from repo root on the instance (Ubuntu 22.04/24.04 arm64 recommended).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${BLADE_S3_BUCKET:?set BLADE_S3_BUCKET}"
export BLADE_DATA_ROOT="${BLADE_DATA_ROOT:-/data/wind-turbine-dtu}"
export BLADE_DB="${BLADE_DB:-/data/results.db}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PORT="${BLADE_PORT:-8765}"

echo "==> OpenCV / Python env"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

mkdir -p "$BLADE_DATA_ROOT" "$(dirname "$BLADE_DB")" data/overlays

echo "==> Sync usable JPGs from s3://${BLADE_S3_BUCKET} → ${BLADE_DATA_ROOT}"
python scripts/sync_usable_from_s3.py --data-root "$BLADE_DATA_ROOT"

echo "==> OpenCV $(python -c 'import cv2; print(cv2.__version__)')"

echo "==> Start API on 0.0.0.0:${PORT}"
exec uvicorn api.main:app --app-dir . --host 0.0.0.0 --port "$PORT"
