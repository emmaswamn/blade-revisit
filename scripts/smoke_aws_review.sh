#!/usr/bin/env bash
# Smoke: OpenCV import, optional S3 list, API health + /review HTML.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
API="${BLADE_API:-http://127.0.0.1:8765}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> OpenCV"
python - <<'PY'
import cv2
print("opencv", cv2.__version__)
assert cv2.__version__, "empty opencv version"
PY

if [[ -n "${BLADE_S3_BUCKET:-}" ]]; then
  echo "==> S3 list-only (bucket=${BLADE_S3_BUCKET})"
  python worker/run_s3.py --list-only
else
  echo "==> skip S3 (BLADE_S3_BUCKET unset)"
fi

CURL_AUTH=()
if [[ -n "${BLADE_BASIC_USER:-}" && -n "${BLADE_BASIC_PASS:-}" ]]; then
  CURL_AUTH=(-u "${BLADE_BASIC_USER}:${BLADE_BASIC_PASS}")
  echo "==> Basic Auth on (user=${BLADE_BASIC_USER})"
fi

echo "==> API health ${API}"
curl -fsS "${API}/api/health" | python -c "import sys,json; d=json.load(sys.stdin); assert d.get('ok'), d; print('health ok', d.get('data_root'))"

echo "==> /review page"
code=$(curl -fsS "${CURL_AUTH[@]}" -o /tmp/blade_review.html -w '%{http_code}' "${API}/review")
test "$code" = "200"
grep -q 'blade-revisit\|review\|评测\|Eval' /tmp/blade_review.html
echo "review html ok (${code})"

echo "==> /api/eval/pairs"
curl -fsS "${CURL_AUTH[@]}" "${API}/api/eval/pairs" | python -c "
import sys,json
d=json.load(sys.stdin)
n=len(d.get('pairs') or [])
print('pairs', n)
assert n >= 1, 'eval pairs empty'
"

echo "SMOKE OK"
