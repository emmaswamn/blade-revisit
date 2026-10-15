"""Debug API — stage 1.1/1.2 first; cross-year search is对照 only.

See docs/pipeline-stages.md
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

_log = logging.getLogger("blade.img")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
else:
    _log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blade.blade_mask import (
    debug_expand_close_pass,
    debug_expand_lines_pass,
    debug_expand_trap_pass,
    debug_geo2_pass,
    debug_seed_aux_pass,
    debug_tip_edge_pass,
    debug_tip_need_pass,
    debug_white_refine,
    estimate_blade_mask,
)
from blade.config import load_thresholds
from blade.lighting import estimate_lighting
from blade.match import extract_features
from blade.pipeline import STAGE_LABELS, Stage
from blade.near_shot import list_near_images
from blade.retrieve import (
    GALLERY_FEATURE_TAG,
    build_gallery,
    list_images,
    search_query_in_2017,
)
from blade.store import ResultStore
from blade.background import (
    build_bg_match_texture_debug,
    cache_path_v2,
    draw_bg_block_sig_fast,
    draw_bg_blocks_debug,
    extract_bg_blocks,
    load_bg_block_sig,
    search_bg_blocks_gallery,
    search_bg_gallery,
)
from blade.rust_search import search_rust_gallery, write_rust_locus
from blade.stage1_views import STAGE1_VIEWS, encode_jpeg, write_stage1_view
from blade.usable import is_complex_structure, list_usable_near_images

DATA_ROOT = Path(os.environ.get("BLADE_DATA_ROOT", Path.home() / "data" / "wind-turbine-dtu"))
DB_PATH = Path(os.environ.get("BLADE_DB", ROOT / "data" / "results.db"))
OVERLAY_DIR = Path(os.environ.get("BLADE_OVERLAY_DIR", ROOT / "data" / "overlays"))
PREVIEW_DIR = Path(os.environ.get("BLADE_PREVIEW_DIR", ROOT / "data" / "debug_preview"))
CONFIG_PATH = ROOT / "configs" / "thresholds.yaml"
WEB_DIR = ROOT / "web"
EVAL_PAIRS_PATH = ROOT / "eval" / "pairs_15.json"
EVAL_OVERRIDES_PATH = ROOT / "eval" / "pairs_15.overrides.json"

# Display JPEGs only. Algorithms still run on full-res; 0 = no downscale.
VIEW_MAX_SIDE = int(os.environ.get("BLADE_VIEW_MAX_SIDE", "960"))
VIEW_JPEG_QUALITY = int(os.environ.get("BLADE_VIEW_JPEG_QUALITY", "80"))
# Browser may reuse stable preview URLs (no ?t= bust). force=1 regenerates disk cache.
_PREVIEW_CACHE_HEADERS = {"Cache-Control": "public, max-age=604800"}

# HTTP Basic Auth when both set (public Graviton). Local unset = open.
_BASIC_USER = (os.environ.get("BLADE_BASIC_USER") or "").strip()
_BASIC_PASS = (os.environ.get("BLADE_BASIC_PASS") or "").strip()
_BASIC_AUTH_ON = bool(_BASIC_USER and _BASIC_PASS)
_AUTH_OPEN_PATHS = frozenset({"/api/health"})


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _BASIC_AUTH_ON or request.url.path in _AUTH_OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("Authorization") or ""
        if header.startswith("Basic "):
            try:
                raw = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
                user, _, password = raw.partition(":")
                user_ok = secrets.compare_digest(user, _BASIC_USER)
                pass_ok = secrets.compare_digest(password, _BASIC_PASS)
                if user_ok and pass_ok:
                    return await call_next(request)
            except Exception:
                pass
        return StarletteResponse(
            content="Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="blade-revisit"'},
            media_type="text/plain",
        )


# Image / preview GETs — log enter + leave so slow TTFB is visible in docker logs.
_IMAGE_PATH_PREFIXES = (
    "/api/stage1/view",
    "/api/rust_preview/",
    "/api/rust_locus/",
    "/api/image/",
    "/api/bg_blocks/",
    "/api/overlay/",
)


def _is_image_api(path: str) -> bool:
    if path == "/api/stage1/view":
        return True
    return any(path.startswith(p) for p in _IMAGE_PATH_PREFIXES if p.endswith("/"))


class _ImageAccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method != "GET" or not _is_image_api(path):
            return await call_next(request)
        client = request.client.host if request.client else "?"
        qs = request.url.query
        target = f"{path}?{qs}" if qs else path
        t0 = time.perf_counter()
        _log.info("img REQ %s client=%s", target, client)
        try:
            response = await call_next(request)
        except Exception:
            ms = (time.perf_counter() - t0) * 1000.0
            _log.exception("img FAIL %s client=%s %.0fms", target, client, ms)
            raise
        ms = (time.perf_counter() - t0) * 1000.0
        clen = response.headers.get("content-length", "?")
        _log.info(
            "img RES %s status=%s bytes=%s client=%s %.0fms",
            target,
            response.status_code,
            clen,
            client,
            ms,
        )
        return response


app = FastAPI(title="blade-revisit debug", version="0.3.0")
app.add_middleware(_BasicAuthMiddleware)
app.add_middleware(_ImageAccessLogMiddleware)
store = ResultStore(DB_PATH)
thresholds = load_thresholds(CONFIG_PATH)

_gallery = None
_gallery_ref_limit: tuple[int, bool] | None = None
_gallery_feature_tag: str | None = None

# Stage-debug page only shows 1.x. Background → /bg · rust search → /match · review → /review.
PRODUCT_STAGES = [
    {
        "id": "1.0",
        "title": "光照工况",
        "status": "doing",
        "blurb": "sunny/overcast · 先调试图不改 mask",
    },
    {
        "id": "1.1",
        "title": "整机 mask",
        "status": "doing",
        "blurb": "边缘填充 + 轮毂连通（已弃用取白）",
    },
    {
        "id": "1.2",
        "title": "机上特征点",
        "status": "doing",
        "blurb": "只在 mask 内提点并画出来",
    },
]

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class SearchBody(BaseModel):
    query: str = Field(..., description="2018 filename (legacy ORB path)")
    top_k: int = Field(5, ge=1, le=20)
    # 0 = usable 全池（约 77）；>0 = usable 池内再截
    ref_limit: int = Field(0, ge=0, le=500)
    usable_only: bool = Field(True, description="2017 gallery = usable 近景池")


class RustSearchBody(BaseModel):
    """Main line: overcast 2017 query → sunny 2018 usable gallery."""

    query: str = Field("DJI_0752.JPG", description="2017 filename")
    query_year: str = Field("2017")
    gallery_year: str = Field("2018")
    top_k: int = Field(8, ge=1, le=30)
    limit: int = Field(0, ge=0, le=500, description="0 = all usable")
    gold_name: str = Field("DJI_0190.JPG")


class BgSearchBody(BaseModel):
    """Stage 2: sunny 2018 query → 2017 usable gallery (station prior)."""

    query: str = Field("DJI_0190.JPG", description="2018 filename")
    query_year: str = Field("2018")
    gallery_year: str = Field("2017")
    top_k: int = Field(8, ge=1, le=30)
    limit: int = Field(0, ge=0, le=500, description="0 = all usable")
    gold_name: str = Field("DJI_0752.JPG")
    method: str = Field(
        "blocks",
        description="blocks = v2 color+edge vote; hsv = v1 spatial HSV",
    )


class BgPairDebugBody(BaseModel):
    """Annotate matched blocks + texture crops for one query/ref pair."""

    query: str = Field("DJI_0190.JPG")
    ref: str = Field(..., description="2017 gallery filename")
    query_year: str = Field("2018")
    gallery_year: str = Field("2017")


class EvalPairPatch(BaseModel):
    expected_tier: str | None = Field(
        None, description="retake|reject|watch|escalate"
    )
    human_note: str | None = None


def _load_eval_pairs() -> dict:
    if not EVAL_PAIRS_PATH.is_file():
        raise HTTPException(404, "eval/pairs_15.json missing")

    data = json.loads(EVAL_PAIRS_PATH.read_text(encoding="utf-8"))
    overrides: dict = {}
    if EVAL_OVERRIDES_PATH.is_file():
        try:
            overrides = json.loads(EVAL_OVERRIDES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            overrides = {}
    pairs = []
    for p in data.get("pairs") or []:
        o = overrides.get(p["id"]) or {}
        merged = {**p, **{k: v for k, v in o.items() if v is not None}}
        pairs.append(merged)
    return {
        "version": data.get("version"),
        "provisional": data.get("provisional", True),
        "note": data.get("note"),
        "direction": data.get("direction"),
        "tiers": data.get("tiers"),
        "pairs": pairs,
    }


def _get_gallery(ref_limit: int, *, usable_only: bool = True):
    global _gallery, _gallery_ref_limit, _gallery_feature_tag
    pool_key = (ref_limit, usable_only)
    need = (
        _gallery is None
        or _gallery_ref_limit != pool_key
        or _gallery_feature_tag != GALLERY_FEATURE_TAG
    )
    if need:
        print(
            f"[api] building 2017 gallery ref_limit={ref_limit} "
            f"usable_only={usable_only} tag={GALLERY_FEATURE_TAG} "
            f"(mask内提点 · 磁盘特征缓存)"
        )
        _gallery = build_gallery(
            DATA_ROOT / "2017",
            thresholds,
            ref_limit=ref_limit,
            usable_only=usable_only,
            log_print=True,
        )
        _gallery_ref_limit = pool_key
        _gallery_feature_tag = GALLERY_FEATURE_TAG
    return _gallery


def _load_bgr(year: str, name: str):
    if year not in ("2017", "2018"):
        raise HTTPException(400, "year must be 2017 or 2018")
    safe = Path(name).name
    path = DATA_ROOT / year / safe
    if not path.is_file():
        raise HTTPException(404, f"image not found: {year}/{safe}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, f"unreadable: {safe}")
    return safe, bgr


def _cached_preview_response(path: Path) -> FileResponse:
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers=_PREVIEW_CACHE_HEADERS,
    )


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(404, "web/index.html missing")
    return FileResponse(index_path)


@app.get("/review")
def review_page():
    """Product review UI: 15-pair rust eval + step playback + optional bg."""
    path = WEB_DIR / "review.html"
    if not path.is_file():
        raise HTTPException(404, "web/review.html missing")
    return FileResponse(path)


@app.get("/match")
def match_page():
    """Rust search对照 UI: 2017 overcast query → 2018 usable gallery."""
    path = WEB_DIR / "match.html"
    if not path.is_file():
        raise HTTPException(404, "web/match.html missing")
    return FileResponse(path)


@app.get("/bg")
def bg_page():
    """Background / station prior UI: 2018 → 2017 usable."""
    path = WEB_DIR / "bg.html"
    if not path.is_file():
        raise HTTPException(404, "web/bg.html missing")
    return FileResponse(path)


@app.get("/api/eval/pairs")
def eval_pairs_list():
    """Provisional 15-pair eval table (human can PATCH expected tiers)."""
    return _load_eval_pairs()


@app.patch("/api/eval/pairs/{pair_id}")
def eval_pair_patch(pair_id: str, body: EvalPairPatch):
    data = _load_eval_pairs()
    ids = {p["id"] for p in data["pairs"]}
    if pair_id not in ids:
        raise HTTPException(404, f"unknown pair id: {pair_id}")
    allowed = {"retake", "reject", "watch", "escalate", "skip"}
    if body.expected_tier is not None and body.expected_tier not in allowed:
        raise HTTPException(400, f"tier must be one of {sorted(allowed)}")
    overrides: dict = {}
    if EVAL_OVERRIDES_PATH.is_file():
        try:
            overrides = json.loads(EVAL_OVERRIDES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            overrides = {}
    cur = dict(overrides.get(pair_id) or {})
    if body.expected_tier is not None:
        cur["expected_tier"] = body.expected_tier
    if body.human_note is not None:
        cur["human_note"] = body.human_note
    overrides[pair_id] = cur
    EVAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OVERRIDES_PATH.write_text(
        json.dumps(overrides, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    refreshed = _load_eval_pairs()
    pair = next(p for p in refreshed["pairs"] if p["id"] == pair_id)
    return {"ok": True, "pair": pair}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "db": str(DB_PATH),
        "data_root": str(DATA_ROOT),
        "doc": None,
        "product_stages": PRODUCT_STAGES,
        "opencv_stages": [{"id": s.value, "label": STAGE_LABELS[s]} for s in Stage],
    }


@app.get("/api/stages")
def stages():
    # Narrative / acceptance criteria live in the report & docs — not linked from UI.
    return {"stages": PRODUCT_STAGES}


@app.get("/api/images/{year}")
def list_year(
    year: str,
    limit: int = Query(500, ge=1, le=1000),
    near_only: bool = Query(True, description="Drop whole-turbine far shots"),
    usable_only: bool = Query(
        True, description="Also drop edge-unreadable / weak-seed frames (0687-class)"
    ),
):
    if year not in ("2017", "2018"):
        raise HTTPException(400, "year must be 2017 or 2018")
    folder = DATA_ROOT / year
    if not folder.is_dir():
        raise HTTPException(404, f"missing {folder}")
    if near_only and usable_only:
        paths, filt = list_usable_near_images(folder, limit=limit, year=year)
        return {
            "year": year,
            "near_only": True,
            "usable_only": True,
            "filter": filt,
            "images": [{"name": p.name} for p in paths],
        }
    if near_only:
        paths, filt = list_near_images(folder, limit=limit)
        return {
            "year": year,
            "near_only": True,
            "usable_only": False,
            "filter": filt,
            "images": [{"name": p.name} for p in paths],
        }
    return {
        "year": year,
        "near_only": False,
        "usable_only": False,
        "images": [{"name": p.name} for p in list_images(folder)[:limit]],
    }


@app.get("/api/stage1/meta")
def stage1_meta(year: str, name: str):
    """1.0 lighting + 1.1 + 1.2 stats for one image (no cross-year)."""
    safe, bgr = _load_bgr(year, name)
    lit = estimate_lighting(bgr, year=year)
    mr = estimate_blade_mask(bgr, year=year, lighting=lit.lighting)
    if is_complex_structure(safe):
        mr.usable = False
        mr.reject_reason = "complex_structure"
    wr = debug_white_refine(bgr, year=year, lighting=lit.lighting)
    sa = debug_seed_aux_pass(bgr, year=year)
    g2 = debug_geo2_pass(bgr, year=year, lighting=lit.lighting)
    tn = debug_tip_need_pass(bgr, year=year, lighting=lit.lighting)
    te = debug_tip_edge_pass(bgr, year=year, lighting=lit.lighting)
    ex = debug_expand_lines_pass(bgr, year=year, lighting=lit.lighting)
    et = debug_expand_trap_pass(bgr, year=year, lighting=lit.lighting)
    ec = debug_expand_close_pass(bgr, year=year, lighting=lit.lighting)
    full = extract_features(safe, bgr, thresholds, mask=None)
    masked = extract_features(safe, bgr, thresholds, mask=mr.mask)
    added_px = int((wr.added > 0).sum())
    seed_px = int((wr.seed > 0).sum())
    cut_px = int((g2.cut_added > 0).sum())
    keep_add_px = int((g2.kept_added > 0).sum())
    tip_add_px = int(((te.tip_fill > 0) & (te.base_mask == 0)).sum())
    return {
        "year": year,
        "name": safe,
        "lighting": {
            "label": lit.lighting,
            "lighting": lit.lighting,
            "sky_v": lit.sky_v,
            "sky_std": lit.sky_std,
            "sky_s": lit.sky_s,
            "contrast": lit.contrast,
            "lap": lit.lap,
            "veg_frac": lit.veg_frac,
            "sky_ground_gap": lit.sky_ground_gap,
            "score_sunny": lit.score_sunny,
            "score_overcast": lit.score_overcast,
            "year_bias_used": lit.year_bias_used,
            "detail": lit.detail,
            "legend": "黄框=天空带 · 青框=水面带 · 品红=植被参考 · 横幅=sunny/overcast",
        },
        "mask": {
            "area_px": mr.area_px,
            "coverage": mr.coverage,
            "components": mr.num_components_kept,
            "detail": mr.detail,
            "hub_xy": list(mr.hub_xy) if mr.hub_xy else None,
            "method": "edge_hub",
            "usable": mr.usable,
            "reject_reason": mr.reject_reason,
        },
        "usable": {
            "ok": mr.usable,
            "reason": mr.reject_reason,
            "legend": "边不可读/弱种子/结构复杂 → reject，不进 1.2 验收",
        },
        "white_refine": {
            "white_frac": wr.white_frac,
            "max_dist": wr.max_dist,
            "seed_px": seed_px,
            "added_px": added_px,
            "reject_px": int((wr.rejected > 0).sum())
            if wr.rejected is not None
            else 0,
            "note": wr.note,
            "lighting": wr.lighting,
            "interior_fill": wr.lighting == "overcast"
            or "overcast_interior_fill" in wr.note
            or "overcast_skip_white_grow" in wr.note,
            "skipped": wr.lighting == "overcast"
            or "overcast_interior_fill" in wr.note
            or "overcast_skip_white_grow" in wr.note,
            "legend": (
                "阴天 1.1b：外轮廓内填门控 · 青=种子 · 红=填入 · 橙=拒填 · 外扩见 1.3"
                if wr.lighting == "overcast"
                or "overcast_interior_fill" in wr.note
                or "overcast_skip_white_grow" in wr.note
                else "seed=几何起点 · added=白扩新增 · dmax=允许扩到的距离"
            ),
        },
        "seed_aux": {
            "n_rim": len(sa.raw_segs),
            "n_lower": len(sa.lower_segs),
            "n_upper": len(sa.upper_segs),
            "n_chains": len(sa.linked_chains),
            "best_chain_len": float(
                sum(
                    (
                        (sa.linked_chains[0][i + 1][0] - sa.linked_chains[0][i][0]) ** 2
                        + (sa.linked_chains[0][i + 1][1] - sa.linked_chains[0][i][1]) ** 2
                    )
                    ** 0.5
                    for i in range(len(sa.linked_chains[0]) - 1)
                )
                if sa.linked_chains and len(sa.linked_chains[0]) >= 2
                else 0.0
            ),
            "note": sa.note,
            "legend": (
                "1.1a 辅助线（只画不切）：黄=缘 Hough · 蓝=上缘 · 橙=下缘 · "
                "绿=断开边连成链 · 看视图 seed_aux"
            ),
        },
        "geo2": {
            "mode": g2.mode,
            "cut_px": cut_px,
            "keep_add_px": keep_add_px,
            "note": g2.note,
            "legend": "青=种子 · 绿=geo2保留的白扩 · 红=砍掉的扩大 · 品红=中心线",
        },
        "tip_need": {
            "needed": tn.needed,
            "reason": tn.reason,
            "tip_w": tn.tip_w,
            "mid_w": tn.mid_w,
            "elong": tn.elong,
            "paint_gap": tn.paint_gap,
        },
        "tip_edge": {
            "note": te.note,
            "needed": te.needed,
            "need_note": te.need_note,
            "add_px": tip_add_px,
            "left_n": len(te.left_poly),
            "right_n": len(te.right_poly),
            "tip_point": list(te.tip_point) if te.tip_point else None,
            "legend": "先看 tip_need；YES 才跑 tip_edge",
        },
        "expand_lines": {
            "paired": ex.paired,
            "n_all": len(ex.all_segs),
            "n_ring": len(ex.ring_segs),
            "has_left": ex.left_seg is not None,
            "has_right": ex.right_seg is not None,
            "has_phantom": ex.phantom_seg is not None,
            "note": ex.note,
            "legend": (
                "1.3a 只画线：黄=贴主体候选 · 橙=环内 · 绿/蓝=选用"
                "（原黄/橙保留，勿全图收）"
            ),
        },
        "expand_trap": {
            "n_traps": et.n_traps,
            "n_accept": et.n_accept,
            "n_reject": et.n_reject,
            "note": et.note,
            "legend": (
                "1.3c 趋势梯形探边（参考 tip_edge）：黄框=梯形 · 白=Canny"
                " · 绿=接受 · 品红圈=尖端终点 · 红=拒天/植/水 · 不改 mask"
            ),
        },
        "expand_close": {
            "n_yellow": ec.n_yellow,
            "add_px": ec.add_px,
            "hull_n": len(ec.hull_pts),
            "mode": ec.mode,
            "gate": ec.gate,
            "note": ec.note,
            "legend": (
                "阴天优先 trap 缘→尖端多边形闭合；失败回退黄凸包"
                " · 青=1.1b · 红=新增 · 品红=闭合线"
            ),
        },
        "keypoints": {
            "full_frame": len(full.keypoints),
            "masked": len(masked.keypoints),
            "detector": thresholds.detector,
        },
        "steps": [
            {
                "stage": "3_lighting",
                "status": "ok",
                "detail": (
                    f"lighting={lit.lighting} su={lit.score_sunny:.1f} oc={lit.score_overcast:.1f}"
                    f" sky_v={lit.sky_v:.0f} sky_std={lit.sky_std:.1f} sky_s={lit.sky_s:.0f}"
                    f" contrast={lit.contrast:.1f} lap={lit.lap:.0f} veg={lit.veg_frac:.2f}"
                    + (" year_bias" if lit.year_bias_used else "")
                    + " · 看视图 lighting"
                ),
            },
            {"stage": "4_blade_mask", "status": "ok" if mr.num_components_kept else "fail", "detail": mr.detail},
            {
                "stage": "4a_usable",
                "status": "ok" if mr.usable else "reject",
                "detail": (
                    f"usable={mr.usable} reject={mr.reject_reason}"
                    " · 边不可读(0687)/多叶毂(0600)/结构复杂(0782)筛掉 · 列表 usable_only"
                ),
            },
            {
                "stage": "4a2_seed_aux",
                "status": "ok",
                "detail": (
                    f"{sa.note} · 看视图 seed_aux（只画线，不断种子）"
                ),
            },
            {
                "stage": "4b_white_grow",
                "status": "ok" if mr.usable else "skip",
                "detail": (
                    f"white_frac={wr.white_frac:.2f} dmax={wr.max_dist:.0f} "
                    f"seed={seed_px} added={added_px} · 看视图 seed / refine_delta"
                ),
            },
            {
                "stage": "4c_geo2",
                "status": "ok" if mr.usable else "skip",
                "detail": (
                    f"{g2.note} · cut={cut_px} keep_add={keep_add_px} · 看视图 geo2"
                ),
            },
            {
                "stage": "4d_tip_need",
                "status": ("ok" if tn.needed else "skip") if mr.usable else "skip",
                "detail": (
                    f"{'YES' if tn.needed else 'NO'} · {tn.reason}"
                    f" · tw={tn.tip_w:.1f} gap={tn.paint_gap} · 看视图 tip_need"
                ),
            },
            {
                "stage": "4e_tip_edge",
                "status": ("ok" if te.needed else "skip") if mr.usable else "skip",
                "detail": (
                    f"{te.note} · add={tip_add_px} L/R={len(te.left_poly)}/{len(te.right_poly)}"
                    " · 看视图 tip_edge"
                ),
            },
            {
                "stage": "4f_expand_lines",
                "status": "ok" if (ex.left_seg or ex.right_seg or ex.all_segs) else "fail",
                "detail": (
                    f"paired={ex.paired} ring={len(ex.ring_segs)} "
                    f"L/R={int(ex.left_seg is not None)}/{int(ex.right_seg is not None)} "
                    f"phantom={int(ex.phantom_seg is not None)} · 看视图 expand_lines"
                ),
            },
            {
                "stage": "4f2_expand_trap",
                "status": "ok" if et.n_accept > 0 or et.n_traps > 0 else "fail",
                "detail": (
                    f"traps={et.n_traps} accept={et.n_accept} reject={et.n_reject}"
                    " · 看视图 expand_trap"
                ),
            },
            {
                "stage": "4g_expand_close",
                "status": "ok" if ec.add_px > 0 or ec.n_yellow > 0 else "skip",
                "detail": (
                    f"mode={ec.mode} gate={ec.gate or '-'} add={ec.add_px} "
                    f"poly={len(ec.hull_pts)} · 看视图 expand_delta"
                ),
            },
            {
                "stage": "5_features",
                "status": (
                    "ok"
                    if (mr.usable and masked.keypoints)
                    else ("reject" if not mr.usable else "fail")
                ),
                "detail": (
                    f"masked={len(masked.keypoints)} full={len(full.keypoints)}"
                    + ("" if mr.usable else " · gated_off usable=false")
                ),
            },
        ],
        "acceptance_hint": (
            "先看 1.0 光照；再看 1.1；4a_usable=reject 的图（边不可读）应从列表筛掉，不进 1.2 验收。"
        ),
    }


@app.get("/api/stage1/view")
def stage1_view(
    year: str,
    name: str,
    view: str = Query(
        "overlay",
        description=(
            "original|lighting|sky_water|sunny_water|edge_seed|seed_aux|overlay|blade|seed|refine_delta|"
            "geo2|tip_need|tip_edge|expand_lines|expand_trap|expand_delta|kps_masked|kps_full"
        ),
    ),
    force: bool = Query(False, description="Rebuild disk JPEG even if cache is fresh"),
):
    """Serve a stage1 debug JPEG; prefer disk cache under data/debug_preview/."""
    if view not in STAGE1_VIEWS:
        raise HTTPException(400, f"unknown view: {view}")
    if year not in ("2017", "2018"):
        raise HTTPException(400, "year must be 2017 or 2018")
    src = _image_path(year, name)
    try:
        path = write_stage1_view(
            src,
            year=year,
            view=view,
            out_dir=PREVIEW_DIR,
            thresholds=thresholds,
            max_side=VIEW_MAX_SIDE,
            quality=VIEW_JPEG_QUALITY,
            force=force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _cached_preview_response(path)


@app.post("/api/search")
def search(body: SearchBody):
    """Legacy ORB path: 2018→2017（对照保留，非主线）。"""
    qpath = DATA_ROOT / "2018" / Path(body.query).name
    if not qpath.is_file():
        raise HTTPException(404, f"2018 image not found: {body.query}")
    gallery = _get_gallery(body.ref_limit, usable_only=body.usable_only)
    return search_query_in_2017(
        query_path=qpath,
        gallery=gallery,
        thresholds=thresholds,
        store=store,
        overlay_dir=OVERLAY_DIR,
        top_k=body.top_k,
    )


@app.post("/api/rust_search")
def rust_search(body: RustSearchBody):
    """主线：阴天 2017 锈签名 → 搜 2018 usable（四点 rust↔rust + 缘距比例）。"""
    if body.query_year not in ("2017", "2018") or body.gallery_year not in (
        "2017",
        "2018",
    ):
        raise HTTPException(400, "years must be 2017 or 2018")
    qpath = DATA_ROOT / body.query_year / Path(body.query).name
    if not qpath.is_file():
        raise HTTPException(404, f"query not found: {body.query_year}/{body.query}")
    try:
        return search_rust_gallery(
            data_root=DATA_ROOT,
            query_name=body.query,
            query_year=body.query_year,
            gallery_year=body.gallery_year,
            top_k=body.top_k,
            limit=body.limit,
            use_cache=True,
            preview_dir=PREVIEW_DIR,
            gold_name=body.gold_name,
            log_print=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/rust_preview/{name}")
def rust_preview(name: str):
    """Serve rust/bg preview JPEG from debug_preview/."""
    safe = Path(name).name
    if not safe.endswith((".jpg", ".jpeg", ".JPG", ".JPEG")):
        raise HTTPException(400, "preview must be jpeg")
    path = PREVIEW_DIR / safe
    if not path.is_file():
        raise HTTPException(404, f"preview not found: {safe}")
    return _cached_preview_response(path)


@app.get("/api/rust_locus/{year}/{name}")
def rust_locus(
    year: str,
    name: str,
    force: bool = Query(False),
    precise: bool = Query(
        False,
        description="True=rebuild atlas crop (slow). Default uses rust_sig .npz spots.",
    ),
):
    """Original-frame annotation: rust spots + crop box (provenance)."""
    if year not in ("2017", "2018"):
        raise HTTPException(400, "year must be 2017 or 2018")
    src = DATA_ROOT / year / Path(name).name
    if not src.is_file():
        raise HTTPException(404, f"image not found: {year}/{name}")
    written = write_rust_locus(
        src,
        year=year,
        out_dir=PREVIEW_DIR,
        color=(0, 220, 255) if year == "2017" else (0, 255, 120),
        force=force,
        precise=precise,
        max_side=VIEW_MAX_SIDE,
        quality=VIEW_JPEG_QUALITY,
    )
    if not written:
        raise HTTPException(400, "rust locus failed (mask / few spots)")
    path = PREVIEW_DIR / written
    if not path.is_file():
        raise HTTPException(404, f"locus preview missing: {written}")
    return _cached_preview_response(path)


@app.post("/api/bg_pair_debug")
def bg_pair_debug(body: BgPairDebugBody):
    """同色 M# 标注匹配块 + 下方纹理裁剪，方便人眼 debug。"""
    if body.query_year not in ("2017", "2018") or body.gallery_year not in (
        "2017",
        "2018",
    ):
        raise HTTPException(400, "years must be 2017 or 2018")
    try:
        panel, vote = build_bg_match_texture_debug(
            data_root=DATA_ROOT,
            query_name=body.query,
            ref_name=body.ref,
            query_year=body.query_year,
            gallery_year=body.gallery_year,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"bg_match_tex_{Path(body.query).stem}_vs_{Path(body.ref).stem}.jpg"
    out = PREVIEW_DIR / stem
    import cv2

    cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {
        "query": f"{body.query_year}/{Path(body.query).name}",
        "ref": f"{body.gallery_year}/{Path(body.ref).name}",
        "score": vote.get("score"),
        "hits": vote.get("hits"),
        "adj_score": vote.get("adj_score"),
        "pairs": vote.get("pairs"),
        "adj_pairs": vote.get("adj_pairs"),
        "preview": stem,
    }


@app.post("/api/bg_search")
def bg_search(body: BgSearchBody):
    """阶段 2：晴天 2018 背景 → 搜 2017 usable（站位弱先验）。

    method=blocks（默认）：色块多锚投票；method=hsv：旧网格直方图。
    """
    if body.query_year not in ("2017", "2018") or body.gallery_year not in (
        "2017",
        "2018",
    ):
        raise HTTPException(400, "years must be 2017 or 2018")
    method = (body.method or "blocks").strip().lower()
    if method not in ("blocks", "hsv", "bg_sig_v2", "bg_sig_v1"):
        raise HTTPException(400, "method must be blocks or hsv")
    use_blocks = method in ("blocks", "bg_sig_v2")
    qpath = DATA_ROOT / body.query_year / Path(body.query).name
    if not qpath.is_file():
        raise HTTPException(404, f"query not found: {body.query_year}/{body.query}")
    try:
        fn = search_bg_blocks_gallery if use_blocks else search_bg_gallery
        return fn(
            data_root=DATA_ROOT,
            query_name=body.query,
            query_year=body.query_year,
            gallery_year=body.gallery_year,
            top_k=body.top_k,
            limit=body.limit,
            use_cache=True,
            preview_dir=PREVIEW_DIR,
            gold_name=body.gold_name,
            log_print=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/pairs")
def list_pairs(
    limit: int = Query(50, ge=1, le=500),
    tier: str | None = None,
    min_inlier: float | None = None,
    best_only: bool = False,
    query_id: str | None = None,
):
    if best_only:
        rows = store.best_per_query(limit=limit)
    else:
        rows = store.list_pairs(
            limit=limit, tier=tier, min_inlier=min_inlier, query_id=query_id
        )
    return {"pairs": rows}


@app.get("/api/pairs/{pair_id}")
def get_pair(pair_id: int):
    row = store.get_pair(pair_id)
    if not row:
        raise HTTPException(404, "pair not found")
    return row


def _image_path(year: str, name: str) -> Path:
    safe = Path(name).name
    path = DATA_ROOT / year / safe
    if not path.is_file():
        raise HTTPException(404, f"image not found: {year}/{safe}")
    return path


def _display_image(year: str, name: str, *, full: bool, force: bool) -> Response:
    """Browser display: downscaled cached JPEG. full=1 serves original file."""
    src = _image_path(year, name)
    if full:
        return FileResponse(src)
    try:
        path = write_stage1_view(
            src,
            year=year,
            view="original",
            out_dir=PREVIEW_DIR,
            thresholds=thresholds,
            max_side=VIEW_MAX_SIDE,
            quality=VIEW_JPEG_QUALITY,
            force=force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _cached_preview_response(path)


@app.get("/api/image/2017/{name}")
def image_2017(
    name: str,
    full: bool = Query(False, description="True=raw original (large)"),
    force: bool = Query(False),
):
    return _display_image("2017", name, full=full, force=force)


@app.get("/api/image/2018/{name}")
def image_2018(
    name: str,
    full: bool = Query(False, description="True=raw original (large)"),
    force: bool = Query(False),
):
    return _display_image("2018", name, full=full, force=force)


@app.get("/api/bg_blocks/{year}/{name}")
def bg_blocks_preview(
    year: str,
    name: str,
    precise: bool = Query(
        False,
        description="True=full mask+contours (slow). Default draws anchors from bg_sig npz.",
    ),
    force: bool = Query(False),
):
    """On-demand color-block debug JPEG for /bg 2×2 view."""
    if year not in ("2017", "2018"):
        raise HTTPException(400, "year must be 2017 or 2018")
    src = _image_path(year, name)
    stem = Path(src.name).stem
    # Separate cache files so fast/precise previews do not clobber each other.
    tag = "precise" if precise else "fast"
    out = PREVIEW_DIR / f"bg_blocks_{year}_{stem}_{tag}.jpg"
    if (
        force
        or (not out.is_file())
        or (out.stat().st_mtime < src.stat().st_mtime)
    ):
        bgr = cv2.imread(str(src))
        if bgr is None:
            raise HTTPException(404, f"unreadable: {year}/{src.name}")
        vis = None
        if not precise:
            sig = load_bg_block_sig(cache_path_v2(year, src.name))
            if sig is not None and sig.anchors:
                vis = draw_bg_block_sig_fast(bgr, sig)
        if vis is None:
            res = extract_bg_blocks(src.name, bgr, year=year)
            vis = draw_bg_blocks_debug(res)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        # Same display budget as stage1 / locus (not full-res contours).
        out.write_bytes(
            encode_jpeg(vis, max_side=VIEW_MAX_SIDE, quality=VIEW_JPEG_QUALITY)
        )
    return _cached_preview_response(out)


@app.get("/api/overlay/{pair_id}")
def overlay(pair_id: int):
    row = store.get_pair(pair_id)
    if not row or not row.get("overlay_uri"):
        raise HTTPException(404, "overlay missing")
    path = Path(row["overlay_uri"])
    if not path.is_file():
        alt = OVERLAY_DIR / f"{Path(row['query_id']).stem}__{Path(row['ref_id']).stem}.jpg"
        if alt.is_file():
            path = alt
        else:
            raise HTTPException(404, f"overlay file missing: {path}")
    return FileResponse(path)
