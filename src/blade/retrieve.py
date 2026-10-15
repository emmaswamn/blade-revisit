"""2018 query → search 2017 gallery. Shared by CLI worker and debug API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from blade.blade_mask import estimate_blade_mask
from blade.decide import decide_pair
from blade.match import (
    FeaturePack,
    draw_match_overlay,
    extract_features,
    match_features,
)
from blade.pipeline import PipelineLog, Stage
from blade.sharpness import laplacian_variance
from blade.store import ResultStore
from blade.types import PairResult, Thresholds, Tier
from blade.usable import FILTER_VERSION, list_usable_near_images


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# Bump when feature geometry changes so API gallery cache cannot reuse全图点.
GALLERY_FEATURE_TAG = "masked_v1"

_DEFAULT_FEATURE_CACHE = (
    Path(__file__).resolve().parents[2] / "data" / "gallery_features"
)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in IMAGE_EXTS)


def _year_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part in ("2017", "2018"):
            return part
    return None


def _mask_for_features(
    bgr: np.ndarray, *, year: str | None = None
) -> tuple[np.ndarray, str]:
    """Blade mask for feature detect — slight erode to cut rim→background leak."""
    mr = estimate_blade_mask(bgr, year=year)
    mask = mr.mask
    if mask.max() == 0:
        return mask, f"empty|{mr.detail}"
    # 1px erode: drop fringe that often sits on sky/field after resize
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, k, iterations=1)
    if mask.max() == 0:
        mask = mr.mask  # fall back if erode wiped a thin blade
    return mask, mr.detail


def _feature_cache_key(thresholds: Thresholds) -> str:
    raw = (
        f"{GALLERY_FEATURE_TAG}|det={thresholds.detector}|"
        f"max_side={thresholds.max_side}|usable_v={FILTER_VERSION}"
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _cache_dir_for(thresholds: Thresholds, cache_root: Path) -> Path:
    return cache_root / f"{GALLERY_FEATURE_TAG}_{_feature_cache_key(thresholds)}"


def _npz_path(cache_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    return cache_dir / f"{stem}.npz"


def _save_feature_pack(path: Path, pack: FeaturePack, thresholds: Thresholds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kps = pack.keypoints
    if kps:
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        sizes = np.array([kp.size for kp in kps], dtype=np.float32)
        angles = np.array([kp.angle for kp in kps], dtype=np.float32)
        responses = np.array([kp.response for kp in kps], dtype=np.float32)
        octaves = np.array([kp.octave for kp in kps], dtype=np.int32)
        class_ids = np.array([kp.class_id for kp in kps], dtype=np.int32)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
        sizes = np.zeros((0,), dtype=np.float32)
        angles = np.zeros((0,), dtype=np.float32)
        responses = np.zeros((0,), dtype=np.float32)
        octaves = np.zeros((0,), dtype=np.int32)
        class_ids = np.zeros((0,), dtype=np.int32)
    desc = pack.descriptors
    if desc is None:
        desc = np.zeros((0, 1), dtype=np.uint8)
    np.savez_compressed(
        path,
        image_id=np.array(pack.image_id),
        gray=pack.gray,
        scale=np.float64(pack.scale),
        pts=pts,
        sizes=sizes,
        angles=angles,
        responses=responses,
        octaves=octaves,
        class_ids=class_ids,
        descriptors=desc,
        feature_tag=np.array(GALLERY_FEATURE_TAG),
        detector=np.array(thresholds.detector),
        max_side=np.int32(thresholds.max_side),
        usable_v=np.int32(FILTER_VERSION),
    )


def _load_feature_pack(
    path: Path, thresholds: Thresholds
) -> FeaturePack | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    loaded: dict | None = None
    try:
        tag = str(data["feature_tag"])
        detector = str(data["detector"])
        max_side = int(data["max_side"])
        usable_v = int(data["usable_v"])
        if (
            tag == GALLERY_FEATURE_TAG
            and detector == thresholds.detector
            and max_side == thresholds.max_side
            and usable_v == FILTER_VERSION
        ):
            loaded = {
                "pts": np.array(data["pts"], copy=True),
                "sizes": np.array(data["sizes"], copy=True),
                "angles": np.array(data["angles"], copy=True),
                "responses": np.array(data["responses"], copy=True),
                "octaves": np.array(data["octaves"], copy=True),
                "class_ids": np.array(data["class_ids"], copy=True),
                "gray": np.array(data["gray"], copy=True),
                "desc": np.array(data["descriptors"], copy=True),
                "image_id": str(data["image_id"]),
                "scale": float(data["scale"]),
            }
    except (KeyError, ValueError, TypeError):
        loaded = None
    finally:
        data.close()

    if loaded is None:
        return None

    pts = loaded["pts"]
    kps = [
        cv2.KeyPoint(
            x=float(pts[i, 0]),
            y=float(pts[i, 1]),
            size=float(loaded["sizes"][i]),
            angle=float(loaded["angles"][i]),
            response=float(loaded["responses"][i]),
            octave=int(loaded["octaves"][i]),
            class_id=int(loaded["class_ids"][i]),
        )
        for i in range(len(pts))
    ]
    desc = loaded["desc"]
    if desc.ndim != 2 or desc.shape[0] == 0:
        desc = None
    return FeaturePack(
        image_id=loaded["image_id"],
        gray=loaded["gray"],
        keypoints=kps,
        descriptors=desc,
        scale=loaded["scale"],
    )


def _write_cache_meta(
    cache_dir: Path,
    *,
    names: list[str],
    thresholds: Thresholds,
    hit: int,
    miss: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_tag": GALLERY_FEATURE_TAG,
        "detector": thresholds.detector,
        "max_side": thresholds.max_side,
        "usable_v": FILTER_VERSION,
        "n": len(names),
        "names_sha1": hashlib.sha1("\n".join(names).encode()).hexdigest()[:16],
        "disk_hit": hit,
        "disk_miss": miss,
    }
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


@dataclass
class GalleryCache:
    """2017 feature packs. BGR is loaded lazily for Top-k decide only."""

    packs: list[tuple[Path, FeaturePack]]
    feature_tag: str = GALLERY_FEATURE_TAG
    pool: str = "usable"
    cache_hit: int = 0
    cache_miss: int = 0
    cache_dir: str | None = None
    names: list[str] = field(default_factory=list)


def build_gallery(
    year2017: Path,
    thresholds: Thresholds,
    *,
    ref_limit: int = 0,
    usable_only: bool = True,
    feature_cache_dir: Path | None = None,
    use_disk_cache: bool = True,
    log_print: bool = True,
) -> GalleryCache:
    """Extract 2017 features **inside blade mask only** (no background inliers).

    Default gallery = usable near-shot pool (same口径 as stage-1 list).
    Features are cached under data/gallery_features/ so reload does not re-mask.
    """
    year2017 = Path(year2017)
    year = _year_from_path(year2017) or "2017"
    if usable_only:
        refs, filt = list_usable_near_images(
            year2017, limit=5000, year=year
        )
        pool = "usable"
        pool_note = (
            f"usable≈{filt.get('usable', len(refs))} "
            f"(cached={filt.get('cached')})"
        )
    else:
        refs = list_images(year2017)
        pool = "all"
        pool_note = f"all_files={len(refs)}"

    if ref_limit > 0:
        refs = refs[:ref_limit]

    cache_root = (
        Path(feature_cache_dir)
        if feature_cache_dir is not None
        else _DEFAULT_FEATURE_CACHE
    )
    cache_dir = _cache_dir_for(thresholds, cache_root) if use_disk_cache else None

    if log_print:
        print(
            f"[pipeline] STAGE {Stage.FEATURES.value} "
            f"提取 2017 gallery ×{len(refs)} "
            f"(mask内提点 {GALLERY_FEATURE_TAG} · pool={pool} · {pool_note})"
        )
        if cache_dir is not None:
            print(f"  feature disk cache → {cache_dir}")

    packs: list[tuple[Path, FeaturePack]] = []
    hit = 0
    miss = 0
    for i, path in enumerate(refs, 1):
        pack: FeaturePack | None = None
        if cache_dir is not None:
            pack = _load_feature_pack(_npz_path(cache_dir, path.name), thresholds)
            if pack is not None:
                hit += 1
        if pack is None:
            miss += 1
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                if log_print:
                    print(f"  skip unreadable {path.name}")
                continue
            mask, mdetail = _mask_for_features(img, year=year)
            pack = extract_features(path.name, img, thresholds, mask=mask)
            if cache_dir is not None:
                _save_feature_pack(
                    _npz_path(cache_dir, path.name), pack, thresholds
                )
            if log_print and i == 1:
                print(
                    f"    mask_detail={mdetail[:120]} "
                    f"cov={(mask > 0).mean():.3f}"
                )
        packs.append((path, pack))
        if log_print and (i % 10 == 0 or i == len(refs)):
            print(
                f"  gallery features {i}/{len(refs)} {path.name} "
                f"kps={len(pack.keypoints)} (hit={hit} miss={miss})"
            )

    if cache_dir is not None:
        _write_cache_meta(
            cache_dir,
            names=[p.name for p, _ in packs],
            thresholds=thresholds,
            hit=hit,
            miss=miss,
        )

    if log_print:
        print(
            f"  gallery ready ×{len(packs)} pool={pool} "
            f"disk_hit={hit} disk_miss={miss}"
        )

    return GalleryCache(
        packs=packs,
        feature_tag=GALLERY_FEATURE_TAG,
        pool=pool,
        cache_hit=hit,
        cache_miss=miss,
        cache_dir=str(cache_dir) if cache_dir else None,
        names=[p.name for p, _ in packs],
    )


def probe_query_2018(
    query_path: Path,
    thresholds: Thresholds,
) -> tuple[np.ndarray | None, FeaturePack | None, PipelineLog, PairResult | None]:
    """Load 2018 query; return skip PairResult if too blurry / too few keypoints."""
    log = PipelineLog()
    img = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if img is None:
        log.add(Stage.LOAD, "fail", f"unreadable {query_path.name}")
        skip = PairResult(
            query_id=query_path.name,
            ref_id="",
            sharpness_ok=False,
            inlier_ratio=None,
            blade_width_px=None,
            area_ratio_2017=None,
            area_ratio_2018=None,
            area_delta=None,
            tier=Tier.SKIP,
            reason=f"无法读取 2018 图 {query_path.name}",
            thresholds_used=thresholds.snapshot(),
            lap_2018=None,
            lap_2017=None,
            sharp_2018_ok=False,
            sharp_2017_ok=None,
            pipeline_furthest=log.furthest,
            pipeline_steps=log.to_list(),
        )
        return None, None, log, skip

    log.add(Stage.LOAD, "ok", f"{query_path.name} shape={img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log.add(Stage.GRAY_RESIZE, "ok", f"gray={gray.shape} (features 时再缩到 max_side)")

    lap = laplacian_variance(gray)
    ok = lap >= thresholds.blur_laplacian_min
    log.add(
        Stage.SHARPNESS,
        "ok" if ok else "fail",
        f"2018 lap={lap:.1f} thr={thresholds.blur_laplacian_min} → {'ok' if ok else 'FAIL'}",
    )
    if not ok:
        print(
            f"[pipeline] SKIP {query_path.name}: 2018 清晰度不足 "
            f"lap={lap:.1f} < {thresholds.blur_laplacian_min}，不搜 2017"
        )
        skip = PairResult(
            query_id=query_path.name,
            ref_id="",
            sharpness_ok=False,
            inlier_ratio=None,
            blade_width_px=None,
            area_ratio_2017=None,
            area_ratio_2018=None,
            area_delta=None,
            tier=Tier.SKIP,
            reason=(
                f"2018清晰度不足(lap={lap:.1f}<{thresholds.blur_laplacian_min})，跳过匹配"
            ),
            thresholds_used=thresholds.snapshot(),
            lap_2018=lap,
            lap_2017=None,
            sharp_2018_ok=False,
            sharp_2017_ok=None,
            pipeline_furthest=log.furthest,
            pipeline_steps=log.to_list(),
        )
        return img, None, log, skip

    year = _year_from_path(query_path) or "2018"
    mask, mdetail = _mask_for_features(img, year=year)
    cov = float((mask > 0).mean())
    log.add(
        Stage.BLADE_MASK,
        "ok" if mask.max() > 0 else "fail",
        f"mask内提点 cov={cov:.3f} · {mdetail[:160]}",
    )
    pack = extract_features(query_path.name, img, thresholds, mask=mask)
    nkp = len(pack.keypoints)
    log.add(
        Stage.FEATURES,
        "ok" if nkp >= thresholds.min_query_keypoints else "fail",
        f"2018 masked keypoints={nkp} (min={thresholds.min_query_keypoints}) "
        f"detector on {pack.gray.shape}",
    )
    if nkp < thresholds.min_query_keypoints:
        print(
            f"[pipeline] SKIP {query_path.name}: keypoints={nkp} < "
            f"{thresholds.min_query_keypoints}，不好匹配，不搜 2017"
        )
        skip = PairResult(
            query_id=query_path.name,
            ref_id="",
            sharpness_ok=True,
            inlier_ratio=None,
            blade_width_px=None,
            area_ratio_2017=None,
            area_ratio_2018=None,
            area_delta=None,
            tier=Tier.SKIP,
            reason=(
                f"2018特征点过少({nkp}<{thresholds.min_query_keypoints})，跳过匹配"
            ),
            thresholds_used=thresholds.snapshot(),
            lap_2018=lap,
            lap_2017=None,
            sharp_2018_ok=True,
            sharp_2017_ok=None,
            pipeline_furthest=log.furthest,
            pipeline_steps=log.to_list(),
        )
        return img, pack, log, skip

    return img, pack, log, None


def search_query_in_2017(
    *,
    query_path: Path,
    gallery: GalleryCache,
    thresholds: Thresholds,
    store: ResultStore,
    overlay_dir: Path,
    top_k: int = 5,
) -> dict:
    """Run one 2018→2017 search. Returns JSON-serializable summary for the UI."""
    print(
        f"\n[pipeline] ===== query 2018={query_path.name} → "
        f"gallery 2017 ×{len(gallery.packs)} pool={gallery.pool} ====="
    )
    qimg, qpack, qlog, skip = probe_query_2018(query_path, thresholds)
    for ev in qlog.events:
        print(f"  [{ev.status}] {ev.stage}: {ev.detail}")

    if skip is not None:
        pair_id = store.upsert_pair(skip)
        return {
            "skipped": True,
            "query_id": query_path.name,
            "pair_ids": [pair_id],
            "pairs": [store.get_pair(pair_id)],
            "pipeline_steps": skip.pipeline_steps,
            "pipeline_furthest": skip.pipeline_furthest,
            "reason": skip.reason,
            "gallery_pool": gallery.pool,
            "gallery_n": len(gallery.packs),
            "no_valid_match": False,
        }

    assert qimg is not None and qpack is not None
    print(
        f"[pipeline] STAGE {Stage.MATCH_RATIO.value} "
        f"对 {len(gallery.packs)} 张 2017 做匹配…"
    )
    ranked: list[tuple[float, Path, FeaturePack, object]] = []
    for rpath, rpack in gallery.packs:
        stats = match_features(qpack, rpack, thresholds)
        score = stats.inlier_ratio if stats.inlier_ratio is not None else -1.0
        ranked.append((score, rpath, rpack, stats))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]

    any_valid = any(score >= 0 for score, *_ in ranked)
    if not any_valid:
        print(
            "[pipeline] 无有效 Homography/inlier — "
            "Top-K 仅为名单占位，不是真实排名"
        )

    pair_ids: list[int] = []
    print(f"[pipeline] STAGE {Stage.HOMOGRAPHY.value} / {Stage.DECIDE.value} top-{top_k}")
    for rank, (score, rpath, rpack, stats) in enumerate(top, 1):
        # Fresh log per pair, prepend query-side events
        plog = PipelineLog()
        for ev in qlog.events:
            plog.events.append(ev)
            plog.furthest = ev.stage
        plog.add(
            Stage.MATCH_RATIO,
            "ok" if stats.num_matches >= thresholds.min_matches else "fail",
            f"vs {rpath.name}: good_matches={stats.num_matches}",
        )
        rimg = cv2.imread(str(rpath), cv2.IMREAD_COLOR)
        if rimg is None:
            print(f"  #{rank} skip unreadable ref {rpath.name}")
            continue
        result = decide_pair(
            query_id=query_path.name,
            ref_id=rpath.name,
            query_bgr=qimg,
            ref_bgr=rimg,
            thresholds=thresholds,
            stats=stats,
            log=plog,
        )
        if not any_valid:
            result.reason = (
                "无有效匹配（全员无 Homography/inlier）· " + result.reason
            )
        overlay = draw_match_overlay(qpack, rpack, stats)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{query_path.stem}__{rpath.stem}.jpg"
        cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        result.overlay_uri = str(overlay_path)
        pair_id = store.upsert_pair(result)
        pair_ids.append(pair_id)
        ratio_s = f"{score:.3f}" if score >= 0 else "n/a"
        print(
            f"  #{rank} 2017={rpath.name} inlier={ratio_s} "
            f"m/i={stats.num_matches}/{stats.num_inliers} "
            f"lap2018={result.lap_2018:.1f} lap2017={result.lap_2017:.1f} "
            f"tier={result.tier.value} · {result.reason}"
        )

    rows = [store.get_pair(i) for i in pair_ids]
    best = rows[0] if rows else None
    return {
        "skipped": False,
        "query_id": query_path.name,
        "pair_ids": pair_ids,
        "pairs": rows,
        "pipeline_steps": best.get("pipeline_steps") if best else qlog.to_list(),
        "pipeline_furthest": best.get("pipeline_furthest") if best else qlog.furthest,
        "reason": None if any_valid else "无有效匹配（全员无 Homography/inlier）",
        "no_valid_match": not any_valid,
        "gallery_pool": gallery.pool,
        "gallery_n": len(gallery.packs),
        "gallery_has_0752": "DJI_0752.JPG" in gallery.names,
        "cache_hit": gallery.cache_hit,
        "cache_miss": gallery.cache_miss,
    }
