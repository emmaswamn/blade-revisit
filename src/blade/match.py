from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from blade.types import MatchStats, Thresholds


@dataclass
class FeaturePack:
    image_id: str
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    scale: float  # original / resized


def _resize_max_side(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return gray, 1.0
    scale = long_side / max_side
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _make_detector(name: str):
    """Build a Feature2D detector.

    OpenCV 5.x pip wheels may omit AKAZE; contest brief prefers AKAZE then ORB.
    Resolve by availability, not by 4.x memory.
    """
    requested = (name or "orb").lower()
    builders = {
        "akaze": lambda: _create_if_present("AKAZE", nfeatures=None),
        "orb": lambda: cv2.ORB_create(nfeatures=4000),
        "sift": lambda: cv2.SIFT_create(nfeatures=4000),
    }
    order = [requested] + [k for k in ("akaze", "orb", "sift") if k != requested]
    last_err: Exception | None = None
    for key in order:
        factory = builders.get(key)
        if factory is None:
            continue
        try:
            det = factory()
            if det is not None:
                if key != requested:
                    print(f"[blade.match] detector '{requested}' unavailable; using '{key}'")
                return det
        except Exception as exc:  # noqa: BLE001 — probe optional factories
            last_err = exc
            continue
    raise RuntimeError(
        f"no usable Feature2D detector (wanted '{requested}'); last error: {last_err}"
    )


def _create_if_present(class_name: str, **kwargs):
    """cv2.AKAZE_create / cv2.AKAZE.create if the build includes it."""
    create_fn = getattr(cv2, f"{class_name}_create", None)
    if callable(create_fn):
        return create_fn() if not kwargs else create_fn(**{k: v for k, v in kwargs.items() if v is not None})
    cls = getattr(cv2, class_name, None)
    if cls is not None and hasattr(cls, "create"):
        # Most Feature2D types ignore unknown kwargs; keep call minimal.
        return cls.create()
    return None


def extract_features(
    image_id: str,
    bgr_or_gray: np.ndarray,
    thresholds: Thresholds,
    mask: np.ndarray | None = None,
) -> FeaturePack:
    """Detect features. If mask is given (full-res 0/255), only use those pixels."""
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    small, scale = _resize_max_side(gray, thresholds.max_side)
    mask_small = None
    if mask is not None:
        if mask.shape[:2] != gray.shape[:2]:
            raise ValueError(
                f"mask size {mask.shape[:2]} != image {gray.shape[:2]}"
            )
        if scale == 1.0:
            mask_small = mask
        else:
            mask_small = cv2.resize(
                mask,
                (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    detector = _make_detector(thresholds.detector)
    kps, desc = detector.detectAndCompute(small, mask_small)
    return FeaturePack(
        image_id=image_id,
        gray=small,
        keypoints=list(kps or []),
        descriptors=desc,
        scale=scale,
    )


def draw_keypoints_bgr(
    bgr: np.ndarray,
    pack: FeaturePack,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw FeaturePack keypoints (in resized coords) onto full-res BGR."""
    out = bgr.copy()
    scale = pack.scale
    for kp in pack.keypoints:
        x = int(round(kp.pt[0] * scale))
        y = int(round(kp.pt[1] * scale))
        cv2.circle(out, (x, y), 4, color, 1, lineType=cv2.LINE_AA)
    return out


def match_features(
    query: FeaturePack,
    ref: FeaturePack,
    thresholds: Thresholds,
) -> MatchStats:
    if (
        query.descriptors is None
        or ref.descriptors is None
        or len(query.keypoints) < 4
        or len(ref.keypoints) < 4
    ):
        return MatchStats(inlier_ratio=None, num_matches=0, num_inliers=0, H=None)

    # AKAZE/ORB are binary; use Hamming. Fall back to L2 if float desc appear.
    if query.descriptors.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    raw = matcher.knnMatch(query.descriptors, ref.descriptors, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < thresholds.ratio_test * n.distance:
            good.append(m)

    if len(good) < thresholds.min_matches:
        return MatchStats(
            inlier_ratio=None,
            num_matches=len(good),
            num_inliers=0,
            H=None,
        )

    src = np.float32([query.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.ransac_reproj_threshold,
    )
    if H is None or mask is None:
        return MatchStats(
            inlier_ratio=None,
            num_matches=len(good),
            num_inliers=0,
            H=None,
        )

    inliers = int(mask.ravel().sum())
    ratio = inliers / len(good) if good else None
    return MatchStats(
        inlier_ratio=ratio,
        num_matches=len(good),
        num_inliers=inliers,
        H=H,
    )


def draw_match_overlay(
    query: FeaturePack,
    ref: FeaturePack,
    stats: MatchStats,
    max_draw: int = 60,
) -> np.ndarray:
    """Side-by-side with a few inlier lines for the debug UI."""
    if stats.H is None or stats.num_inliers == 0:
        q = cv2.cvtColor(query.gray, cv2.COLOR_GRAY2BGR)
        r = cv2.cvtColor(ref.gray, cv2.COLOR_GRAY2BGR)
        h = max(q.shape[0], r.shape[0])
        canvas = np.zeros((h, q.shape[1] + r.shape[1], 3), dtype=np.uint8)
        canvas[: q.shape[0], : q.shape[1]] = q
        canvas[: r.shape[0], q.shape[1] :] = r
        return canvas

    # Re-run a light match just for drawing (same params as decision path).
    if query.descriptors.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = matcher.knnMatch(query.descriptors, ref.descriptors, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src = np.float32([query.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else [0] * len(good)

    # Keep only inliers and cap count for readability.
    inlier_matches = [m for m, keep in zip(good, matches_mask) if keep]
    inlier_matches = inlier_matches[:max_draw]
    mask_draw = [1] * len(inlier_matches)

    return cv2.drawMatches(
        query.gray,
        query.keypoints,
        ref.gray,
        ref.keypoints,
        inlier_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask_draw,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
