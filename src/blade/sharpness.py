from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(gray: np.ndarray, blur_laplacian_min: float) -> bool:
    return laplacian_variance(gray) >= blur_laplacian_min
