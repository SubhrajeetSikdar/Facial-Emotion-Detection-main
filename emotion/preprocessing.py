"""Image preprocessing shared by training, evaluation and inference.

Training and inference MUST use identical preprocessing. The original project trained on
raw 0-255 pixels (ImageDataGenerator() had no rescale) but divided by 255 at inference,
which silently wrecked accuracy. Here the model contains its own Rescaling layer, so every
caller passes plain float32 pixels in [0, 255] and the problem cannot come back.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import IMG_SIZE

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def prepare_face(gray_face: np.ndarray, size: int = IMG_SIZE, equalize: bool = False) -> np.ndarray | None:
    """Gray face crop -> float32 array (size, size, 1) in [0, 255], or None if unusable."""
    if gray_face is None or gray_face.size == 0 or min(gray_face.shape[:2]) < 8:
        return None
    if gray_face.ndim == 3:
        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if gray_face.shape[0] > size else cv2.INTER_CUBIC
    face = cv2.resize(gray_face, (size, size), interpolation=interp)
    if equalize:
        face = _CLAHE.apply(face)
    return face.astype("float32")[..., None]


def expand_box(x: int, y: int, w: int, h: int, frame_shape, margin: float = 0.1):
    """Grow a face box slightly (FER faces include forehead/chin) and clip to the frame."""
    fh, fw = frame_shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(fw, x + w + dx), min(fh, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0
