"""Face detection with a robust fallback chain.

Order of preference: OpenCV DNN-free YuNet (if the ONNX file is present) -> Haar cascade.
YuNet is far more accurate on tilted/small faces; Haar works with no extra download.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CASCADE_PATH, MODELS_DIR

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    def __init__(self, min_face: int = 60, scale_factor: float = 1.1, min_neighbors: int = 6):
        self.min_face = min_face
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self._yunet = None
        self._haar = None
        self._size = None
        if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
            except cv2.error:
                self._yunet = None
        if self._yunet is None:
            if not Path(CASCADE_PATH).exists():
                raise FileNotFoundError(f"Haar cascade not found at {CASCADE_PATH}")
            self._haar = cv2.CascadeClassifier(str(CASCADE_PATH))
            if self._haar.empty():
                raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")

    @property
    def backend(self) -> str:
        return "yunet" if self._yunet is not None else "haar"

    def detect(self, frame_bgr: np.ndarray, gray: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
        if self._yunet is not None:
            h, w = frame_bgr.shape[:2]
            if self._size != (w, h):
                self._yunet.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._yunet.detect(frame_bgr)
            if faces is None:
                return []
            return [tuple(int(v) for v in f[:4]) for f in faces if min(f[2], f[3]) >= self.min_face // 2]
        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(self.min_face, self.min_face), flags=cv2.CASCADE_SCALE_IMAGE)
        return [tuple(int(v) for v in f) for f in faces]
