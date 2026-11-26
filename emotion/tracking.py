"""Lightweight face tracker + temporal smoothing (removes label flicker)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import EMOTIONS


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    tid: int
    box: tuple
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    missed: int = 0

    def update(self, box, probs):
        self.box = box
        self.history.append(probs)
        self.missed = 0

    def smoothed(self) -> np.ndarray:
        """Exponentially weighted mean of recent probability vectors."""
        h = np.array(self.history)
        w = np.exp(np.linspace(-1.5, 0, len(h)))
        return (h * w[:, None]).sum(0) / w.sum()


class FaceTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 8):
        self.iou_thresh, self.max_missed = iou_thresh, max_missed
        self.tracks: dict[int, Track] = {}
        self._next = 0

    def assign(self, boxes) -> list[Track]:
        """Match detections to existing tracks greedily by IoU; create new tracks as needed."""
        result, used = [], set()
        for box in boxes:
            best, best_iou = None, self.iou_thresh
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                v = iou(box, tr.box)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = Track(best, box)
            used.add(best)
            result.append(self.tracks[best])
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        return result


def label_of(probs: np.ndarray) -> tuple[str, float]:
    i = int(np.argmax(probs))
    return EMOTIONS[i], float(probs[i])
