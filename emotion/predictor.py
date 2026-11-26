"""Model loading + batched inference with flip test-time augmentation."""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

from .config import DEFAULT_MODEL, LEGACY_MODEL
from .preprocessing import prepare_face


def resolve_model_path(path: str | Path | None = None) -> Path:
    for p in ([Path(path)] if path else [DEFAULT_MODEL, LEGACY_MODEL]):
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Train one with `python train.py --data <dataset_dir>` "
        "or pass --model path/to/model.keras")


class EmotionPredictor:
    def __init__(self, model_path: str | Path | None = None, tta: bool = True):
        self.path = resolve_model_path(model_path)
        self.tta = tta
        # compile=False: no optimiser state needed, avoids the "compiled metrics" warnings
        self.model = keras.models.load_model(self.path, compile=False)
        # warm-up so the first live frame isn't slow
        self.model.predict(np.zeros((1, 48, 48, 1), "float32"), verbose=0)

    def predict_faces(self, gray_faces: list[np.ndarray]) -> np.ndarray:
        """list of gray crops -> (n, 7) probabilities. One batched call, not one per face."""
        prepared = [prepare_face(f) for f in gray_faces]
        ok = [i for i, p in enumerate(prepared) if p is not None]
        out = np.zeros((len(gray_faces), self.model.output_shape[-1]), "float32")
        if not ok:
            return out
        batch = np.stack([prepared[i] for i in ok])
        probs = np.asarray(self.model(batch, training=False))
        if self.tta:
            probs = (probs + np.asarray(self.model(batch[:, :, ::-1, :], training=False))) / 2
        out[ok] = probs
        return out
