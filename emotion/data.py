"""tf.data input pipeline with augmentation, class weights and optional MixUp."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from .config import EMOTIONS, IMG_SIZE, NUM_CLASSES


def find_split(dataset_dir: Path, names: tuple[str, ...]) -> Path:
    for n in names:
        if (dataset_dir / n).is_dir():
            return dataset_dir / n
    # Kaggle layout sometimes has an extra "images" level
    for n in names:
        if (dataset_dir / "images" / n).is_dir():
            return dataset_dir / "images" / n
    raise FileNotFoundError(f"None of {names} found inside {dataset_dir}")


def _augmenter() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.25),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augment")


def _load(directory: Path, batch_size: int, shuffle: bool, validation_split=None, subset=None, seed=42):
    ds = keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="categorical", class_names=EMOTIONS,
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
        shuffle=shuffle, seed=seed, validation_split=validation_split, subset=subset)
    return ds


def class_counts(train_dir: Path) -> np.ndarray:
    return np.array([len(list((train_dir / e).glob("*"))) for e in EMOTIONS])


def class_weights(counts: np.ndarray) -> dict[int, float]:
    """Balanced weights, softened with a square root so the tiny 'disgust' class helps but doesn't dominate."""
    w = (counts.sum() / (len(counts) * counts)) ** 0.5
    return {i: float(v) for i, v in enumerate(w)}


def _mixup(alpha: float):
    def fn(x, y):
        bs = tf.shape(x)[0]
        lam = tf.random.uniform([], 0, 1)
        lam = tf.maximum(lam, 1 - lam) if alpha else 1.0
        idx = tf.random.shuffle(tf.range(bs))
        return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)
    return fn


def build_datasets(dataset_dir: str | Path, batch_size: int = 64, val_fraction: float = 0.1,
                   mixup: float = 0.0, seed: int = 42):
    """Returns train, val (held out from train, for model selection), test (the dataset's own
    validation/test folder, touched only for the final report) and the class-weight dict."""
    dataset_dir = Path(dataset_dir)
    train_dir = find_split(dataset_dir, ("train",))
    test_dir = find_split(dataset_dir, ("validation", "test"))
    train = _load(train_dir, batch_size, True, val_fraction, "training", seed)
    val = _load(train_dir, batch_size, False, val_fraction, "validation", seed)
    test = _load(test_dir, batch_size, False)
    aug = _augmenter()
    train = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    if mixup:
        train = train.map(_mixup(mixup), num_parallel_calls=tf.data.AUTOTUNE)
    prefetch = lambda d: d.cache().prefetch(tf.data.AUTOTUNE)
    return train.prefetch(tf.data.AUTOTUNE), prefetch(val), prefetch(test), class_weights(class_counts(train_dir))
