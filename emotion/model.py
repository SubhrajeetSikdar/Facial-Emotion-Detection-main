"""Model definition: a compact residual CNN with squeeze-excitation for 48x48 grayscale faces.

~1.5M parameters (the original plain CNN had 4.5M and overfit: 72% train / 61% val accuracy).
Input is raw pixels in [0, 255]; normalisation happens INSIDE the model.
"""
from __future__ import annotations

import keras
from keras import layers

from .config import IMG_SIZE, NUM_CLASSES


@keras.saving.register_keras_serializable(package="emotion")
class PerImageStandardize(layers.Layer):
    """Zero-mean / unit-variance per image -> robust to lighting and camera exposure."""

    def call(self, x):
        mean = keras.ops.mean(x, axis=(1, 2, 3), keepdims=True)
        var = keras.ops.var(x, axis=(1, 2, 3), keepdims=True)
        return (x - mean) / keras.ops.sqrt(var + 1e-6)

    def compute_output_shape(self, input_shape):
        return input_shape


def _se(x, ratio: int = 8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // ratio, 4), activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    return layers.Multiply()([x, layers.Reshape((1, 1, ch))(s)])


def _res_block(x, filters: int, downsample: bool, drop: float):
    stride = 2 if downsample else 1
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = _se(y)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([y, shortcut])
    y = layers.Activation("relu")(y)
    return layers.SpatialDropout2D(drop)(y) if drop else y


def build_model(num_classes: int = NUM_CLASSES, width: int = 32, dropout: float = 0.1) -> keras.Model:
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1), name="face")
    x = PerImageStandardize()(inp)
    x = layers.Conv2D(width, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    for filters, n_blocks in [(width * 2, 2), (width * 4, 2), (width * 8, 2), (width * 16, 1)]:
        for i in range(n_blocks):
            x = _res_block(x, filters, downsample=(i == 0), drop=dropout)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="emotion")(x)
    return keras.Model(inp, out, name="emotion_resnet_se")
