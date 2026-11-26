"""Train the emotion model.

    python train.py --data /path/to/face-expression-recognition-dataset/images
    python train.py --data ... --epochs 120 --batch-size 64 --mixup 0.2
"""
from __future__ import annotations

import argparse
import json

import keras
import numpy as np
import tensorflow as tf

from emotion.config import DEFAULT_MODEL, EMOTIONS, OUTPUT_DIR
from emotion.data import build_datasets
from emotion.evaluation import evaluate_model, plot_history
from emotion.model import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset dir containing train/ and validation/ (or test/)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 disables); 0.2 often helps")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(DEFAULT_MODEL))
    p.add_argument("--mixed-precision", action="store_true", help="faster on modern GPUs")
    return p.parse_args()


def main():
    a = parse_args()
    keras.utils.set_random_seed(a.seed)
    if a.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    OUTPUT_DIR.mkdir(exist_ok=True)

    train, val, test, cw = build_datasets(a.data, a.batch_size, mixup=a.mixup, seed=a.seed)
    print("Class weights:", {EMOTIONS[k]: round(v, 2) for k, v in cw.items()})

    model = build_model(width=a.width)
    steps = tf.data.experimental.cardinality(train).numpy() * a.epochs
    # cosine decay with warm-up: replaces the original fragile fixed LR + ReduceLROnPlateau
    schedule = keras.optimizers.schedules.CosineDecay(
        0.0, int(steps), warmup_target=a.lr, warmup_steps=int(steps * 0.05), alpha=0.01)
    opt = keras.optimizers.AdamW(schedule, weight_decay=a.weight_decay, use_ema=True, ema_momentum=0.99)
    # label smoothing; with mixup the labels are already soft, which is fine
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=a.label_smoothing)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.summary()

    callbacks = [
        # BUG FIX: original monitored 'val_acc' which doesn't exist in modern Keras, so nothing was ever saved.
        keras.callbacks.ModelCheckpoint(a.out, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_log.csv")),
    ]
    history = model.fit(train, validation_data=val, epochs=a.epochs, class_weight=cw if not a.mixup else None,
                        callbacks=callbacks)
    plot_history(history.history, OUTPUT_DIR / "training_curves.png")

    best = keras.models.load_model(a.out, compile=False)
    report = evaluate_model(best, test, OUTPUT_DIR, tta=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nFinal TEST accuracy (flip-TTA): {report['accuracy']:.4f}  macro-F1: {report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
